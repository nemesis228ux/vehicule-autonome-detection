import tempfile
import unittest
import argparse
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from src.detection_app.runtime import _static_output_path
from src.video_reader import VideoReader


class FakeCapture:
    def __init__(self, frames, opened=True):
        self.frames = iter(frames)
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened and not self.released

    def get(self, property_id):
        return {
            5: 30.0,
            3: 640.0,
            4: 480.0,
            7: 2.0,
        }.get(property_id, 0.0)

    def read(self):
        return next(self.frames)

    def release(self):
        self.released = True


class VideoReaderTests(unittest.TestCase):
    def test_video_iteration_stops_at_end_and_releases_capture(self):
        frame = object()
        capture = FakeCapture([(True, frame), (False, None)])

        with tempfile.NamedTemporaryFile(suffix=".mp4") as video_file:
            with patch("src.video_reader.cv2.VideoCapture", return_value=capture):
                reader = VideoReader(Path(video_file.name))
                frames = list(reader)

        self.assertEqual(len(frames), 1)
        self.assertEqual(reader.fps, 30.0)
        self.assertEqual((reader.frame_width, reader.frame_height), (640, 480))
        self.assertTrue(capture.released)

    def test_camera_read_failure_raises(self):
        capture = FakeCapture([(False, None)])
        with patch("src.video_reader.cv2.VideoCapture", return_value=capture):
            reader = VideoReader(0)
            with self.assertRaises(OSError):
                reader.read()
            reader.release()

    def test_unavailable_source_raises_and_releases_capture(self):
        capture = FakeCapture([], opened=False)
        with patch("src.video_reader.cv2.VideoCapture", return_value=capture):
            with self.assertRaises(OSError):
                VideoReader(0)
        self.assertTrue(capture.released)

    def test_missing_video_file_raises_before_opening_capture(self):
        with patch("src.video_reader.cv2.VideoCapture") as open_capture:
            with self.assertRaises(FileNotFoundError):
                VideoReader("fichier-inexistant.mp4")
        open_capture.assert_not_called()

    def test_png_file_is_read_once_as_a_still_image(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "test.png"
            cv2.imwrite(str(image_path), np.zeros((60, 80, 3), dtype=np.uint8))

            with patch("src.video_reader.cv2.VideoCapture") as open_capture:
                reader = VideoReader(image_path)
                frames = list(reader)

        open_capture.assert_not_called()
        self.assertEqual(len(frames), 1)
        self.assertEqual((reader.frame_width, reader.frame_height), (80, 60))
        self.assertEqual(reader.frame_count, 1)
        self.assertEqual(reader.fps, 0.0)

    def test_gif_file_is_read_as_animated_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            gif_path = Path(directory) / "test.gif"
            frames = [
                Image.fromarray(np.zeros((40, 50, 3), dtype=np.uint8)),
                Image.fromarray(np.full((40, 50, 3), 255, dtype=np.uint8)),
            ]
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=100,
                loop=0,
            )

            reader = VideoReader(gif_path)
            gif_frames = list(reader)

        self.assertEqual(len(gif_frames), 2)
        self.assertEqual(reader.frame_count, 2)
        self.assertEqual((reader.frame_width, reader.frame_height), (50, 40))

    def test_static_output_path_uses_jpg_in_outputs(self):
        args = argparse.Namespace(save=True, source=r"data\images\test.png")
        reader = type("StaticReader", (), {"is_live": False, "fps": 0.0})()

        output_path = _static_output_path(
            args,
            reader,
            alert_manager=None,
            risk_evaluator=None,
            tracker=None,
        )

        self.assertEqual(output_path.name, "test_detecte.jpg")


if __name__ == "__main__":
    unittest.main()
