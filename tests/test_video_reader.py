import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
