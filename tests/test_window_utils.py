import unittest

from src.window_utils import compute_window_size


class WindowSizeTests(unittest.TestCase):
    def test_portrait_video_keeps_aspect_ratio(self):
        self.assertEqual(compute_window_size(1080, 1920), (405, 720))

    def test_landscape_video_keeps_aspect_ratio(self):
        self.assertEqual(compute_window_size(1920, 1080), (1280, 720))

    def test_small_video_is_not_upscaled(self):
        self.assertEqual(compute_window_size(240, 426), (240, 426))

    def test_upscale_can_be_enabled_explicitly(self):
        self.assertEqual(
            compute_window_size(100, 100, max_upscale=2.0), (200, 200)
        )

    def test_invalid_dimensions_raise(self):
        with self.assertRaises(ValueError):
            compute_window_size(0, 100)
        with self.assertRaises(ValueError):
            compute_window_size(100, -1)
        with self.assertRaises(ValueError):
            compute_window_size(100, 100, max_upscale=0)


if __name__ == "__main__":
    unittest.main()
