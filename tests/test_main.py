import unittest
from unittest.mock import patch

from src.main import main


class MainTests(unittest.TestCase):
    def test_main_forwards_arguments_to_detection_entrypoint(self):
        arguments = ["--source", "http://phone.local:8080/video"]
        with patch("src.main.run_detection", return_value=0) as run_detection:
            result = main(arguments)

        self.assertEqual(result, 0)
        run_detection.assert_called_once_with(arguments)


if __name__ == "__main__":
    unittest.main()
