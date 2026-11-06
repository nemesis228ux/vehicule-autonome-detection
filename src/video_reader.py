"""Video and camera input utilities."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Iterator, Optional, Union
from urllib.parse import urlparse

import cv2
import numpy as np

VideoSource = Union[int, str, PathLike[str]]


class VideoReader:
    """Read frames from a video file, network stream, or camera."""

    def __init__(self, source: VideoSource) -> None:
        self.source = source
        self._capture_source, self.is_live = self._normalize_source(source)
        self._capture = cv2.VideoCapture(self._capture_source)

        if not self._capture.isOpened():
            self._capture.release()
            raise OSError(f"Impossible d'ouvrir la source vidéo : {source!s}")

        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

    @staticmethod
    def _normalize_source(source: VideoSource) -> tuple[Union[int, str], bool]:
        if isinstance(source, bool):
            raise ValueError("L'identifiant de caméra doit être un entier positif.")

        if isinstance(source, int):
            if source < 0:
                raise ValueError("L'identifiant de caméra doit être un entier positif.")
            return source, True

        source_text = str(source).strip()
        if not source_text:
            raise ValueError("La source vidéo ne peut pas être vide.")

        if source_text.isdecimal():
            return int(source_text), True

        parsed_source = urlparse(source_text)
        if "://" in source_text or parsed_source.scheme.lower() == "file":
            return source_text, True

        path = Path(source_text).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Fichier vidéo introuvable : {path}")
        return str(path), False

    def read(self) -> Optional[np.ndarray]:
        """Return the next BGR frame, or None when a video file reaches EOF."""
        if self._capture is None or not self._capture.isOpened():
            raise RuntimeError("La source vidéo est fermée.")

        success, frame = self._capture.read()
        if success and frame is not None:
            return frame

        if self.is_live:
            raise OSError(f"La lecture de la source en direct a échoué : {self.source!s}")
        return None

    def __iter__(self) -> Iterator[np.ndarray]:
        try:
            while True:
                frame = self.read()
                if frame is None:
                    return
                yield frame
        finally:
            self.release()

    def __enter__(self) -> VideoReader:
        if self._capture is None or not self._capture.isOpened():
            raise RuntimeError("La source vidéo est fermée.")
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.release()

    def release(self) -> None:
        """Release the underlying camera or video handle."""
        if self._capture is not None:
            self._capture.release()