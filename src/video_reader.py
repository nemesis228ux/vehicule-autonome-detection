"""Video and camera input utilities."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Iterator, Optional, Union
from urllib.parse import urlparse

import cv2
import numpy as np

VideoSource = Union[int, str, PathLike[str]]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ANIMATED_IMAGE_SUFFIXES = {".gif"}


class VideoReader:
    """Read frames from a video file, network stream, or camera."""

    def __init__(self, source: VideoSource) -> None:
        self.source = source
        self._capture_source, self.is_live = self._normalize_source(source)
        self._still_frames: list[np.ndarray] = []
        self._still_frame_index = 0
        self._capture = None

        source_path = (
            Path(str(self._capture_source)).expanduser()
            if isinstance(self._capture_source, str)
            else None
        )
        if source_path is not None and source_path.suffix.lower() in (
            IMAGE_SUFFIXES | ANIMATED_IMAGE_SUFFIXES
        ):
            self._load_image_source(source_path)
            return

        self._capture = cv2.VideoCapture(self._capture_source)

        if not self._capture.isOpened():
            self._capture.release()
            raise OSError(f"Impossible d'ouvrir la source vidéo : {source!s}")

        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

    @staticmethod
    def _normalize_source(source: VideoSource) -> tuple[int | str, bool]:
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

    def _load_image_source(self, path: Path) -> None:
        if path.suffix.lower() == ".gif":
            self._load_gif(path)
            return

        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise OSError(f"Impossible de lire l'image : {path}")
        self._still_frames = [frame]
        self.fps = 0.0
        self.frame_height, self.frame_width = frame.shape[:2]
        self.frame_count = 1

    def _load_gif(self, path: Path) -> None:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow est requis pour lire les fichiers GIF.") from exc

        try:
            with Image.open(path) as gif:
                duration = float(gif.info.get("duration", 100) or 100)
                for frame_index in range(gif.n_frames):
                    gif.seek(frame_index)
                    rgb = gif.convert("RGB")
                    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
                    self._still_frames.append(bgr)
        except (OSError, ValueError) as exc:
            raise OSError(f"Impossible de lire le GIF : {path}") from exc

        if not self._still_frames:
            raise OSError(f"Le GIF ne contient aucune frame : {path}")

        first = self._still_frames[0]
        self.fps = max(1.0, 1000.0 / duration)
        self.frame_height, self.frame_width = first.shape[:2]
        self.frame_count = len(self._still_frames)

    def read(self) -> Optional[np.ndarray]:
        """Return the next BGR frame, or None when a video file reaches EOF."""
        if self._still_frames:
            if self._still_frame_index >= len(self._still_frames):
                return None
            frame = self._still_frames[self._still_frame_index]
            self._still_frame_index += 1
            return frame

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
        if not self._still_frames and (
            self._capture is None or not self._capture.isOpened()
        ):
            raise RuntimeError("La source vidéo est fermée.")
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.release()

    def release(self) -> None:
        """Release the underlying camera or video handle."""
        if self._capture is not None:
            self._capture.release()