"""Démonstration manuelle du lecteur vidéo VideoReader."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import cv2

try:
    from src.video_reader import VideoReader
    from src.window_utils import fit_window, is_window_open
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.video_reader import VideoReader
    from src.window_utils import fit_window, is_window_open


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Affiche une vidéo, un flux ou une caméra via VideoReader."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Fichier vidéo, URL de flux ou identifiant de caméra (défaut : 0).",
    )
    parser.add_argument(
        "--window",
        default="Test du lecteur video",
        help="Nom ASCII de la fenêtre OpenCV (éviter les accents).",
    )
    parser.add_argument(
        "--show-fps",
        action="store_true",
        help="Affiche les FPS mesurés dans le coin de l'image.",
    )
    return parser.parse_args(argv)


def _update_fps(frame, previous_time: float, smoothed_fps: float, show_fps: bool):
    now = time.perf_counter()
    elapsed = now - previous_time
    if elapsed > 0:
        instant_fps = 1.0 / elapsed
        smoothed_fps = (
            instant_fps
            if smoothed_fps == 0
            else 0.9 * smoothed_fps + 0.1 * instant_fps
        )

    if show_fps:
        cv2.putText(
            frame,
            f"FPS {smoothed_fps:5.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return now, smoothed_fps


def _play_frames(reader, window: str, delay: int, show_fps: bool) -> None:
    smoothed_fps = 0.0
    previous_time = time.perf_counter()
    window_seen = False

    for frame in reader:
        previous_time, smoothed_fps = _update_fps(
            frame, previous_time, smoothed_fps, show_fps
        )
        cv2.imshow(window, frame)
        key = cv2.waitKey(delay) & 0xFF
        if key in (ord("q"), 27):
            break

        if is_window_open(window):
            window_seen = True
        elif window_seen:
            break


def run(source: str, window: str, show_fps: bool) -> int:
    delay = 1
    try:
        with VideoReader(source) as reader:
            print(f"Source       : {source}")
            print(f"Type         : {'direct' if reader.is_live else 'fichier'}")
            print(f"Résolution   : {reader.frame_width}x{reader.frame_height}")
            if not reader.is_live and reader.fps > 0:
                delay = max(1, round(1000 / reader.fps))
                print(f"FPS source   : {reader.fps:.1f}")

            print("Appuyez sur q ou Échap pour quitter.")
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            if reader.frame_width > 0 and reader.frame_height > 0:
                fit_window(window, reader.frame_width, reader.frame_height)

            try:
                _play_frames(reader, window, delay, show_fps)
            except OSError as exc:
                print(f"Lecture interrompue : {exc}", file=sys.stderr)
                return 1
    finally:
        cv2.destroyAllWindows()

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return run(args.source, args.window, args.show_fps)
    except (OSError, ValueError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
