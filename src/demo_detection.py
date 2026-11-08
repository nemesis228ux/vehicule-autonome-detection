"""Démonstration de la détection YOLOv8 sur une source vidéo."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import cv2

try:
    from src.detector import Detector, draw_detections
    from src.video_reader import VideoReader
    from src.window_utils import fit_window, is_window_open
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.detector import Detector, draw_detections
    from src.video_reader import VideoReader
    from src.window_utils import fit_window, is_window_open


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Détecte les objets routiers d'une vidéo, d'un flux ou d'une caméra."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Fichier vidéo, URL de flux ou identifiant de caméra (défaut : 0).",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Modèle Ultralytics (yolov8n/s/m/l/x.pt) ou chemin vers un .pt (défaut : yolov8n.pt).",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Seuil de confiance.")
    parser.add_argument("--iou", type=float, default=0.45, help="Seuil IoU du NMS.")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Taille d'inférence (640 par défaut ; 320 plus rapide sur CPU).",
    )
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Conserve aussi les classes hors route (affichées en 'autre').",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Enregistre la vidéo annotée dans outputs/.",
    )
    parser.add_argument(
        "--show-fps", action="store_true", help="Affiche les FPS mesurés."
    )
    parser.add_argument(
        "--window",
        default="Detection YOLOv8",
        help="Nom ASCII de la fenêtre OpenCV (éviter les accents).",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    writer: Optional[cv2.VideoWriter] = None
    output_path: Optional[Path] = None
    delay = 1
    smoothed_fps = 0.0
    frame_count = 0
    window_seen = False
    previous_time = time.perf_counter()
    start_time = previous_time

    try:
        detector = Detector(
            model_name=args.model,
            confidence_threshold=args.conf,
            iou_threshold=args.iou,
            imgsz=args.imgsz,
            road_only=not args.all_classes,
        )

        with VideoReader(args.source) as reader:
            print(f"Source     : {args.source}")
            print(f"Modèle     : {args.model}")
            print(f"Résolution : {reader.frame_width}x{reader.frame_height}")
            if not reader.is_live and reader.fps > 0:
                delay = max(1, round(1000 / reader.fps))
                print(f"FPS source : {reader.fps:.1f}")

            print("Appuyez sur q ou Échap pour quitter.")
            cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)
            if reader.frame_width > 0 and reader.frame_height > 0:
                fit_window(
                    args.window, reader.frame_width, reader.frame_height
                )

            if args.save:
                output_dir = Path("outputs")
                output_dir.mkdir(parents=True, exist_ok=True)
                source_name = (
                    Path(str(args.source)).stem if not reader.is_live else "direct"
                )
                output_path = output_dir / f"{source_name}_detecte.mp4"
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    reader.fps if reader.fps > 0 else 25.0,
                    (reader.frame_width, reader.frame_height),
                )
                if not writer.isOpened():
                    raise OSError(
                        f"Impossible de créer la vidéo de sortie : {output_path}"
                    )

            for frame in reader:
                frame_count += 1
                detections = detector.detect(frame)
                draw_detections(frame, detections)

                now = time.perf_counter()
                elapsed = now - previous_time
                previous_time = now
                if elapsed > 0:
                    instant_fps = 1.0 / elapsed
                    smoothed_fps = (
                        instant_fps
                        if smoothed_fps == 0
                        else 0.9 * smoothed_fps + 0.1 * instant_fps
                    )

                if args.show_fps:
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

                if writer is not None:
                    writer.write(frame)

                cv2.imshow(args.window, frame)
                key = cv2.waitKey(delay) & 0xFF
                if key in (ord("q"), 27):
                    break

                if is_window_open(args.window):
                    window_seen = True
                elif window_seen:
                    break

            print(
                f"{frame_count} frames traités en "
                f"{time.perf_counter() - start_time:.1f}s"
            )
    except (ImportError, OSError, ValueError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if output_path is not None:
        print(f"Vidéo enregistrée : {output_path}")

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
