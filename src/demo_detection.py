"""Démonstration de la détection YOLOv8 sur une source vidéo."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import cv2

try:
    from src.alerts import (
        AlertManager,
        NullSoundBackend,
        WinsoundBackend,
        draw_alert_overlay,
    )
    from src.detector import Detector, draw_detections
    from src.hud import draw_risk_hud
    from src.risk_evaluator import DANGER_LABELS, DangerLevel, RiskEvaluator
    from src.tracker import Tracker, draw_tracks
    from src.video_reader import VideoReader
    from src.window_utils import fit_window, is_window_open
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.alerts import (
        AlertManager,
        NullSoundBackend,
        WinsoundBackend,
        draw_alert_overlay,
    )
    from src.detector import Detector, draw_detections
    from src.hud import draw_risk_hud
    from src.risk_evaluator import DANGER_LABELS, DangerLevel, RiskEvaluator
    from src.tracker import Tracker, draw_tracks
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
        "--track",
        action="store_true",
        help="Active le suivi ByteTrack et affiche les IDs.",
    )
    parser.add_argument(
        "--risk",
        action="store_true",
        help="Active l'évaluation du danger et le HUD (implique --track).",
    )
    parser.add_argument(
        "--alerts",
        action="store_true",
        help="Active les alertes visuelles et sonores (implique --risk).",
    )
    parser.add_argument(
        "--no-sound",
        action="store_true",
        help="Désactive le klaxon et conserve les alertes visuelles.",
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Configuration Ultralytics du tracker (bytetrack.yaml ou botsort.yaml).",
    )
    parser.add_argument(
        "--labels",
        choices=("full", "id", "none"),
        default="id",
        help="Étiquettes de tracking : full (ID+catégorie+conf), id seul, ou aucune.",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=30,
        help="Nombre de points de trajectoire conservés par objet.",
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="Frames sans détection avant purge de l'état local d'un ID.",
    )
    parser.add_argument(
        "--window",
        default="Detection YOLOv8",
        help="Nom ASCII de la fenêtre OpenCV (éviter les accents).",
    )
    args = parser.parse_args(argv)
    if args.alerts:
        args.risk = True
    if args.risk:
        args.track = True
    return args


def run(args: argparse.Namespace) -> int:
    writer: Optional[cv2.VideoWriter] = None
    output_path: Optional[Path] = None
    tracker: Optional[Tracker] = None
    risk_evaluator: Optional[RiskEvaluator] = None
    alert_manager: Optional[AlertManager] = None
    tracked_ids: set[int] = set()
    highest_level = DangerLevel.INFORMATION
    critical_frames = 0
    alert_events = 0
    critical_alerts = 0
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

        if args.track:
            tracker = Tracker(
                detector,
                tracker_config=args.tracker,
                trail_length=args.trail_length,
                max_age=args.max_age,
            )
        if args.risk:
            risk_evaluator = RiskEvaluator()
        if args.alerts:
            sound_backend = (
                NullSoundBackend() if args.no_sound else WinsoundBackend()
            )
            alert_manager = AlertManager(sound_backend=sound_backend)

        with VideoReader(args.source) as reader:
            print(f"Source     : {args.source}")
            print(f"Modèle     : {args.model}")
            if tracker is not None:
                print(f"Tracking   : {args.tracker}")
            if risk_evaluator is not None:
                print("Danger     : actif")
            if alert_manager is not None:
                if args.no_sound:
                    sound_state = "coupé"
                elif alert_manager.sound_available:
                    sound_state = "klaxon actif"
                else:
                    sound_state = "indisponible"
                print(f"Alertes    : actives (son {sound_state})")
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
                if alert_manager is not None:
                    suffix = "_alertes"
                elif risk_evaluator is not None:
                    suffix = "_risque"
                elif tracker is not None:
                    suffix = "_tracke"
                else:
                    suffix = "_detecte"
                output_path = output_dir / f"{source_name}{suffix}.mp4"
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
                objects = None
                detections = None
                if tracker is not None:
                    objects = tracker.update(frame)
                    tracked_ids.update(obj.track_id for obj in objects)
                else:
                    detections = detector.detect(frame)

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

                if objects is not None and risk_evaluator is not None:
                    summary = risk_evaluator.evaluate(
                        objects, frame.shape, reader.fps or 25.0
                    )
                    if summary.level > highest_level:
                        highest_level = summary.level
                    if summary.level == DangerLevel.CRITIQUE:
                        critical_frames += 1
                    alert_event = None
                    if alert_manager is not None:
                        alert_event = alert_manager.update(
                            summary, now_s=now, frame_index=frame_count
                        )
                        if (
                            alert_event.changed
                            and alert_event.level
                            != DangerLevel.INFORMATION
                        ):
                            alert_events += 1
                        if (
                            alert_event.changed
                            and alert_event.level == DangerLevel.CRITIQUE
                        ):
                            critical_alerts += 1
                    draw_risk_hud(
                        frame,
                        objects,
                        summary,
                        fps=smoothed_fps if args.show_fps else None,
                        frame_index=frame_count,
                    )
                    if alert_event is not None:
                        draw_alert_overlay(frame, alert_event)
                elif objects is not None:
                    draw_tracks(frame, objects, labels=args.labels)
                else:
                    draw_detections(frame, detections)

                if args.show_fps and risk_evaluator is None:
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
                if key in (ord("m"), ord("M")) and alert_manager is not None:
                    muted = alert_manager.toggle_mute()
                    print(f"Son : {'coupé' if muted else 'actif'}")

                if is_window_open(args.window):
                    window_seen = True
                elif window_seen:
                    break

            print(
                f"{frame_count} frames traités en "
                f"{time.perf_counter() - start_time:.1f}s"
            )
            if tracker is not None:
                print(f"IDs suivis : {len(tracked_ids)}")
            if risk_evaluator is not None:
                print(f"Niveau max : {DANGER_LABELS[highest_level]}")
                print(f"Frames critiques : {critical_frames}")
            if alert_manager is not None:
                print(f"Alertes : {alert_events} (critiques: {critical_alerts})")
                if alert_manager.muted:
                    sound_state = "coupé"
                elif alert_manager.sound_available:
                    sound_state = "actif"
                else:
                    sound_state = "indisponible"
                print(f"Son : {sound_state}")
    except (ImportError, OSError, ValueError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
    finally:
        if alert_manager is not None:
            alert_manager.close()
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
