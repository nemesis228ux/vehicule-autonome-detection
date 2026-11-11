"""Runtime orchestration for video detection."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2

from ..alerts import AlertManager, NullSoundBackend, WinsoundBackend
from ..detector import Detector
from ..risk_evaluator import RiskEvaluator
from ..tracker import Tracker
from ..video_reader import VideoReader
from ..window_utils import fit_window, is_window_open
from .rendering import (
    RunStats,
    handle_frame_key,
    print_summary,
    render_frame,
    update_fps,
)


def _create_components(args: argparse.Namespace):
    detector = Detector(
        model_name=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
        road_only=not args.all_classes,
    )
    tracker = (
        Tracker(
            detector,
            tracker_config=args.tracker,
            trail_length=args.trail_length,
            max_age=args.max_age,
        )
        if args.track
        else None
    )
    risk_evaluator = RiskEvaluator() if args.risk else None
    alert_manager = None
    if args.alerts:
        backend = NullSoundBackend() if args.no_sound else WinsoundBackend()
        alert_manager = AlertManager(sound_backend=backend)
    return detector, tracker, risk_evaluator, alert_manager


def _print_reader_info(args, reader, tracker, risk_evaluator, alert_manager) -> int:
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
    delay = 1
    if not reader.is_live and reader.fps > 0:
        delay = max(1, round(1000 / reader.fps))
        print(f"FPS source : {reader.fps:.1f}")
    return delay


def _create_writer(args, reader, alert_manager, risk_evaluator, tracker):
    if not args.save:
        return None, None
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_name = Path(str(args.source)).stem if not reader.is_live else "direct"
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
        raise OSError(f"Impossible de créer la vidéo de sortie : {output_path}")
    return writer, output_path


def run(args: argparse.Namespace) -> int:
    writer: Optional[cv2.VideoWriter] = None
    output_path: Optional[Path] = None
    tracker: Optional[Tracker] = None
    risk_evaluator: Optional[RiskEvaluator] = None
    alert_manager: Optional[AlertManager] = None
    stats = RunStats()
    smoothed_fps = 0.0
    frame_count = 0
    window_seen = False
    start_time = time.perf_counter()
    previous_time = start_time

    try:
        detector, tracker, risk_evaluator, alert_manager = _create_components(args)
        with VideoReader(args.source) as reader:
            delay = _print_reader_info(
                args, reader, tracker, risk_evaluator, alert_manager
            )
            print("Appuyez sur q ou Échap pour quitter.")
            cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)
            if reader.frame_width > 0 and reader.frame_height > 0:
                fit_window(args.window, reader.frame_width, reader.frame_height)
            writer, output_path = _create_writer(
                args, reader, alert_manager, risk_evaluator, tracker
            )

            for frame in reader:
                frame_count += 1
                objects = tracker.update(frame) if tracker is not None else None
                detections = None if tracker is not None else detector.detect(frame)
                if objects is not None:
                    stats.tracked_ids.update(obj.track_id for obj in objects)
                now = time.perf_counter()
                smoothed_fps = update_fps(now, previous_time, smoothed_fps)
                previous_time = now
                render_frame(
                    frame,
                    objects,
                    detections,
                    risk_evaluator,
                    alert_manager,
                    args,
                    stats,
                    smoothed_fps,
                    frame_count,
                    now,
                    reader.fps,
                )
                if writer is not None:
                    writer.write(frame)
                cv2.imshow(args.window, frame)
                key = cv2.waitKey(delay) & 0xFF
                if handle_frame_key(key, alert_manager):
                    break
                if is_window_open(args.window):
                    window_seen = True
                elif window_seen:
                    break

            print_summary(
                frame_count, start_time, tracker, risk_evaluator, alert_manager, stats
            )
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
