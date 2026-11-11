"""Frame rendering, keyboard handling, and run statistics."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2

from ..alerts import AlertManager, draw_alert_overlay
from ..detector import draw_detections
from ..hud import draw_risk_hud
from ..risk_evaluator import DANGER_LABELS, DangerLevel
from ..tracker import draw_tracks


@dataclass
class RunStats:
    tracked_ids: set[int] = field(default_factory=set)
    highest_level: DangerLevel = DangerLevel.INFORMATION
    critical_frames: int = 0
    alert_events: int = 0
    critical_alerts: int = 0


def update_fps(now: float, previous_time: float, smoothed_fps: float) -> float:
    elapsed = now - previous_time
    if elapsed > 0:
        instant_fps = 1.0 / elapsed
        return (
            instant_fps
            if smoothed_fps == 0
            else 0.9 * smoothed_fps + 0.1 * instant_fps
        )
    return smoothed_fps


def render_frame(
    frame,
    objects,
    detections,
    risk_evaluator,
    alert_manager,
    args,
    stats: RunStats,
    smoothed_fps: float,
    frame_count: int,
    now: float,
    source_fps: float,
) -> None:
    if objects is not None and risk_evaluator is not None:
        summary = risk_evaluator.evaluate(objects, frame.shape, source_fps or 25.0)
        stats.highest_level = max(stats.highest_level, summary.level)
        if summary.level == DangerLevel.CRITIQUE:
            stats.critical_frames += 1
        alert_event = None
        if alert_manager is not None:
            alert_event = alert_manager.update(
                summary, now_s=now, frame_index=frame_count
            )
            if alert_event.changed and alert_event.level != DangerLevel.INFORMATION:
                stats.alert_events += 1
            if alert_event.changed and alert_event.level == DangerLevel.CRITIQUE:
                stats.critical_alerts += 1
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


def handle_frame_key(key: int, alert_manager: Optional[AlertManager]) -> bool:
    if key in (ord("q"), 27):
        return True
    if key in (ord("m"), ord("M")) and alert_manager is not None:
        muted = alert_manager.toggle_mute()
        print(f"Son : {'coupé' if muted else 'actif'}")
    return False


def print_summary(
    frame_count, start_time, tracker, risk_evaluator, alert_manager, stats: RunStats
) -> None:
    print(
        f"{frame_count} frames traités en "
        f"{time.perf_counter() - start_time:.1f}s"
    )
    if tracker is not None:
        print(f"IDs suivis : {len(stats.tracked_ids)}")
    if risk_evaluator is not None:
        print(f"Niveau max : {DANGER_LABELS[stats.highest_level]}")
        print(f"Frames critiques : {stats.critical_frames}")
    if alert_manager is not None:
        print(f"Alertes : {stats.alert_events} (critiques: {stats.critical_alerts})")
        if alert_manager.muted:
            sound_state = "coupé"
        elif alert_manager.sound_available:
            sound_state = "actif"
        else:
            sound_state = "indisponible"
        print(f"Son : {sound_state}")
