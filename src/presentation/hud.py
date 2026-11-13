"""Reusable risk HUD rendering."""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

try:
    from src.risk.models import (
        DANGER_COLORS,
        DANGER_LABELS,
        DangerLevel,
        RiskAssessment,
        RiskSummary,
    )
    from src.tracker import TrackedObject, image_motion_label
except ModuleNotFoundError:
    from risk.models import (
        DANGER_COLORS,
        DANGER_LABELS,
        DangerLevel,
        RiskAssessment,
        RiskSummary,
    )
    from tracker import TrackedObject, image_motion_label

from .hud_components import (
    _blend_rect,
    _draw_panel,
    _fit_text,
)
from .labels import CATEGORY_LABELS


def draw_risk_hud(
    frame: np.ndarray,
    objects: Sequence[TrackedObject],
    summary: RiskSummary,
    fps: Optional[float] = None,
    frame_index: int = 0,
) -> np.ndarray:
    """Dessine le HUD de danger sur la frame (modifiée sur place)."""
    if frame is None or frame.size == 0:
        raise ValueError("La frame est vide.")

    _, frame_width = frame.shape[:2]
    scale = max(0.4, min(0.8, frame_width / 1280.0))
    compact = frame_width < 480
    objects_by_id = {obj.track_id: obj for obj in objects}
    movement_by_id = {
        obj.track_id: image_motion_label(
            obj, frame.shape, compact=compact
        )
        for obj in objects
    }

    for assessment in summary.assessments:
        _draw_assessment(
            frame,
            assessment,
            objects_by_id.get(assessment.track_id),
            movement_by_id.get(assessment.track_id, "--"),
            scale,
            compact,
            frame_index,
        )

    _draw_banner(frame, summary, fps, scale, compact)
    if not compact and summary.assessments:
        _draw_panel(frame, summary, scale, movement_by_id)

    return frame


def _draw_assessment(
    frame: np.ndarray,
    assessment: RiskAssessment,
    obj: Optional[TrackedObject],
    movement: str,
    scale: float,
    compact: bool,
    frame_index: int,
) -> None:
    detection = assessment.detection
    color = DANGER_COLORS[assessment.level]
    thickness = 2 if assessment.level == DangerLevel.CRITIQUE else 1
    if frame.shape[1] >= 640 and assessment.level == DangerLevel.CRITIQUE:
        thickness = 3

    if obj is not None and len(obj.trail) > 1:
        points = np.asarray(obj.trail, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [points], False, color, 1, cv2.LINE_AA)

    cv2.rectangle(
        frame,
        (detection.x1, detection.y1),
        (detection.x2, detection.y2),
        color,
        thickness,
    )

    if assessment.level == DangerLevel.CRITIQUE and (frame_index // 4) % 2 == 0:
        cv2.rectangle(
            frame,
            (max(detection.x1 - 2, 0), max(detection.y1 - 2, 0)),
            (
                min(detection.x2 + 2, frame.shape[1] - 1),
                min(detection.y2 + 2, frame.shape[0] - 1),
            ),
            color,
            1,
        )

    category = CATEGORY_LABELS.get(assessment.category, assessment.category)
    if compact:
        label = f"#{assessment.track_id} IMG {movement}"
    else:
        label = f"#{assessment.track_id} {category} | IMG {movement}"
        available_width = max(frame.shape[1] - max(detection.x1, 0) - 6, 1)
        label_width = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1
        )[0][0]
        if label_width > available_width:
            label = (
                f"#{assessment.track_id} IMG "
                f"{image_motion_label(obj, frame.shape, compact=True)}"
                if obj is not None
                else f"#{assessment.track_id} IMG --"
            )

    _draw_label(frame, label, detection.x1, detection.y1, color, scale)


def _draw_label(
    frame: np.ndarray,
    label: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    scale: float,
) -> None:
    available_width = max(frame.shape[1] - max(x, 0) - 6, 1)
    label = _fit_text(label, available_width, scale, 1)
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1
    )
    label_top = max(y, text_height + baseline + 2)
    box_x = min(max(x, 0), max(frame.shape[1] - text_width - 4, 0))
    cv2.rectangle(
        frame,
        (box_x, label_top - text_height - baseline - 2),
        (box_x + text_width + 4, label_top),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (box_x + 2, label_top - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def _draw_banner(
    frame: np.ndarray,
    summary: RiskSummary,
    fps: Optional[float],
    scale: float,
    compact: bool,
) -> None:
    frame_height, frame_width = frame.shape[:2]
    banner_height = min(70, max(34, int(frame_height * 0.09)))
    color = DANGER_COLORS[summary.level]

    _blend_rect(
        frame, (0, 0), (frame_width, banner_height), (18, 18, 18), 0.65
    )
    accent_width = max(6, int(8 * scale))
    cv2.rectangle(frame, (0, 0), (accent_width, banner_height), color, -1)

    message_scale = max(0.42, 0.72 * scale)
    if compact:
        message_scale = 0.42

    fps_text = "" if fps is None else f"FPS {fps:.1f}"
    fps_width = 0
    if fps_text:
        fps_width = cv2.getTextSize(
            fps_text, cv2.FONT_HERSHEY_SIMPLEX, message_scale, 1
        )[0][0]

    available = frame_width - accent_width - 20 - fps_width
    message = _fit_text(
        _status_message(summary, compact), available, message_scale, 1
    )

    (_, text_height), baseline = cv2.getTextSize(
        message or " ", cv2.FONT_HERSHEY_SIMPLEX, message_scale, 1
    )
    text_y = banner_height // 2 + text_height // 2
    cv2.putText(
        frame,
        message,
        (accent_width + 10, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        message_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    if fps_text:
        cv2.putText(
            frame,
            fps_text,
            (frame_width - fps_width - 10, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            message_scale,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )


def _status_message(summary: RiskSummary, compact: bool) -> str:
    primary = summary.primary
    if primary is None:
        return "0 OBJET DETECTE" if compact else "AUCUN OBJET DETECTE"

    if summary.level == DangerLevel.INFORMATION:
        if compact:
            return f"{len(summary.assessments)} OBJET(S) - OK"
        return (
            f"{len(summary.assessments)} OBJET(S) - "
            "AUCUN RISQUE ESTIME"
        )

    category = CATEGORY_LABELS.get(primary.category, primary.category)
    level = DANGER_LABELS[summary.level]
    if compact:
        return f"{level} #{primary.track_id}"

    return f"{level} - {category} #{primary.track_id}"
