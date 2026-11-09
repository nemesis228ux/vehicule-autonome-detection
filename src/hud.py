"""HUD de danger réutilisable par la démo et l'application finale."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import cv2
import numpy as np

try:
    from src.risk_evaluator import (
        DANGER_COLORS,
        DANGER_LABELS,
        DangerLevel,
        RiskAssessment,
        RiskSummary,
    )
    from src.tracker import TrackedObject
except ModuleNotFoundError:
    from risk_evaluator import (
        DANGER_COLORS,
        DANGER_LABELS,
        DangerLevel,
        RiskAssessment,
        RiskSummary,
    )
    from tracker import TrackedObject

CATEGORY_LABELS: Dict[str, str] = {
    "pieton": "Pieton",
    "deux_roues": "Deux-roues",
    "vehicule": "Vehicule",
    "nid_de_poule": "Nid-de-poule",
    "autre": "Objet",
}

PANEL_LABELS: Dict[DangerLevel, str] = {
    DangerLevel.CRITIQUE: "Critique",
    DangerLevel.ATTENTION: "Attention",
    DangerLevel.INFORMATION: "Info",
}


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

    frame_height, frame_width = frame.shape[:2]
    scale = max(0.4, min(0.8, frame_width / 1280.0))
    compact = frame_width < 480
    objects_by_id = {obj.track_id: obj for obj in objects}

    for assessment in summary.assessments:
        _draw_assessment(
            frame,
            assessment,
            objects_by_id.get(assessment.track_id),
            scale,
            compact,
            frame_index,
        )

    _draw_banner(frame, summary, fps, scale, compact)
    if not compact and summary.assessments:
        _draw_panel(frame, summary, scale)
    if not compact and frame_height >= 360:
        _draw_legend(frame, scale)

    return frame


def _draw_assessment(
    frame: np.ndarray,
    assessment: RiskAssessment,
    obj: Optional[TrackedObject],
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
        label = f"#{assessment.track_id}"
    else:
        label = f"#{assessment.track_id} {category}"
        if assessment.ttc is not None:
            label += f" {assessment.ttc:.1f}s"

    _draw_label(frame, label, detection.x1, detection.y1, color, scale)


def _draw_label(
    frame: np.ndarray,
    label: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    scale: float,
) -> None:
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
        return "Aucun danger immediat"

    if summary.level == DangerLevel.INFORMATION:
        if compact:
            return "OK"
        return f"Surveillance - {len(summary.assessments)} objet(s)"

    category = CATEGORY_LABELS.get(primary.category, primary.category)
    level = DANGER_LABELS[summary.level]
    if compact:
        return f"{level} #{primary.track_id}"

    message = f"{level} - {category} #{primary.track_id}"
    if primary.ttc is not None:
        message += f" - TTC {primary.ttc:.1f} s"
    return message


def _draw_panel(
    frame: np.ndarray, summary: RiskSummary, scale: float
) -> None:
    frame_height, frame_width = frame.shape[:2]
    panel_width = int(min(max(180, frame_width * 0.26), frame_width * 0.38))
    panel_height = int(min(frame_height * 0.5, max(150, 200 * scale)))
    x0 = frame_width - panel_width - 10
    y0 = min(70, max(34, int(frame_height * 0.09))) + 10

    _blend_rect(
        frame, (x0, y0), (x0 + panel_width, y0 + panel_height), (12, 12, 12), 0.6
    )
    cv2.rectangle(
        frame,
        (x0, y0),
        (x0 + panel_width, y0 + panel_height),
        DANGER_COLORS[summary.level],
        2,
    )

    text_scale = max(0.4, 0.55 * scale)
    line_height = int(20 * scale) + 6
    cv2.putText(
        frame,
        "DANGERS",
        (x0 + 10, y0 + line_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    y = y0 + 2 * line_height
    for level in (
        DangerLevel.CRITIQUE,
        DangerLevel.ATTENTION,
        DangerLevel.INFORMATION,
    ):
        color = DANGER_COLORS[level]
        cv2.rectangle(frame, (x0 + 10, y - 10), (x0 + 20, y), color, -1)
        cv2.putText(
            frame,
            f"{PANEL_LABELS[level]} {summary.counts.get(level, 0)}",
            (x0 + 28, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        y += line_height

    y += 4
    cv2.line(
        frame,
        (x0 + 10, y - line_height // 2),
        (x0 + panel_width - 10, y - line_height // 2),
        (80, 80, 80),
        1,
    )

    for assessment in summary.assessments[:3]:
        if y > y0 + panel_height - 6:
            break
        category = CATEGORY_LABELS.get(
            assessment.category, assessment.category
        )
        text = f"#{assessment.track_id} {category}"
        if assessment.ttc is not None:
            text += f" {assessment.ttc:.1f}s"
        text = _fit_text(text, panel_width - 36, text_scale, 1)
        cv2.rectangle(
            frame,
            (x0 + 10, y - 8),
            (x0 + 18, y),
            DANGER_COLORS[assessment.level],
            -1,
        )
        cv2.putText(
            frame,
            text,
            (x0 + 26, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        y += line_height


def _draw_legend(frame: np.ndarray, scale: float) -> None:
    frame_height, frame_width = frame.shape[:2]
    text_scale = max(0.4, 0.5 * scale)
    line_height = int(20 * scale) + 4
    legend_width = int(max(150, 190 * scale))
    legend_height = line_height * 4 + 16
    x0 = 10
    y0 = frame_height - legend_height - 10

    _blend_rect(
        frame, (x0, y0), (x0 + legend_width, y0 + legend_height), (12, 12, 12), 0.6
    )
    cv2.putText(
        frame,
        "LEGENDE",
        (x0 + 10, y0 + line_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    y = y0 + 2 * line_height
    for level in (
        DangerLevel.CRITIQUE,
        DangerLevel.ATTENTION,
        DangerLevel.INFORMATION,
    ):
        cv2.rectangle(frame, (x0 + 10, y - 10), (x0 + 20, y), DANGER_COLORS[level], -1)
        cv2.putText(
            frame,
            PANEL_LABELS[level],
            (x0 + 28, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        y += line_height


def _blend_rect(
    frame: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def _fit_text(
    text: str, max_width: int, scale: float, thickness: int
) -> str:
    if max_width <= 0:
        return ""
    if _text_width(text, scale, thickness) <= max_width:
        return text

    shortened = text
    while shortened and _text_width(shortened + "...", scale, thickness) > max_width:
        shortened = shortened[:-1]
    return (shortened + "...") if shortened else ""


def _text_width(text: str, scale: float, thickness: int) -> int:
    return cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )[0][0]
