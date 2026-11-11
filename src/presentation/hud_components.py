"""Panel and text-layout helpers for the risk HUD."""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

try:
    from src.risk.models import DANGER_COLORS, DangerLevel, RiskSummary
except ModuleNotFoundError:
    from risk.models import DANGER_COLORS, DangerLevel, RiskSummary

from .labels import PANEL_LABELS
from .labels import CATEGORY_LABELS


def _draw_panel(
    frame: np.ndarray,
    summary: RiskSummary,
    scale: float,
    movement_by_id: Dict[int, str],
) -> None:
    frame_height, frame_width = frame.shape[:2]
    panel_width = int(min(max(220, frame_width * 0.3), frame_width * 0.45))
    panel_height = int(min(frame_height * 0.62, max(190, 260 * scale)))
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

    text_scale = max(0.38, 0.48 * scale)
    line_height = int(19 * scale) + 5
    cv2.putText(
        frame,
        "RISQUE ESTIME",
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
        if y > y0 + panel_height - 2 * line_height:
            break
        category = CATEGORY_LABELS.get(
            assessment.category, assessment.category
        )
        movement = movement_by_id.get(assessment.track_id, "--")
        text = f"#{assessment.track_id} IMG {movement} | {category}"
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

    note = "PAS DE DISTANCE MESUREE"
    note_y = y0 + panel_height - 8
    note = _fit_text(note, panel_width - 20, text_scale, 1)
    cv2.putText(
        frame,
        note,
        (x0 + 10, note_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )


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


def _fit_text(text: str, max_width: int, scale: float, thickness: int) -> str:
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
