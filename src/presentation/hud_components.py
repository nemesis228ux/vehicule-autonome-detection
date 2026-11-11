"""Panel, legend, and text-layout helpers for the risk HUD."""

from __future__ import annotations

import cv2
import numpy as np

try:
    from src.risk.models import DANGER_COLORS, DANGER_LABELS, DangerLevel, RiskSummary
except ModuleNotFoundError:
    from risk.models import DANGER_COLORS, DANGER_LABELS, DangerLevel, RiskSummary

from .labels import PANEL_LABELS
from .labels import CATEGORY_LABELS


def _draw_panel(frame: np.ndarray, summary: RiskSummary, scale: float) -> None:
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
    frame_height, _ = frame.shape[:2]
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
        cv2.rectangle(
            frame, (x0 + 10, y - 10), (x0 + 20, y), DANGER_COLORS[level], -1
        )
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
