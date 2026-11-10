"""Visual alert overlay rendering."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

try:
    from src.risk.models import DANGER_COLORS, DangerLevel
except ModuleNotFoundError:
    from risk.models import DANGER_COLORS, DangerLevel

from .config import AlertEvent


def draw_alert_overlay(
    frame: np.ndarray,
    event: AlertEvent,
    thickness: Optional[int] = None,
) -> np.ndarray:
    """Dessine une bordure d'alerte et l'état du son."""
    if event.level == DangerLevel.INFORMATION:
        return frame

    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        return frame

    color = DANGER_COLORS[event.level]
    if thickness is None:
        base = max(2, int(min(width, height) * 0.008))
        thickness = base if event.level == DangerLevel.ATTENTION else base * 2
    thickness = max(1, int(thickness))

    if event.level == DangerLevel.ATTENTION or event.flash_on:
        offset = thickness // 2
        cv2.rectangle(
            frame,
            (offset, offset),
            (max(offset, width - 1 - offset), max(offset, height - 1 - offset)),
            color,
            thickness,
            cv2.LINE_AA,
        )

    if event.muted:
        scale = max(0.4, min(0.8, min(width, height) / 500.0))
        text_thickness = max(1, int(scale * 2))
        position = (10, max(18, height - 12))
        cv2.putText(
            frame,
            "SON COUPE",
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            text_thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "SON COUPE",
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )
    return frame
