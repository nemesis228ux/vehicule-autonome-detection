"""Drawing and aggregation helpers for detections."""

from __future__ import annotations

from typing import Dict, Sequence

import cv2
import numpy as np

from .models import CATEGORY_COLORS, Detection


def draw_detections(
    frame: np.ndarray, detections: Sequence[Detection]
) -> np.ndarray:
    """Dessine les boîtes et les étiquettes sur la frame (modifiée sur place)."""
    for detection in detections:
        color = CATEGORY_COLORS.get(detection.category, CATEGORY_COLORS["autre"])
        cv2.rectangle(
            frame,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            color,
            2,
        )

        label = f"{detection.category} {detection.confidence:.0%}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        label_top = max(detection.y1, text_height + baseline + 2)
        cv2.rectangle(
            frame,
            (detection.x1, label_top - text_height - baseline - 2),
            (detection.x1 + text_width + 4, label_top),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (detection.x1 + 2, label_top - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return frame


def count_by_category(detections: Sequence[Detection]) -> Dict[str, int]:
    """Compte les détections par catégorie routière."""
    counts: Dict[str, int] = {}
    for detection in detections:
        counts[detection.category] = counts.get(detection.category, 0) + 1
    return counts
