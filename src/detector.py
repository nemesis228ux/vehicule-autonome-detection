"""Compatibility exports for the detection package."""

try:
    from src.detection import (
        CATEGORY_COLORS,
        DEFAULT_MODEL,
        ROAD_CLASS_MAP,
        Detection,
        Detector,
        count_by_category,
        draw_detections,
    )
except ModuleNotFoundError:
    from detection import (
        CATEGORY_COLORS,
        DEFAULT_MODEL,
        ROAD_CLASS_MAP,
        Detection,
        Detector,
        count_by_category,
        draw_detections,
    )

__all__ = [
    "CATEGORY_COLORS",
    "DEFAULT_MODEL",
    "ROAD_CLASS_MAP",
    "Detection",
    "Detector",
    "count_by_category",
    "draw_detections",
]