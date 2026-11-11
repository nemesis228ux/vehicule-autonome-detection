"""Compatibility exports for the risk evaluation package."""

try:
    from src.risk import (
        CATEGORY_WEIGHTS,
        DANGER_COLORS,
        DANGER_LABELS,
        POTHOLE_CATEGORIES,
        VRU_CATEGORIES,
        DangerLevel,
        RiskAssessment,
        RiskConfig,
        RiskEvaluator,
        RiskSummary,
    )
except ModuleNotFoundError:
    from risk import (
        CATEGORY_WEIGHTS,
        DANGER_COLORS,
        DANGER_LABELS,
        POTHOLE_CATEGORIES,
        VRU_CATEGORIES,
        DangerLevel,
        RiskAssessment,
        RiskConfig,
        RiskEvaluator,
        RiskSummary,
    )

__all__ = [
    "CATEGORY_WEIGHTS",
    "DANGER_COLORS",
    "DANGER_LABELS",
    "POTHOLE_CATEGORIES",
    "VRU_CATEGORIES",
    "DangerLevel",
    "RiskAssessment",
    "RiskConfig",
    "RiskEvaluator",
    "RiskSummary",
]
