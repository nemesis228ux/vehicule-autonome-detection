"""Compatibility exports for risk HUD rendering."""

try:
    from src.risk.models import (
        DANGER_COLORS,
        DANGER_LABELS,
        DangerLevel,
        RiskAssessment,
        RiskSummary,
    )
    from src.tracker import TrackedObject
    from src.presentation.hud import draw_risk_hud
    from src.presentation.labels import CATEGORY_LABELS, PANEL_LABELS
except ModuleNotFoundError:
    from risk.models import (
        DANGER_COLORS,
        DANGER_LABELS,
        DangerLevel,
        RiskAssessment,
        RiskSummary,
    )
    from tracker import TrackedObject
    from presentation.hud import draw_risk_hud
    from presentation.labels import CATEGORY_LABELS, PANEL_LABELS

__all__ = [
    "CATEGORY_LABELS",
    "DANGER_COLORS",
    "DANGER_LABELS",
    "DangerLevel",
    "PANEL_LABELS",
    "RiskAssessment",
    "RiskSummary",
    "TrackedObject",
    "draw_risk_hud",
]
