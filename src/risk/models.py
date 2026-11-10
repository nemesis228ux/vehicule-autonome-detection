"""Risk levels, configuration, and evaluation data structures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional, Tuple

try:
    from src.detection import Detection
except ModuleNotFoundError:
    from detection import Detection


class DangerLevel(IntEnum):
    INFORMATION = 0
    ATTENTION = 1
    CRITIQUE = 2


DANGER_COLORS: Dict[DangerLevel, Tuple[int, int, int]] = {
    DangerLevel.CRITIQUE: (0, 0, 255),
    DangerLevel.ATTENTION: (0, 140, 255),
    DangerLevel.INFORMATION: (0, 180, 0),
}

DANGER_LABELS: Dict[DangerLevel, str] = {
    DangerLevel.CRITIQUE: "CRITIQUE",
    DangerLevel.ATTENTION: "ATTENTION",
    DangerLevel.INFORMATION: "INFORMATION",
}

CATEGORY_WEIGHTS: Dict[str, float] = {
    "pieton": 1.0,
    "deux_roues": 0.95,
    "nid_de_poule": 0.85,
    "vehicule": 0.7,
    "autre": 0.3,
}

VRU_CATEGORIES = frozenset({"pieton", "deux_roues"})
POTHOLE_CATEGORIES = frozenset({"nid_de_poule"})


@dataclass(frozen=True)
class RiskConfig:
    """Seuils et paramètres de l'évaluation du danger."""

    horizon_ratio: float = 0.5
    corridor_near_half_width: float = 0.38
    corridor_far_half_width: float = 0.06
    proximity_attention: float = 0.60
    proximity_critical: float = 0.82
    ttc_attention: float = 3.0
    ttc_critical_vehicle: float = 1.5
    ttc_critical_vru: float = 2.0
    ttc_critical_pothole: float = 2.0
    max_ttc: float = 10.0
    lateral_ttc_attention: float = 2.0
    min_hits: int = 3
    min_confidence: float = 0.40
    release_frames: int = 10
    state_max_age: int = 30
    score_smoothing: float = 0.6
    ttc_smoothing: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.horizon_ratio < 1.0:
            raise ValueError("horizon_ratio doit être dans [0, 1[.")
        if not 0.0 <= self.corridor_far_half_width <= self.corridor_near_half_width:
            raise ValueError("Les largeurs de corridor sont invalides.")
        if not 0.0 <= self.proximity_attention <= 1.0:
            raise ValueError("proximity_attention doit être dans [0, 1].")
        if not 0.0 <= self.proximity_critical <= 1.0:
            raise ValueError("proximity_critical doit être dans [0, 1].")
        if min(
            self.ttc_attention,
            self.ttc_critical_vehicle,
            self.ttc_critical_vru,
            self.ttc_critical_pothole,
            self.max_ttc,
            self.lateral_ttc_attention,
        ) <= 0.0:
            raise ValueError("Les seuils TTC doivent être strictement positifs.")
        if self.min_hits < 1:
            raise ValueError("min_hits doit être au moins 1.")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence doit être dans [0, 1].")
        if self.release_frames < 0 or self.state_max_age < 0:
            raise ValueError("release_frames et state_max_age doivent être positifs ou nuls.")
        if not 0.0 <= self.score_smoothing <= 1.0:
            raise ValueError("score_smoothing doit être dans [0, 1].")
        if not 0.0 <= self.ttc_smoothing <= 1.0:
            raise ValueError("ttc_smoothing doit être dans [0, 1].")


@dataclass(frozen=True)
class RiskAssessment:
    """Niveau de danger estimé pour un objet suivi."""

    track_id: int
    category: str
    level: DangerLevel
    score: float
    ttc: Optional[float]
    proximity: float
    in_path: bool
    lateral_ratio: float
    reasons: Tuple[str, ...]
    detection: Detection


@dataclass(frozen=True)
class RiskSummary:
    """Synthèse du danger pour une frame."""

    level: DangerLevel
    primary: Optional[RiskAssessment]
    assessments: Tuple[RiskAssessment, ...]
    counts: Dict[DangerLevel, int]


@dataclass
class _RiskState:
    prev_height: Optional[float] = None
    prev_bottom: Optional[float] = None
    smoothed_ttc: Optional[float] = None
    smoothed_score: float = 0.0
    score_initialized: bool = False
    level: DangerLevel = DangerLevel.INFORMATION
    release_counter: int = 0
    missed: int = 0
