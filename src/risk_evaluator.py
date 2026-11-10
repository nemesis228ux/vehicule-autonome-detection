"""Évaluation du danger à partir des objets suivis (approche monoculaire)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from src.detector import Detection
    from src.tracker import TrackedObject
except ModuleNotFoundError:
    from detector import Detection
    from tracker import TrackedObject


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


class RiskEvaluator:
    """Transforme des pistes en niveaux de danger stables et explicables."""

    def __init__(self, config: Optional[RiskConfig] = None) -> None:
        self.config = config or RiskConfig()
        self._states: Dict[int, _RiskState] = {}

    def reset(self) -> None:
        self._states.clear()

    def evaluate(
        self,
        objects: Sequence[TrackedObject],
        frame_shape: Tuple[int, ...],
        fps: float = 30.0,
    ) -> RiskSummary:
        if frame_shape is None or len(frame_shape) < 2:
            raise ValueError("frame_shape est invalide.")
        frame_height = int(frame_shape[0])
        frame_width = int(frame_shape[1])
        if frame_height <= 0 or frame_width <= 0:
            raise ValueError("Les dimensions de la frame sont invalides.")
        if fps is None or fps <= 0:
            raise ValueError("fps doit être strictement positif.")

        for state in self._states.values():
            state.missed += 1

        seen: set[int] = set()
        assessments: List[RiskAssessment] = []

        for obj in objects:
            track_id = obj.track_id
            if track_id is None:
                continue

            seen.add(track_id)
            state = self._states.get(track_id)
            if state is None:
                state = _RiskState()
                self._states[track_id] = state
            state.missed = 0
            assessments.append(
                self._assess(obj, state, frame_width, frame_height, float(fps))
            )

        for track_id, state in list(self._states.items()):
            if track_id in seen:
                continue
            if state.missed > self.config.state_max_age:
                del self._states[track_id]

        assessments.sort(key=lambda a: (int(a.level), a.score), reverse=True)
        counts = {level: 0 for level in DangerLevel}
        for assessment in assessments:
            counts[assessment.level] += 1

        primary = assessments[0] if assessments else None
        level = primary.level if primary is not None else DangerLevel.INFORMATION
        return RiskSummary(level, primary, tuple(assessments), counts)

    def _assess(
        self,
        obj: TrackedObject,
        state: _RiskState,
        frame_width: int,
        frame_height: int,
        fps: float,
    ) -> RiskAssessment:
        config = self.config
        detection = obj.detection
        category = detection.category

        height_px = float(max(detection.height, 1))
        bottom_px = float(detection.y2)

        proximity = self._proximity(bottom_px, frame_height)
        ttc = self._update_ttc(
            state, height_px, bottom_px, frame_height, fps
        )
        in_path, lateral_ratio = self._corridor(
            detection, proximity, frame_width
        )
        lateral_ttc = self._lateral_ttc(obj, frame_width, fps)

        mature = (
            obj.hits >= config.min_hits
            and detection.confidence >= config.min_confidence
            and obj.time_since_update == 0
        )

        critical_ttc = self._critical_ttc(category)
        critical = in_path and mature and (
            (ttc is not None and ttc <= critical_ttc)
            or proximity >= config.proximity_critical
        )
        attention = in_path and (
            (ttc is not None and ttc <= config.ttc_attention)
            or proximity >= config.proximity_attention
        )
        crossing = (
            category in VRU_CATEGORIES
            and not in_path
            and lateral_ttc is not None
            and lateral_ttc <= config.lateral_ttc_attention
            and proximity >= 0.25
        )
        if crossing:
            attention = True

        if critical:
            new_level = DangerLevel.CRITIQUE
        elif attention:
            new_level = DangerLevel.ATTENTION
        else:
            new_level = DangerLevel.INFORMATION

        if not mature and new_level == DangerLevel.CRITIQUE:
            new_level = DangerLevel.ATTENTION

        score = self._score(
            category, ttc, proximity, in_path, lateral_ratio, crossing, new_level
        )
        if not state.score_initialized:
            state.smoothed_score = score
            state.score_initialized = True
        else:
            alpha = config.score_smoothing
            state.smoothed_score = (
                alpha * state.smoothed_score + (1.0 - alpha) * score
            )

        if new_level > state.level:
            state.level = new_level
            state.release_counter = 0
        elif new_level < state.level:
            state.release_counter += 1
            if state.release_counter >= config.release_frames:
                state.level = new_level
                state.release_counter = 0
        else:
            state.release_counter = 0

        reasons = self._reasons(
            in_path=in_path,
            ttc=ttc,
            proximity=proximity,
            crossing=crossing,
            category=category,
            mature=mature,
            held=state.level > new_level,
        )

        return RiskAssessment(
            track_id=obj.track_id,
            category=category,
            level=state.level,
            score=state.smoothed_score,
            ttc=ttc,
            proximity=proximity,
            in_path=in_path,
            lateral_ratio=lateral_ratio,
            reasons=reasons,
            detection=detection,
        )

    def _critical_ttc(self, category: str) -> float:
        if category in VRU_CATEGORIES:
            return self.config.ttc_critical_vru
        if category in POTHOLE_CATEGORIES:
            return self.config.ttc_critical_pothole
        return self.config.ttc_critical_vehicle

    def _proximity(self, bottom_px: float, frame_height: int) -> float:
        horizon = frame_height * self.config.horizon_ratio
        span = max(1.0, frame_height - horizon)
        return max(0.0, min(1.0, (bottom_px - horizon) / span))

    def _corridor(
        self,
        detection: Detection,
        proximity: float,
        frame_width: int,
    ) -> Tuple[bool, float]:
        half_image = max(frame_width / 2.0, 1.0)
        x_norm = (detection.center[0] - frame_width / 2.0) / half_image
        half_width = self.config.corridor_far_half_width + (
            self.config.corridor_near_half_width
            - self.config.corridor_far_half_width
        ) * proximity
        object_half_width = (detection.width / 2.0) / half_image
        allowed = max(half_width + object_half_width, 1e-6)
        lateral_ratio = abs(x_norm) / allowed
        return lateral_ratio <= 1.0, lateral_ratio

    def _lateral_ttc(
        self, obj: TrackedObject, frame_width: int, fps: float
    ) -> Optional[float]:
        vx = obj.velocity[0]
        if abs(vx) < 0.5:
            return None
        center_x = obj.detection.center[0]
        toward_center = (center_x > frame_width / 2.0 and vx < 0) or (
            center_x < frame_width / 2.0 and vx > 0
        )
        if not toward_center:
            return None
        return (abs(center_x - frame_width / 2.0) / abs(vx)) / fps

    def _update_ttc(
        self,
        state: _RiskState,
        height_px: float,
        bottom_px: float,
        frame_height: int,
        fps: float,
    ) -> Optional[float]:
        config = self.config
        ttc: Optional[float] = None

        if state.prev_height is not None and state.prev_height > 1.0:
            ratio = height_px / state.prev_height
            if ratio > 1.0 + 1e-3:
                ttc = (1.0 / (ratio - 1.0)) / fps

        if ttc is None and state.prev_bottom is not None:
            delta = bottom_px - state.prev_bottom
            if delta > 0.5:
                ttc = (max(frame_height - bottom_px, 1.0) / delta) / fps

        state.prev_height = height_px
        state.prev_bottom = bottom_px

        if ttc is None:
            state.smoothed_ttc = None
            return None

        ttc = min(ttc, config.max_ttc)
        if state.smoothed_ttc is None:
            state.smoothed_ttc = ttc
        else:
            alpha = config.ttc_smoothing
            state.smoothed_ttc = alpha * state.smoothed_ttc + (1.0 - alpha) * ttc
        return state.smoothed_ttc

    def _score(
        self,
        category: str,
        ttc: Optional[float],
        proximity: float,
        in_path: bool,
        lateral_ratio: float,
        crossing: bool,
        level: DangerLevel,
    ) -> float:
        weight = CATEGORY_WEIGHTS.get(category, 0.3)
        ttc_component = (
            0.0
            if ttc is None
            else max(0.0, 1.0 - min(ttc, self.config.max_ttc) / self.config.max_ttc)
        )
        if in_path:
            path_component = 1.0
        else:
            path_component = max(
                0.0, 1.0 - min(max(lateral_ratio - 1.0, 0.0), 1.0)
            )
        score = weight * (
            0.5 * ttc_component + 0.3 * proximity + 0.2 * path_component
        )
        if crossing:
            score = max(score, weight * 0.55)
        if level == DangerLevel.CRITIQUE:
            score = max(score, 0.75)
        elif level == DangerLevel.ATTENTION:
            score = max(score, 0.45)
        return min(score, 1.0)

    def _reasons(
        self,
        in_path: bool,
        ttc: Optional[float],
        proximity: float,
        crossing: bool,
        category: str,
        mature: bool,
        held: bool,
    ) -> Tuple[str, ...]:
        reasons: List[str] = []
        reasons.append("dans la trajectoire" if in_path else "hors trajectoire")
        if ttc is not None:
            reasons.append(f"TTC {ttc:.1f} s")
        if proximity >= self.config.proximity_attention:
            reasons.append(f"proximite {proximity:.0%}")
        if crossing:
            reasons.append("traversee imminente")
        if category in VRU_CATEGORIES:
            reasons.append("usager vulnerable")
        if category in POTHOLE_CATEGORIES:
            reasons.append("nid-de-poule")
        if not mature:
            reasons.append("track en confirmation")
        if held:
            reasons.append("danger maintenu")
        if len(reasons) == 1 and not in_path:
            reasons.append("aucun danger immediat")
        return tuple(reasons)
