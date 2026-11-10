"""Suivi multi-objets basé sur ByteTrack, avec vitesse et trajectoire."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import hypot
from typing import TYPE_CHECKING, Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from src.detector import CATEGORY_COLORS, Detection
except ModuleNotFoundError:
    from detector import CATEGORY_COLORS, Detection

if TYPE_CHECKING:
    from src.detector import Detector

DEFAULT_TRACKER_CONFIG = "bytetrack.yaml"


@dataclass
class TrackedObject:
    """Objet suivi : détection courante enrichie d'un identifiant stable."""

    track_id: int
    detection: Detection
    age: int
    hits: int
    time_since_update: int
    velocity: Tuple[float, float]
    trail: Tuple[Tuple[int, int], ...]

    @property
    def speed(self) -> float:
        """Vitesse instantanée en pixels par frame."""
        return hypot(self.velocity[0], self.velocity[1])

    @property
    def category(self) -> str:
        return self.detection.category


@dataclass
class _TrackState:
    center: Tuple[float, float]
    velocity: Tuple[float, float] = (0.0, 0.0)
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    trail: Deque[Tuple[int, int]] = field(default_factory=deque)


class Tracker:
    """Enrichit les IDs ByteTrack avec vitesse, trajectoire et cycle de vie.

    ByteTrack (Ultralytics) réalise l'association entre frames. Cette classe
    conserve un état local par ``track_id`` pour fournir au module de danger
    des informations temporelles (vitesse, trajectoire, présence).
    """

    def __init__(
        self,
        detector: "Detector",
        tracker_config: str = DEFAULT_TRACKER_CONFIG,
        trail_length: int = 30,
        max_age: int = 30,
        velocity_smoothing: float = 0.6,
        min_confidence: Optional[float] = None,
    ) -> None:
        if trail_length <= 0:
            raise ValueError("trail_length doit être strictement positif.")
        if max_age < 0:
            raise ValueError("max_age doit être positif ou nul.")
        if not 0.0 <= velocity_smoothing <= 1.0:
            raise ValueError("velocity_smoothing doit être compris entre 0 et 1.")
        if min_confidence is not None and not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence doit être compris entre 0 et 1.")

        self.detector = detector
        self.tracker_config = tracker_config
        self.trail_length = trail_length
        self.max_age = max_age
        self.velocity_smoothing = velocity_smoothing
        self.min_confidence = (
            detector.confidence_threshold
            if min_confidence is None
            else min_confidence
        )
        self._states: Dict[int, _TrackState] = {}

    def update(self, frame: np.ndarray) -> List[TrackedObject]:
        """Suit les objets d'une frame BGR et renvoie les objets confirmés."""
        detections = self.detector.track(frame, tracker=self.tracker_config)
        return self.update_detections(detections)

    def update_detections(
        self, detections: Sequence[Detection]
    ) -> List[TrackedObject]:
        """Met à jour l'état interne à partir de détections déjà suivies."""
        for state in self._states.values():
            state.age += 1

        seen: set[int] = set()
        objects: List[TrackedObject] = []

        for detection in detections:
            track_id = detection.track_id
            if track_id is None:
                continue

            seen.add(track_id)
            center = detection.center
            state = self._states.get(track_id)

            if state is None:
                state = _TrackState(
                    center=center,
                    trail=deque([center], maxlen=self.trail_length),
                )
                self._states[track_id] = state
            else:
                gap = state.time_since_update + 1
                dx = (center[0] - state.center[0]) / gap
                dy = (center[1] - state.center[1]) / gap
                alpha = self.velocity_smoothing
                state.velocity = (
                    alpha * state.velocity[0] + (1.0 - alpha) * dx,
                    alpha * state.velocity[1] + (1.0 - alpha) * dy,
                )
                state.center = center
                state.hits += 1
                state.time_since_update = 0
                state.trail.append(center)

            if detection.confidence >= self.min_confidence:
                objects.append(
                    TrackedObject(
                        track_id=track_id,
                        detection=detection,
                        age=state.age,
                        hits=state.hits,
                        time_since_update=state.time_since_update,
                        velocity=state.velocity,
                        trail=tuple(state.trail),
                    )
                )

        for track_id, state in list(self._states.items()):
            if track_id in seen:
                continue
            state.time_since_update += 1
            if state.time_since_update > self.max_age:
                del self._states[track_id]

        return objects

    def reset(self) -> None:
        """Vide l'état local (ByteTrack garde ses IDs côté modèle)."""
        self._states.clear()


def draw_tracks(
    frame: np.ndarray,
    objects: Sequence[TrackedObject],
    show_trail: bool = True,
    labels: str = "full",
    font_scale: Optional[float] = None,
) -> np.ndarray:
    """Dessine les objets suivis, leurs IDs et leurs trajectoires.

    ``labels`` vaut ``"full"`` (ID + catégorie + confiance), ``"id"`` (ID
    seul) ou ``"none"`` (boîtes et trajectoires seulement).
    """
    if labels not in ("full", "id", "none"):
        raise ValueError("labels doit être 'full', 'id' ou 'none'.")

    scale = (
        font_scale
        if font_scale is not None
        else (0.35 if frame.shape[1] < 480 else 0.5)
    )

    for obj in objects:
        detection = obj.detection
        color = CATEGORY_COLORS.get(
            detection.category, CATEGORY_COLORS["autre"]
        )

        if show_trail and len(obj.trail) > 1:
            points = np.asarray(obj.trail, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                frame, [points], False, color, 1, cv2.LINE_AA
            )

        cv2.rectangle(
            frame,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            color,
            2,
        )

        if labels == "none":
            continue

        if labels == "id":
            label = f"#{obj.track_id}"
        else:
            label = (
                f"#{obj.track_id} {detection.category} "
                f"{detection.confidence:.0%}"
            )

        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1
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
            scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return frame
