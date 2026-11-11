"""Detection data structures and road-category definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

ROAD_CLASS_MAP: Dict[str, str] = {
    "person": "pieton",
    "bicycle": "deux_roues",
    "motorcycle": "deux_roues",
    "car": "vehicule",
    "bus": "vehicule",
    "truck": "vehicule",
    "pothole": "nid_de_poule",
    "nid-de-poule": "nid_de_poule",
    "nid_de_poule": "nid_de_poule",
}

CATEGORY_COLORS: Dict[str, tuple[int, int, int]] = {
    "pieton": (0, 0, 255),
    "deux_roues": (0, 165, 255),
    "vehicule": (0, 200, 0),
    "nid_de_poule": (255, 0, 255),
    "autre": (255, 255, 0),
}

DEFAULT_MODEL = "yolov8n.pt"


@dataclass(frozen=True)
class Detection:
    """Objet détecté dans une frame, exprimé en pixels."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str
    category: str
    track_id: Optional[int] = None

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    @property
    def bottom_center(self) -> tuple[int, int]:
        return (self.x1 + self.x2) // 2, self.y2
