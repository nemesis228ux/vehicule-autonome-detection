"""Détection d'objets routiers avec YOLOv8 (Ultralytics)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

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


class Detector:
    """Encapsule un modèle YOLOv8 et convertit ses résultats en ``Detection``."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        max_detections: int = 100,
        imgsz: int = 640,
        road_only: bool = True,
        device: Optional[str] = None,
        class_map: Optional[Dict[str, str]] = None,
        model: Any = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold doit être compris entre 0 et 1.")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold doit être compris entre 0 et 1.")
        if max_detections <= 0:
            raise ValueError("max_detections doit être strictement positif.")
        if imgsz <= 0:
            raise ValueError("imgsz doit être strictement positif.")

        self.model_name = self._resolve_model_name(model_name)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.imgsz = imgsz
        self.road_only = road_only
        self.device = device
        self.class_map = dict(ROAD_CLASS_MAP if class_map is None else class_map)
        self.model = model if model is not None else self._load_model()

    @staticmethod
    def _resolve_model_name(model_name: str) -> str:
        path = Path(model_name).expanduser()
        if path.is_file():
            return str(path)

        if not path.suffix:
            path = path.with_suffix(".pt")

        local_model = Path("models") / path.name
        if local_model.is_file():
            return str(local_model)

        return str(path)

    def _load_model(self) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics est requis pour YOLOv8. Installez 'ultralytics'."
            ) from exc

        return YOLO(self.model_name)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Détecte les objets d'une frame BGR et renvoie les objets retenus."""
        if frame is None or frame.size == 0:
            raise ValueError("La frame est vide.")

        kwargs: Dict[str, Any] = {
            "conf": self.confidence_threshold,
            "iou": self.iou_threshold,
            "max_det": self.max_detections,
            "imgsz": self.imgsz,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device

        results = self.model(frame, **kwargs)
        height, width = frame.shape[:2]
        return self._parse_results(results, width, height)

    def _parse_results(
        self, results: Any, frame_width: int, frame_height: int
    ) -> List[Detection]:
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = self._to_numpy(boxes.xyxy)
        confidences = self._to_numpy(boxes.conf)
        class_ids = self._to_numpy(boxes.cls)
        names = getattr(result, "names", None) or getattr(self.model, "names", {})

        detections: List[Detection] = []
        for coords, confidence, class_id in zip(xyxy, confidences, class_ids):
            confidence_value = float(confidence)
            if confidence_value < self.confidence_threshold:
                continue

            class_id_value = int(class_id)
            class_name = self._class_name(names, class_id_value)
            category = self.class_map.get(class_name)
            if category is None:
                if self.road_only:
                    continue
                category = "autre"

            x1, y1, x2, y2 = coords[:4]
            detections.append(
                Detection(
                    x1=self._clamp(x1, frame_width),
                    y1=self._clamp(y1, frame_height),
                    x2=self._clamp(x2, frame_width),
                    y2=self._clamp(y2, frame_height),
                    confidence=confidence_value,
                    class_id=class_id_value,
                    class_name=class_name,
                    category=category,
                )
            )
            if len(detections) >= self.max_detections:
                break

        return detections

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value, dtype=float)

    @staticmethod
    def _class_name(names: Any, class_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, Sequence) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    @staticmethod
    def _clamp(value: float, limit: int) -> int:
        return int(min(max(round(value), 0), max(limit - 1, 0)))


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
