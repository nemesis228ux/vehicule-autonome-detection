"""YOLOv8 inference and result parsing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .models import DEFAULT_MODEL, ROAD_CLASS_MAP, Detection


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
        self.class_ids = self._resolve_class_ids()

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
        model_path = Path(self.model_name)
        if model_path.is_file() and model_path.stat().st_size == 0:
            raise OSError(
                f"Le fichier de modèle est vide : {model_path}. "
                "Fournissez un checkpoint YOLO valide."
            )

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics est requis pour YOLOv8. Installez 'ultralytics'."
            ) from exc

        return YOLO(self.model_name)

    def _resolve_class_ids(self) -> Optional[List[int]]:
        if not self.road_only:
            return None

        names = getattr(self.model, "names", None)
        if not names:
            return None

        items = names.items() if isinstance(names, dict) else enumerate(names)
        class_ids = [
            int(class_id)
            for class_id, name in items
            if self._normalize_class_name(name) in self.class_map
        ]
        return sorted(set(class_ids)) or None

    @staticmethod
    def _normalize_class_name(name: Any) -> str:
        return " ".join(str(name).strip().lower().replace("_", " ").split())

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
        if self.class_ids is not None:
            kwargs["classes"] = self.class_ids

        results = self.model(frame, **kwargs)
        height, width = frame.shape[:2]
        return self._parse_results(results, width, height)

    def track(
        self, frame: np.ndarray, tracker: str = "bytetrack.yaml"
    ) -> List[Detection]:
        """Détecte et suit les objets d'une frame, avec un ``track_id`` stable.

        Les détections de faible confiance sont conservées : ByteTrack en a
        besoin pour maintenir les IDs. Le filtrage d'affichage est assuré par
        :class:`src.tracker.Tracker`.
        """
        if frame is None or frame.size == 0:
            raise ValueError("La frame est vide.")

        kwargs: Dict[str, Any] = {
            "persist": True,
            "tracker": tracker,
            "conf": 0.1,
            "iou": self.iou_threshold,
            "max_det": self.max_detections,
            "imgsz": self.imgsz,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        if self.class_ids is not None:
            kwargs["classes"] = self.class_ids

        results = self.model.track(frame, **kwargs)
        height, width = frame.shape[:2]
        return self._parse_results(
            results, width, height, min_confidence=0.0
        )

    def _parse_results(
        self,
        results: Any,
        frame_width: int,
        frame_height: int,
        min_confidence: Optional[float] = None,
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

        threshold = (
            self.confidence_threshold
            if min_confidence is None
            else min_confidence
        )

        raw_ids = getattr(boxes, "id", None)
        track_ids = (
            self._to_numpy(raw_ids).reshape(-1) if raw_ids is not None else None
        )

        detections: List[Detection] = []
        for index, (coords, confidence, class_id) in enumerate(
            zip(xyxy, confidences, class_ids)
        ):
            detection = self._parse_detection(
                coords,
                confidence,
                class_id,
                index,
                names,
                track_ids,
                threshold,
                frame_width,
                frame_height,
            )
            if detection is None:
                continue
            detections.append(detection)
            if len(detections) >= self.max_detections:
                break

        return detections

    def _parse_detection(
        self,
        coords: np.ndarray,
        confidence: float,
        class_id: float,
        index: int,
        names: Any,
        track_ids: Optional[np.ndarray],
        threshold: float,
        frame_width: int,
        frame_height: int,
    ) -> Optional[Detection]:
        confidence_value = float(confidence)
        if confidence_value < threshold:
            return None

        class_id_value = int(class_id)
        class_name = self._class_name(names, class_id_value)
        category = self.class_map.get(self._normalize_class_name(class_name))
        if category is None:
            if self.road_only:
                return None
            category = "autre"

        track_id = None
        if track_ids is not None and index < len(track_ids):
            track_id = int(round(float(track_ids[index])))

        x1, y1, x2, y2 = coords[:4]
        return Detection(
            x1=self._clamp(x1, frame_width),
            y1=self._clamp(y1, frame_height),
            x2=self._clamp(x2, frame_width),
            y2=self._clamp(y2, frame_height),
            confidence=confidence_value,
            class_id=class_id_value,
            class_name=class_name,
            category=category,
            track_id=track_id,
        )

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
