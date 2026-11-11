"""Command-line interface for the detection demo."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .runtime import run


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Détecte les objets routiers d'une vidéo, d'un flux ou d'une caméra."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Fichier vidéo, URL de flux ou identifiant de caméra (défaut : 0).",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Modèle Ultralytics (yolov8n/s/m/l/x.pt) ou chemin vers un .pt (défaut : yolov8n.pt).",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Seuil de confiance.")
    parser.add_argument("--iou", type=float, default=0.45, help="Seuil IoU du NMS.")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Taille d'inférence (640 par défaut ; 320 plus rapide sur CPU).",
    )
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Conserve aussi les classes hors route (affichées en 'autre').",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Enregistre la vidéo annotée dans outputs/.",
    )
    parser.add_argument(
        "--show-fps", action="store_true", help="Affiche les FPS mesurés."
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Active le suivi ByteTrack et affiche les IDs.",
    )
    parser.add_argument(
        "--risk",
        action="store_true",
        help="Active l'évaluation du danger et le HUD (implique --track).",
    )
    parser.add_argument(
        "--alerts",
        action="store_true",
        help="Active les alertes visuelles et sonores (implique --risk).",
    )
    parser.add_argument(
        "--no-sound",
        action="store_true",
        help="Désactive le klaxon et conserve les alertes visuelles.",
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Configuration Ultralytics du tracker (bytetrack.yaml ou botsort.yaml).",
    )
    parser.add_argument(
        "--labels",
        choices=("full", "id", "none"),
        default="id",
        help="Étiquettes de tracking : full (ID+catégorie+conf), id seul, ou aucune.",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=30,
        help="Nombre de points de trajectoire conservés par objet.",
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="Frames sans détection avant purge de l'état local d'un ID.",
    )
    parser.add_argument(
        "--window",
        default="Detection YOLOv8",
        help="Nom ASCII de la fenêtre OpenCV (éviter les accents).",
    )
    args = parser.parse_args(argv)
    if args.alerts:
        args.risk = True
    if args.risk:
        args.track = True
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(parse_args(argv))
