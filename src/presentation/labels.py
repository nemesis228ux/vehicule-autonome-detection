"""Labels shared by risk HUD components."""

from typing import Dict

try:
    from src.risk.models import DangerLevel
except ModuleNotFoundError:
    from risk.models import DangerLevel

CATEGORY_LABELS: Dict[str, str] = {
    "pieton": "Pieton",
    "deux_roues": "Deux-roues",
    "vehicule": "Vehicule",
    "nid_de_poule": "Nid-de-poule",
    "autre": "Objet",
}

PANEL_LABELS: Dict[DangerLevel, str] = {
    DangerLevel.CRITIQUE: "Critique",
    DangerLevel.ATTENTION: "Attention",
    DangerLevel.INFORMATION: "Info",
}
