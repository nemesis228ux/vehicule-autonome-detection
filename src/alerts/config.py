"""Public alert configuration and event data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    from src.risk.models import DangerLevel
except ModuleNotFoundError:
    from risk.models import DangerLevel


@dataclass(frozen=True)
class AlertConfig:
    """Paramètres de temporisation et d'affichage des alertes."""

    attention_debounce_frames: int = 2
    release_frames: int = 5
    attention_cooldown: float = 3.0
    critical_cooldown: float = 1.2
    flash_hz: float = 2.0

    def __post_init__(self) -> None:
        if self.attention_debounce_frames < 1:
            raise ValueError("attention_debounce_frames doit être >= 1.")
        if self.release_frames < 1:
            raise ValueError("release_frames doit être >= 1.")
        if self.attention_cooldown <= 0.0:
            raise ValueError("attention_cooldown doit être strictement positif.")
        if self.critical_cooldown <= 0.0:
            raise ValueError("critical_cooldown doit être strictement positif.")
        if self.flash_hz <= 0.0:
            raise ValueError("flash_hz doit être strictement positif.")


@dataclass(frozen=True)
class AlertEvent:
    """Décision d'alerte pour une frame donnée."""

    level: DangerLevel
    changed: bool
    sound: Optional[str]
    muted: bool
    flash_on: bool
    duration: float
