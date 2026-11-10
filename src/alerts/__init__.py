"""Alert management, sound backends, and visual overlays."""

from .config import AlertConfig, AlertEvent
from .drawing import draw_alert_overlay
from .manager import AlertManager
from .sound import (
    ATTENTION_SOUND,
    CRITICAL_SOUND,
    NullSoundBackend,
    SOUND_CANDIDATES,
    SoundBackend,
    WinsoundBackend,
    build_sound_backend,
)

__all__ = [
    "ATTENTION_SOUND",
    "CRITICAL_SOUND",
    "AlertConfig",
    "AlertEvent",
    "AlertManager",
    "NullSoundBackend",
    "SOUND_CANDIDATES",
    "SoundBackend",
    "WinsoundBackend",
    "build_sound_backend",
    "draw_alert_overlay",
]
