"""Compatibility exports for the alerts package."""

try:
    from src.alerts import (
        ATTENTION_SOUND,
        CRITICAL_SOUND,
        AlertConfig,
        AlertEvent,
        AlertManager,
        NullSoundBackend,
        SOUND_CANDIDATES,
        SoundBackend,
        WinsoundBackend,
        build_sound_backend,
        draw_alert_overlay,
    )
except ModuleNotFoundError:
    from alerts import (
        ATTENTION_SOUND,
        CRITICAL_SOUND,
        AlertConfig,
        AlertEvent,
        AlertManager,
        NullSoundBackend,
        SOUND_CANDIDATES,
        SoundBackend,
        WinsoundBackend,
        build_sound_backend,
        draw_alert_overlay,
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
