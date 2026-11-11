"""Alert state management for risk events."""

from __future__ import annotations

import time
from typing import Dict, Optional

try:
    from src.risk.models import DangerLevel, RiskSummary
except ModuleNotFoundError:
    from risk.models import DangerLevel, RiskSummary

from .config import AlertConfig, AlertEvent
from .sound import (
    ATTENTION_SOUND,
    CRITICAL_SOUND,
    NullSoundBackend,
    SoundBackend,
)


class AlertManager:
    """Transforme les niveaux de risque en alertes stables et non répétitives."""

    def __init__(
        self,
        config: Optional[AlertConfig] = None,
        sound_backend: Optional[SoundBackend] = None,
        muted: bool = False,
    ) -> None:
        self.config = config or AlertConfig()
        self._backend = sound_backend or NullSoundBackend()
        self._muted = muted
        self._active_level = DangerLevel.INFORMATION
        self._pending_level = DangerLevel.INFORMATION
        self._pending_frames = 0
        self._release_frames = 0
        self._last_sound: Dict[DangerLevel, float] = {}
        self._counts: Dict[DangerLevel, int] = {
            level: 0 for level in DangerLevel
        }
        self._since = 0.0
        self._closed = False

    @property
    def muted(self) -> bool:
        return self._muted

    @property
    def sound_available(self) -> bool:
        return self._backend.available

    @property
    def active_level(self) -> DangerLevel:
        return self._active_level

    @property
    def counts(self) -> Dict[DangerLevel, int]:
        return dict(self._counts)

    def toggle_mute(self) -> bool:
        self._muted = not self._muted
        return self._muted

    def update(
        self,
        summary: RiskSummary,
        now_s: Optional[float] = None,
        frame_index: Optional[int] = None,
    ) -> AlertEvent:
        del frame_index
        now = time.monotonic() if now_s is None else now_s
        target = summary.level
        changed = False
        triggered = False

        if target > self._active_level:
            self._release_frames = 0
            if target == DangerLevel.CRITIQUE:
                self._set_active(target, now)
                changed = True
                triggered = True
            else:
                if self._pending_level == target:
                    self._pending_frames += 1
                else:
                    self._pending_level = target
                    self._pending_frames = 1
                if (
                    self._pending_frames
                    >= self.config.attention_debounce_frames
                ):
                    self._set_active(target, now)
                    changed = True
                    triggered = True
        elif target < self._active_level:
            self._pending_level = DangerLevel.INFORMATION
            self._pending_frames = 0
            self._release_frames += 1
            if self._release_frames >= self.config.release_frames:
                self._set_active(target, now)
                changed = True
        else:
            self._pending_level = target
            self._pending_frames = 0
            self._release_frames = 0

        sound = self._select_sound(target, now, triggered)
        flash_on = (
            self._active_level != DangerLevel.INFORMATION
            and int(now * self.config.flash_hz) % 2 == 0
        )
        duration = (
            0.0
            if self._active_level == DangerLevel.INFORMATION
            else max(0.0, now - self._since)
        )
        return AlertEvent(
            level=self._active_level,
            changed=changed,
            sound=sound,
            muted=self._muted,
            flash_on=flash_on,
            duration=duration,
        )

    def reset(self) -> None:
        self._active_level = DangerLevel.INFORMATION
        self._pending_level = DangerLevel.INFORMATION
        self._pending_frames = 0
        self._release_frames = 0
        self._last_sound.clear()
        self._counts = {level: 0 for level in DangerLevel}
        self._since = 0.0

    def close(self) -> None:
        if self._closed:
            return
        self._backend.close()
        self._closed = True

    def _set_active(self, level: DangerLevel, now: float) -> None:
        self._active_level = level
        self._pending_level = level
        self._pending_frames = 0
        self._release_frames = 0
        self._since = now
        if level != DangerLevel.INFORMATION:
            self._counts[level] += 1

    def _select_sound(
        self, target: DangerLevel, now: float, triggered: bool
    ) -> Optional[str]:
        if self._active_level == DangerLevel.INFORMATION:
            return None

        pattern = (
            CRITICAL_SOUND
            if self._active_level == DangerLevel.CRITIQUE
            else ATTENTION_SOUND
        )
        should_play = triggered
        if not should_play and target >= self._active_level:
            cooldown = (
                self.config.critical_cooldown
                if self._active_level == DangerLevel.CRITIQUE
                else self.config.attention_cooldown
            )
            last_sound = self._last_sound.get(self._active_level)
            should_play = (
                last_sound is None or now - last_sound >= cooldown
            )

        if not should_play:
            return None
        if self._muted:
            return None

        self._last_sound[self._active_level] = now
        self._backend.play(pattern)
        return pattern
