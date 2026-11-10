"""Alertes visuelles et sonores pour les situations à risque."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Protocol

import cv2
import numpy as np

try:
    import winsound
except ImportError:
    winsound = None

try:
    from src.risk_evaluator import DANGER_COLORS, DangerLevel, RiskSummary
except ModuleNotFoundError:
    from risk_evaluator import DANGER_COLORS, DangerLevel, RiskSummary

ATTENTION_SOUND = "klaxon_attention"
CRITICAL_SOUND = "klaxon_critique"

SOUND_CANDIDATES: Dict[str, tuple[str, ...]] = {
    ATTENTION_SOUND: ("klaxon_attention.mp3", "klaxon_attention.wav"),
    CRITICAL_SOUND: ("klaxon_critique.mp3", "klaxon_critique.wav"),
}


class SoundBackend(Protocol):
    """Interface minimale pour jouer un motif sonore."""

    @property
    def available(self) -> bool:
        ...

    def play(self, pattern: str) -> None:
        ...

    def close(self) -> None:
        ...


class NullSoundBackend:
    """Backend silencieux, utilisé hors Windows ou avec ``--no-sound``."""

    available = False

    def play(self, pattern: str) -> None:
        return None

    def close(self) -> None:
        return None


class WinsoundBackend:
    """Joue les klaxons sous Windows.

    Les MP3 sont lus via MCI (asynchrone, sans dépendance externe) ; les WAV
    restent lus par ``winsound``. Les MP3 sont prioritaires s'ils existent.
    """

    def __init__(self, sounds_dir: Optional[Path] = None) -> None:
        self._sounds_dir = (
            Path(sounds_dir)
            if sounds_dir is not None
            else Path(__file__).resolve().parents[1] / "assets" / "sounds"
        )
        self._resolved: Dict[str, Path] = {}
        for pattern, filenames in SOUND_CANDIDATES.items():
            for filename in filenames:
                path = self._sounds_dir / filename
                if path.is_file():
                    self._resolved[pattern] = path
                    break
        self._missing_patterns = tuple(
            pattern
            for pattern in SOUND_CANDIDATES
            if pattern not in self._resolved
        )
        self._platform_supported = sys.platform == "win32" and winsound is not None
        self._available = self._platform_supported and bool(self._resolved)
        self._failed: set[str] = set()
        self._mci_send = self._create_mci_sender()
        self._mci_aliases: Dict[str, str] = {}
        if self._platform_supported:
            for pattern, path in self._resolved.items():
                if path.suffix.lower() == ".mp3":
                    self._open_mci(pattern, path)
            self._available = any(
                self._is_playable(pattern) for pattern in self._resolved
            )

    @property
    def available(self) -> bool:
        return self._available

    @property
    def missing_patterns(self) -> tuple[str, ...]:
        return self._missing_patterns

    @property
    def missing_files(self) -> tuple[str, ...]:
        return tuple(
            str(self._sounds_dir / filename)
            for filenames in SOUND_CANDIDATES.values()
            for filename in filenames
            if not (self._sounds_dir / filename).is_file()
        )

    @property
    def resolved_files(self) -> Dict[str, Path]:
        return dict(self._resolved)

    def play(self, pattern: str) -> None:
        if not self._available or pattern in self._failed:
            return

        path = self._resolved.get(pattern)
        if path is None:
            return

        if path.suffix.lower() == ".mp3" and pattern in self._mci_aliases:
            self._stop_winsound()
            self._play_mci(pattern)
            return

        if path.suffix.lower() == ".mp3":
            fallback = self._wav_fallback(pattern)
            if fallback is None:
                self._failed.add(pattern)
                return
            path = fallback

        self._stop_mci()
        if winsound is None:
            return
        flags = (
            winsound.SND_FILENAME
            | winsound.SND_ASYNC
            | winsound.SND_NODEFAULT
        )
        try:
            winsound.PlaySound(str(path), flags)
        except (OSError, RuntimeError):
            self._failed.add(pattern)

    def close(self) -> None:
        self._stop_winsound()
        self._stop_mci()
        for alias in self._mci_aliases.values():
            if self._mci_send is not None:
                self._mci_send(f"close {alias}")
        self._mci_aliases.clear()

    def _create_mci_sender(self):
        if sys.platform != "win32":
            return None
        try:
            import ctypes
            from ctypes import wintypes

            winmm = ctypes.WinDLL("winmm")
            sender = winmm.mciSendStringW
            sender.argtypes = [
                wintypes.LPCWSTR,
                wintypes.LPWSTR,
                wintypes.UINT,
                wintypes.HWND,
            ]
            sender.restype = wintypes.UINT

            def send(command: str) -> bool:
                return sender(command, None, 0, None) == 0

            return send
        except (AttributeError, OSError):
            return None

    def _open_mci(self, pattern: str, path: Path) -> None:
        if self._mci_send is None:
            return
        alias = f"riskalert_{id(self):x}_{pattern}"
        if self._mci_send(f'open "{path}" type mpegvideo alias {alias}'):
            self._mci_aliases[pattern] = alias

    def _wav_fallback(self, pattern: str) -> Optional[Path]:
        for filename in SOUND_CANDIDATES.get(pattern, ()):
            path = self._sounds_dir / filename
            if path.suffix.lower() == ".wav" and path.is_file():
                return path
        return None

    def _is_playable(self, pattern: str) -> bool:
        path = self._resolved.get(pattern)
        if path is None:
            return False
        if path.suffix.lower() != ".mp3":
            return True
        return pattern in self._mci_aliases or self._wav_fallback(pattern) is not None

    def _play_mci(self, pattern: str) -> None:
        if self._mci_send is None:
            return
        alias = self._mci_aliases.get(pattern)
        if alias is None:
            self._failed.add(pattern)
            return
        for other_pattern, other_alias in self._mci_aliases.items():
            if other_pattern != pattern:
                self._mci_send(f"stop {other_alias}")
        self._mci_send(f"stop {alias}")
        if not self._mci_send(f"seek {alias} to start"):
            self._mci_send(f"play {alias} from 0")
        else:
            self._mci_send(f"play {alias}")

    def _stop_mci(self) -> None:
        if self._mci_send is None:
            return
        for alias in self._mci_aliases.values():
            self._mci_send(f"stop {alias}")

    def _stop_winsound(self) -> None:
        if winsound is None:
            return
        try:
            winsound.PlaySound(None, 0)
        except (OSError, RuntimeError):
            pass


def build_sound_backend(enabled: bool = True) -> SoundBackend:
    if not enabled:
        return NullSoundBackend()
    return WinsoundBackend()


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


def draw_alert_overlay(
    frame: np.ndarray,
    event: AlertEvent,
    thickness: Optional[int] = None,
) -> np.ndarray:
    """Dessine une bordure d'alerte et l'état du son."""
    if event.level == DangerLevel.INFORMATION:
        return frame

    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        return frame

    color = DANGER_COLORS[event.level]
    if thickness is None:
        base = max(2, int(min(width, height) * 0.008))
        thickness = base if event.level == DangerLevel.ATTENTION else base * 2
    thickness = max(1, int(thickness))

    if event.level == DangerLevel.ATTENTION or event.flash_on:
        offset = thickness // 2
        cv2.rectangle(
            frame,
            (offset, offset),
            (max(offset, width - 1 - offset), max(offset, height - 1 - offset)),
            color,
            thickness,
            cv2.LINE_AA,
        )

    if event.muted:
        scale = max(0.4, min(0.8, min(width, height) / 500.0))
        text_thickness = max(1, int(scale * 2))
        position = (10, max(18, height - 12))
        cv2.putText(
            frame,
            "SON COUPE",
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            text_thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "SON COUPE",
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )
    return frame
