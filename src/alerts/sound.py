"""Sound backend implementations for alert tones."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Protocol

try:
    import winsound
except ImportError:
    winsound = None

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

    def play(self, _pattern: str) -> None:
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
            else Path(__file__).resolve().parents[2] / "assets" / "sounds"
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
