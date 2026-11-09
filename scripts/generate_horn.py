"""Génère les klaxons synthétiques utilisés par le module d'alertes."""

from __future__ import annotations

from pathlib import Path
import wave

import numpy as np

SAMPLE_RATE = 44100


def _envelope(length: int, attack: float = 0.005, release: float = 0.03) -> np.ndarray:
    attack_samples = min(max(1, int(attack * SAMPLE_RATE)), length)
    release_samples = min(max(1, int(release * SAMPLE_RATE)), length)
    envelope = np.ones(length, dtype=np.float32)
    envelope[:attack_samples] = np.linspace(
        0.0, 1.0, attack_samples, dtype=np.float32
    )
    envelope[-release_samples:] = np.linspace(
        1.0, 0.0, release_samples, dtype=np.float32
    )
    return envelope


def _honk(duration: float, amplitude: float) -> np.ndarray:
    time = np.arange(int(SAMPLE_RATE * duration)) / SAMPLE_RATE
    signal = np.zeros_like(time)
    for frequency, weight in ((392.0, 0.6), (494.0, 0.5)):
        signal += weight * np.sin(2 * np.pi * frequency * time)
        signal += 0.35 * weight * np.sin(2 * np.pi * 2 * frequency * time)
        signal += 0.15 * weight * np.sin(2 * np.pi * 3 * frequency * time)
    signal = np.tanh(1.6 * signal)
    signal *= _envelope(signal.size)
    peak = float(np.max(np.abs(signal))) or 1.0
    return (signal / peak * amplitude).astype(np.float32)


def _silence(duration: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * duration), dtype=np.float32)


def _write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm.tobytes())


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "assets" / "sounds"
    attention = _honk(0.35, 0.55)
    critical = np.concatenate(
        (
            _honk(0.18, 0.85),
            _silence(0.12),
            _honk(0.18, 0.85),
            _silence(0.12),
            _honk(0.18, 0.85),
        )
    )
    _write_wav(output_dir / "klaxon_attention.wav", attention)
    _write_wav(output_dir / "klaxon_critique.wav", critical)
    print(f"Klaxons générés dans {output_dir}")


if __name__ == "__main__":
    main()
