import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.alerts import (
    ATTENTION_SOUND,
    CRITICAL_SOUND,
    AlertConfig,
    AlertEvent,
    AlertManager,
    NullSoundBackend,
    WinsoundBackend,
    draw_alert_overlay,
)
from src.demo_detection import parse_args
from src.risk_evaluator import DangerLevel, RiskSummary


def make_summary(level):
    counts = {item: 0 for item in DangerLevel}
    counts[level] = 1
    return RiskSummary(level, None, (), counts)


class FakeSoundBackend:
    def __init__(self, available=True):
        self.available = available
        self.played = []
        self.stop_count = 0
        self.closed = False

    def play(self, pattern):
        self.played.append(pattern)

    def stop(self):
        self.stop_count += 1

    def close(self):
        self.closed = True


class AlertConfigTests(unittest.TestCase):
    def test_invalid_configuration_raises(self):
        with self.assertRaises(ValueError):
            AlertConfig(attention_debounce_frames=0)
        with self.assertRaises(ValueError):
            AlertConfig(release_frames=0)
        with self.assertRaises(ValueError):
            AlertConfig(attention_cooldown=0.0)
        with self.assertRaises(ValueError):
            AlertConfig(critical_cooldown=0.0)
        with self.assertRaises(ValueError):
            AlertConfig(flash_hz=0.0)


class AlertManagerTests(unittest.TestCase):
    def make_manager(self, **config):
        backend = FakeSoundBackend()
        manager = AlertManager(AlertConfig(**config), backend)
        return manager, backend

    def test_information_is_silent(self):
        manager, backend = self.make_manager()

        event = manager.update(make_summary(DangerLevel.INFORMATION), now_s=0.0)

        self.assertEqual(event.level, DangerLevel.INFORMATION)
        self.assertIsNone(event.sound)
        self.assertFalse(event.changed)
        self.assertEqual(backend.played, [])

    def test_attention_is_debounced(self):
        manager, backend = self.make_manager(
            attention_debounce_frames=2, release_frames=3
        )

        first = manager.update(
            make_summary(DangerLevel.ATTENTION), now_s=0.1
        )
        second = manager.update(
            make_summary(DangerLevel.ATTENTION), now_s=0.2
        )

        self.assertEqual(first.level, DangerLevel.INFORMATION)
        self.assertIsNone(first.sound)
        self.assertEqual(second.level, DangerLevel.ATTENTION)
        self.assertTrue(second.changed)
        self.assertEqual(second.sound, ATTENTION_SOUND)
        self.assertEqual(backend.played, [ATTENTION_SOUND])

    def test_critical_is_immediate(self):
        manager, backend = self.make_manager()

        event = manager.update(
            make_summary(DangerLevel.CRITIQUE), now_s=0.0
        )

        self.assertEqual(event.level, DangerLevel.CRITIQUE)
        self.assertTrue(event.changed)
        self.assertEqual(event.sound, CRITICAL_SOUND)
        self.assertEqual(backend.played, [CRITICAL_SOUND])
        self.assertEqual(manager.counts[DangerLevel.CRITIQUE], 1)

    def test_critical_reminder_respects_cooldown(self):
        manager, backend = self.make_manager(critical_cooldown=1.2)

        manager.update(make_summary(DangerLevel.CRITIQUE), now_s=0.0)
        before = manager.update(
            make_summary(DangerLevel.CRITIQUE), now_s=1.1
        )
        after = manager.update(
            make_summary(DangerLevel.CRITIQUE), now_s=1.3
        )

        self.assertIsNone(before.sound)
        self.assertEqual(after.sound, CRITICAL_SOUND)
        self.assertEqual(backend.played, [CRITICAL_SOUND, CRITICAL_SOUND])

    def test_attention_reminder_respects_cooldown(self):
        manager, backend = self.make_manager(
            attention_debounce_frames=1, attention_cooldown=3.0
        )

        manager.update(make_summary(DangerLevel.ATTENTION), now_s=0.0)
        before = manager.update(
            make_summary(DangerLevel.ATTENTION), now_s=2.9
        )
        after = manager.update(
            make_summary(DangerLevel.ATTENTION), now_s=3.1
        )

        self.assertIsNone(before.sound)
        self.assertEqual(after.sound, ATTENTION_SOUND)
        self.assertEqual(backend.played, [ATTENTION_SOUND, ATTENTION_SOUND])

    def test_escalation_triggers_critical_horn(self):
        manager, backend = self.make_manager(attention_debounce_frames=2)

        manager.update(make_summary(DangerLevel.ATTENTION), now_s=0.1)
        manager.update(make_summary(DangerLevel.ATTENTION), now_s=0.2)
        event = manager.update(
            make_summary(DangerLevel.CRITIQUE), now_s=0.25
        )

        self.assertEqual(event.level, DangerLevel.CRITIQUE)
        self.assertTrue(event.changed)
        self.assertEqual(event.sound, CRITICAL_SOUND)
        self.assertEqual(
            backend.played, [ATTENTION_SOUND, CRITICAL_SOUND]
        )

    def test_deescalation_is_silent(self):
        manager, backend = self.make_manager(release_frames=3)

        manager.update(make_summary(DangerLevel.CRITIQUE), now_s=0.0)
        held_one = manager.update(
            make_summary(DangerLevel.INFORMATION), now_s=0.1
        )
        held_two = manager.update(
            make_summary(DangerLevel.INFORMATION), now_s=0.2
        )
        released = manager.update(
            make_summary(DangerLevel.INFORMATION), now_s=0.3
        )

        self.assertEqual(held_one.level, DangerLevel.CRITIQUE)
        self.assertEqual(held_two.level, DangerLevel.CRITIQUE)
        self.assertEqual(released.level, DangerLevel.INFORMATION)
        self.assertTrue(released.changed)
        self.assertIsNone(released.sound)
        self.assertEqual(backend.played, [CRITICAL_SOUND])

    def test_empty_scene_clears_alert_and_stops_sound_immediately(self):
        manager, backend = self.make_manager(release_frames=10)
        manager.update(make_summary(DangerLevel.CRITIQUE), now_s=0.0)
        empty_scene = RiskSummary(
            DangerLevel.INFORMATION,
            None,
            (),
            {level: 0 for level in DangerLevel},
        )

        event = manager.update(empty_scene, now_s=0.1)

        self.assertEqual(event.level, DangerLevel.INFORMATION)
        self.assertTrue(event.changed)
        self.assertEqual(event.duration, 0.0)
        self.assertEqual(backend.stop_count, 1)

        repeated_empty = manager.update(empty_scene, now_s=0.2)
        self.assertFalse(repeated_empty.changed)
        self.assertEqual(backend.stop_count, 1)

    def test_mute_blocks_then_unmute_restores(self):
        backend = FakeSoundBackend()
        manager = AlertManager(sound_backend=backend, muted=True)

        muted_event = manager.update(
            make_summary(DangerLevel.CRITIQUE), now_s=0.0
        )
        self.assertIsNone(muted_event.sound)
        self.assertTrue(muted_event.muted)
        self.assertEqual(backend.played, [])

        self.assertFalse(manager.toggle_mute())
        unmuted_event = manager.update(
            make_summary(DangerLevel.CRITIQUE), now_s=0.1
        )
        self.assertEqual(unmuted_event.sound, CRITICAL_SOUND)
        self.assertEqual(backend.played, [CRITICAL_SOUND])

    def test_reset_clears_state_and_counts(self):
        manager, backend = self.make_manager()
        manager.update(make_summary(DangerLevel.CRITIQUE), now_s=0.0)

        manager.reset()
        event = manager.update(
            make_summary(DangerLevel.INFORMATION), now_s=1.0
        )

        self.assertEqual(event.level, DangerLevel.INFORMATION)
        self.assertEqual(
            manager.counts,
            {level: 0 for level in DangerLevel},
        )
        self.assertFalse(backend.closed)

    def test_close_calls_backend_once(self):
        manager, backend = self.make_manager()

        manager.close()
        manager.close()

        self.assertTrue(backend.closed)

    def test_failed_sound_backend_does_not_raise(self):
        backend = FakeSoundBackend(available=False)
        manager = AlertManager(sound_backend=backend)

        event = manager.update(
            make_summary(DangerLevel.CRITIQUE), now_s=0.0
        )

        self.assertEqual(event.sound, CRITICAL_SOUND)
        self.assertEqual(backend.played, [CRITICAL_SOUND])


class SoundBackendTests(unittest.TestCase):
    def test_null_backend_is_silent(self):
        backend = NullSoundBackend()
        backend.play(CRITICAL_SOUND)
        backend.close()
        self.assertFalse(backend.available)

    def test_null_backend_can_stop(self):
        backend = NullSoundBackend()
        backend.stop()

    def test_winsound_backend_reports_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = WinsoundBackend(Path(directory))
            backend.play(CRITICAL_SOUND)
            backend.close()

        self.assertFalse(backend.available)
        self.assertEqual(len(backend.missing_patterns), 2)
        self.assertEqual(len(backend.missing_files), 4)

    def test_mp3_is_preferred_over_wav(self):
        with tempfile.TemporaryDirectory() as directory:
            sounds_dir = Path(directory)
            (sounds_dir / "klaxon_attention.mp3").write_bytes(b"")
            (sounds_dir / "klaxon_attention.wav").write_bytes(b"")
            with patch.object(
                WinsoundBackend, "_create_mci_sender", return_value=None
            ):
                backend = WinsoundBackend(sounds_dir)
            resolved = backend.resolved_files
            backend.close()

        self.assertEqual(resolved[ATTENTION_SOUND].suffix, ".mp3")


class AlertOverlayTests(unittest.TestCase):
    def make_event(self, level, flash_on=True, muted=False):
        return AlertEvent(
            level=level,
            changed=True,
            sound=None,
            muted=muted,
            flash_on=flash_on,
            duration=1.0,
        )

    def test_information_does_not_draw(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        draw_alert_overlay(
            frame, self.make_event(DangerLevel.INFORMATION)
        )
        self.assertFalse(np.any(frame > 0))

    def test_attention_draws_border(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        draw_alert_overlay(frame, self.make_event(DangerLevel.ATTENTION))
        self.assertTrue(np.any(frame > 0))

    def test_critical_flashes(self):
        dark = np.zeros((240, 320, 3), dtype=np.uint8)
        lit = np.zeros((240, 320, 3), dtype=np.uint8)

        draw_alert_overlay(
            dark,
            self.make_event(DangerLevel.CRITIQUE, flash_on=False),
        )
        draw_alert_overlay(
            lit,
            self.make_event(DangerLevel.CRITIQUE, flash_on=True),
        )

        self.assertFalse(np.any(dark > 0))
        self.assertTrue(np.any(lit > 0))

    def test_muted_label_is_drawn(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        draw_alert_overlay(
            frame,
            self.make_event(
                DangerLevel.CRITIQUE, flash_on=False, muted=True
            ),
        )
        self.assertTrue(np.any(frame > 0))

    def test_small_frame_does_not_crash(self):
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        draw_alert_overlay(frame, self.make_event(DangerLevel.ATTENTION))
        self.assertTrue(np.any(frame > 0))


class DemoArgumentsTests(unittest.TestCase):
    def test_alerts_implies_risk_and_tracking(self):
        args = parse_args(["--alerts"])
        self.assertTrue(args.alerts)
        self.assertTrue(args.risk)
        self.assertTrue(args.track)

    def test_no_sound_is_kept(self):
        args = parse_args(["--alerts", "--no-sound"])
        self.assertTrue(args.no_sound)


if __name__ == "__main__":
    unittest.main()
