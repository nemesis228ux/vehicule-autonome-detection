import unittest

import numpy as np

from src.detector import Detection
from src.hud import draw_risk_hud
from src.risk_evaluator import (
    DangerLevel,
    RiskConfig,
    RiskEvaluator,
)
from src.tracker import TrackedObject

FRAME_SHAPE = (720, 1280, 3)
FPS = 30.0


def make_object(
    track_id=1,
    center_x=640,
    y2=450,
    height=200,
    width=100,
    category="pieton",
    confidence=0.9,
    hits=5,
    age=5,
    vx=0.0,
    vy=0.0,
    time_since_update=0,
    trail=None,
):
    x1 = int(center_x - width / 2)
    x2 = x1 + width
    y1 = int(y2 - height)
    class_id = {
        "pieton": 0,
        "deux_roues": 1,
        "vehicule": 2,
        "nid_de_poule": 3,
    }.get(category, 99)
    detection = Detection(
        x1,
        y1,
        x2,
        y2,
        confidence,
        class_id,
        category,
        category,
        track_id,
    )
    return TrackedObject(
        track_id=track_id,
        detection=detection,
        age=age,
        hits=hits,
        time_since_update=time_since_update,
        velocity=(vx, vy),
        trail=tuple(trail or [detection.center]),
    )


def approaching_sequence(category="pieton", **kwargs):
    heights = (200, 202, 205, 209)
    bottoms = (450, 458, 466, 474)
    return [
        make_object(
            y2=bottom,
            height=height,
            category=category,
            **kwargs,
        )
        for height, bottom in zip(heights, bottoms)
    ]


class RiskConfigTests(unittest.TestCase):
    def test_invalid_configuration_raises(self):
        with self.assertRaises(ValueError):
            RiskConfig(horizon_ratio=1.5)
        with self.assertRaises(ValueError):
            RiskConfig(corridor_near_half_width=0.1, corridor_far_half_width=0.2)
        with self.assertRaises(ValueError):
            RiskConfig(ttc_attention=0.0)
        with self.assertRaises(ValueError):
            RiskConfig(min_hits=0)
        with self.assertRaises(ValueError):
            RiskConfig(min_confidence=1.5)
        with self.assertRaises(ValueError):
            RiskConfig(release_frames=-1)
        with self.assertRaises(ValueError):
            RiskConfig(score_smoothing=1.5)
        with self.assertRaises(ValueError):
            RiskConfig(ttc_smoothing=-0.1)


class RiskEvaluatorTests(unittest.TestCase):
    def config(self, **kwargs):
        defaults = {
            "ttc_smoothing": 0.0,
            "score_smoothing": 0.0,
            "release_frames": 3,
        }
        defaults.update(kwargs)
        return RiskConfig(**defaults)

    def test_pedestrian_approaching_in_path_escalates(self):
        evaluator = RiskEvaluator(self.config())
        levels = []
        for obj in approaching_sequence():
            summary = evaluator.evaluate([obj], FRAME_SHAPE, FPS)
            levels.append(summary.level)

        self.assertEqual(
            levels,
            [
                DangerLevel.INFORMATION,
                DangerLevel.INFORMATION,
                DangerLevel.ATTENTION,
                DangerLevel.CRITIQUE,
            ],
        )
        self.assertEqual(summary.primary.track_id, 1)
        self.assertEqual(summary.counts[DangerLevel.CRITIQUE], 1)

    def test_pedestrian_out_of_corridor_stays_information(self):
        evaluator = RiskEvaluator(self.config())
        levels = [
            evaluator.evaluate([obj], FRAME_SHAPE, FPS).level
            for obj in approaching_sequence(center_x=100)
        ]

        self.assertEqual(set(levels), {DangerLevel.INFORMATION})

    def test_vulnerable_crossing_raises_attention(self):
        evaluator = RiskEvaluator(self.config())
        obj = make_object(
            center_x=1100,
            y2=550,
            height=180,
            width=80,
            vx=-25.0,
        )

        summary = evaluator.evaluate([obj], FRAME_SHAPE, FPS)

        self.assertEqual(summary.level, DangerLevel.ATTENTION)
        self.assertIn("traversee imminente", summary.primary.reasons)

    def test_pothole_ahead_becomes_critical(self):
        evaluator = RiskEvaluator(self.config())
        levels = [
            evaluator.evaluate([obj], FRAME_SHAPE, FPS).level
            for obj in approaching_sequence(category="nid_de_poule")
        ]

        self.assertEqual(levels[-1], DangerLevel.CRITIQUE)

    def test_young_track_is_never_critical(self):
        evaluator = RiskEvaluator(self.config())
        objects = approaching_sequence(hits=1)

        levels = [
            evaluator.evaluate([obj], FRAME_SHAPE, FPS).level
            for obj in objects
        ]

        self.assertNotIn(DangerLevel.CRITIQUE, levels)
        self.assertIn(DangerLevel.ATTENTION, levels)

    def test_low_confidence_is_never_critical(self):
        evaluator = RiskEvaluator(self.config())
        objects = approaching_sequence(confidence=0.2)

        levels = [
            evaluator.evaluate([obj], FRAME_SHAPE, FPS).level
            for obj in objects
        ]

        self.assertNotIn(DangerLevel.CRITIQUE, levels)

    def test_stale_track_is_never_critical(self):
        evaluator = RiskEvaluator(self.config())
        objects = approaching_sequence(time_since_update=1)

        levels = [
            evaluator.evaluate([obj], FRAME_SHAPE, FPS).level
            for obj in objects
        ]

        self.assertNotIn(DangerLevel.CRITIQUE, levels)

    def test_hysteresis_holds_critical_then_releases(self):
        evaluator = RiskEvaluator(self.config(release_frames=3))

        for obj in approaching_sequence():
            summary = evaluator.evaluate([obj], FRAME_SHAPE, FPS)
        self.assertEqual(summary.level, DangerLevel.CRITIQUE)

        far = make_object(y2=400, height=200, hits=10)
        levels = [
            evaluator.evaluate([far], FRAME_SHAPE, FPS).level for _ in range(3)
        ]

        self.assertEqual(
            levels,
            [
                DangerLevel.CRITIQUE,
                DangerLevel.CRITIQUE,
                DangerLevel.INFORMATION,
            ],
        )

    def test_state_is_purged_after_max_age(self):
        evaluator = RiskEvaluator(
            self.config(state_max_age=1, release_frames=10)
        )
        for obj in approaching_sequence():
            summary = evaluator.evaluate([obj], FRAME_SHAPE, FPS)
        self.assertEqual(summary.level, DangerLevel.CRITIQUE)

        evaluator.evaluate([], FRAME_SHAPE, FPS)
        evaluator.evaluate([], FRAME_SHAPE, FPS)

        far = make_object(y2=400, height=200, hits=10)
        summary = evaluator.evaluate([far], FRAME_SHAPE, FPS)
        self.assertEqual(summary.level, DangerLevel.INFORMATION)

    def test_reset_clears_history(self):
        evaluator = RiskEvaluator(self.config())
        for obj in approaching_sequence():
            evaluator.evaluate([obj], FRAME_SHAPE, FPS)

        evaluator.reset()
        far = make_object(y2=400, height=200)
        summary = evaluator.evaluate([far], FRAME_SHAPE, FPS)

        self.assertEqual(summary.level, DangerLevel.INFORMATION)

    def test_empty_frame_returns_safe_summary(self):
        evaluator = RiskEvaluator(self.config())

        summary = evaluator.evaluate([], FRAME_SHAPE, FPS)

        self.assertEqual(summary.level, DangerLevel.INFORMATION)
        self.assertIsNone(summary.primary)
        self.assertEqual(summary.assessments, ())
        self.assertEqual(summary.counts[DangerLevel.CRITIQUE], 0)

    def test_invalid_frame_shape_and_fps_raise(self):
        evaluator = RiskEvaluator(self.config())

        with self.assertRaises(ValueError):
            evaluator.evaluate([], (0, 1280), FPS)
        with self.assertRaises(ValueError):
            evaluator.evaluate([], FRAME_SHAPE, 0.0)

    def test_ttc_and_proximity_are_exposed(self):
        evaluator = RiskEvaluator(self.config())
        for obj in approaching_sequence():
            summary = evaluator.evaluate([obj], FRAME_SHAPE, FPS)

        primary = summary.primary
        self.assertIsNotNone(primary.ttc)
        self.assertAlmostEqual(primary.ttc, 1.0 / (209 / 205 - 1.0) / FPS)
        self.assertGreater(primary.proximity, 0.0)
        self.assertTrue(primary.in_path)


class RiskHudTests(unittest.TestCase):
    def test_hud_draws_on_large_frame(self):
        evaluator = RiskEvaluator(
            RiskConfig(ttc_smoothing=0.0, score_smoothing=0.0)
        )
        objects = approaching_sequence()
        summary = evaluator.evaluate([objects[-1]], FRAME_SHAPE, FPS)
        frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)

        result = draw_risk_hud(
            frame, [objects[-1]], summary, fps=25.0, frame_index=0
        )

        self.assertIs(result, frame)
        self.assertTrue(np.any(frame > 0))

    def test_hud_draws_safe_state(self):
        evaluator = RiskEvaluator()
        summary = evaluator.evaluate([], FRAME_SHAPE, FPS)
        frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)

        draw_risk_hud(frame, [], summary)

        self.assertTrue(np.any(frame > 0))

    def test_hud_draws_compact_frame(self):
        evaluator = RiskEvaluator(
            RiskConfig(ttc_smoothing=0.0, score_smoothing=0.0)
        )
        objects = approaching_sequence()
        summary = evaluator.evaluate(
            [objects[-1]], (426, 240, 3), FPS
        )
        frame = np.zeros((426, 240, 3), dtype=np.uint8)

        draw_risk_hud(frame, [objects[-1]], summary, frame_index=2)

        self.assertTrue(np.any(frame > 0))


if __name__ == "__main__":
    unittest.main()
