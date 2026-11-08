import unittest

import numpy as np

from src.detector import Detection
from src.tracker import TrackedObject, Tracker, draw_tracks


def make_detection(
    x1,
    y1,
    x2,
    y2,
    track_id,
    confidence=0.9,
    category="vehicule",
    class_name="car",
    class_id=2,
):
    return Detection(
        x1,
        y1,
        x2,
        y2,
        confidence,
        class_id,
        class_name,
        category,
        track_id,
    )


class FakeDetector:
    confidence_threshold = 0.35

    def __init__(self, detections):
        self.detections = detections
        self.calls = []

    def track(self, frame, tracker="bytetrack.yaml"):
        self.calls.append((frame, tracker))
        return list(self.detections)


class TrackerTests(unittest.TestCase):
    def test_same_id_updates_velocity_and_trail(self):
        tracker = Tracker(FakeDetector([]))

        first = tracker.update_detections(
            [make_detection(0, 0, 10, 10, 1)]
        )[0]
        second = tracker.update_detections(
            [make_detection(2, 0, 12, 10, 1)]
        )[0]

        self.assertEqual(first.velocity, (0.0, 0.0))
        self.assertEqual(first.trail, ((5, 5),))
        self.assertEqual((first.age, first.hits), (1, 1))

        self.assertAlmostEqual(second.velocity[0], 0.8)
        self.assertAlmostEqual(second.velocity[1], 0.0)
        self.assertEqual(second.trail, ((5, 5), (7, 5)))
        self.assertEqual((second.age, second.hits), (2, 2))

    def test_multiple_ids_are_independent(self):
        tracker = Tracker(FakeDetector([]))

        objects = tracker.update_detections(
            [
                make_detection(0, 0, 10, 10, 1),
                make_detection(50, 50, 60, 60, 2),
            ]
        )

        self.assertEqual([obj.track_id for obj in objects], [1, 2])
        self.assertEqual(
            [obj.trail for obj in objects], [((5, 5),), ((55, 55),)]
        )

    def test_missing_track_expires_after_max_age(self):
        tracker = Tracker(FakeDetector([]), max_age=2)
        tracker.update_detections([make_detection(0, 0, 10, 10, 1)])

        for _ in range(3):
            tracker.update_detections([])

        refreshed = tracker.update_detections(
            [make_detection(100, 100, 110, 110, 1)]
        )[0]

        self.assertEqual(refreshed.velocity, (0.0, 0.0))
        self.assertEqual(refreshed.trail, ((105, 105),))
        self.assertEqual((refreshed.age, refreshed.hits), (1, 1))

    def test_track_survives_until_max_age(self):
        tracker = Tracker(
            FakeDetector([]), max_age=2, velocity_smoothing=0.0
        )
        tracker.update_detections([make_detection(0, 0, 10, 10, 1)])

        tracker.update_detections([])
        tracker.update_detections([])

        kept = tracker.update_detections(
            [make_detection(2, 0, 12, 10, 1)]
        )[0]

        self.assertAlmostEqual(kept.velocity[0], 2.0 / 3.0)
        self.assertEqual(len(kept.trail), 2)

    def test_trail_length_is_capped(self):
        tracker = Tracker(FakeDetector([]), trail_length=3)

        for x in (0, 10, 20, 30):
            obj = tracker.update_detections(
                [make_detection(x, 0, x + 10, 10, 1)]
            )[0]

        self.assertEqual(len(obj.trail), 3)
        self.assertEqual(obj.trail[0], (15, 5))
        self.assertEqual(obj.trail[-1], (35, 5))

    def test_low_confidence_is_hidden_but_state_updated(self):
        tracker = Tracker(FakeDetector([]), min_confidence=0.5)

        tracker.update_detections(
            [make_detection(0, 0, 10, 10, 1, confidence=0.9)]
        )
        hidden = tracker.update_detections(
            [make_detection(10, 0, 20, 10, 1, confidence=0.2)]
        )
        shown = tracker.update_detections(
            [make_detection(20, 0, 30, 10, 1, confidence=0.9)]
        )

        self.assertEqual(hidden, [])
        self.assertEqual(len(shown), 1)
        self.assertAlmostEqual(shown[0].velocity[0], 6.4)

    def test_detections_without_track_id_are_ignored(self):
        tracker = Tracker(FakeDetector([]))

        objects = tracker.update_detections(
            [make_detection(0, 0, 10, 10, None)]
        )

        self.assertEqual(objects, [])

    def test_reset_clears_state(self):
        tracker = Tracker(FakeDetector([]))
        tracker.update_detections([make_detection(0, 0, 10, 10, 1)])

        tracker.reset()
        obj = tracker.update_detections(
            [make_detection(100, 100, 110, 110, 1)]
        )[0]

        self.assertEqual(obj.velocity, (0.0, 0.0))
        self.assertEqual(obj.trail, ((105, 105),))

    def test_update_calls_detector_track(self):
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        detector = FakeDetector([make_detection(0, 0, 10, 10, 3)])
        tracker = Tracker(detector, tracker_config="botsort.yaml")

        objects = tracker.update(frame)

        self.assertEqual(len(objects), 1)
        self.assertIs(detector.calls[0][0], frame)
        self.assertEqual(detector.calls[0][1], "botsort.yaml")

    def test_invalid_configuration_raises(self):
        detector = FakeDetector([])

        with self.assertRaises(ValueError):
            Tracker(detector, trail_length=0)
        with self.assertRaises(ValueError):
            Tracker(detector, max_age=-1)
        with self.assertRaises(ValueError):
            Tracker(detector, velocity_smoothing=1.5)
        with self.assertRaises(ValueError):
            Tracker(detector, min_confidence=1.5)

    def test_speed_and_draw_tracks(self):
        detection = make_detection(10, 10, 50, 50, 1)
        obj = TrackedObject(
            track_id=1,
            detection=detection,
            age=2,
            hits=2,
            time_since_update=0,
            velocity=(3.0, 4.0),
            trail=((12, 12), (14, 14)),
        )
        frame = np.zeros((80, 80, 3), dtype=np.uint8)

        result = draw_tracks(frame, [obj])

        self.assertAlmostEqual(obj.speed, 5.0)
        self.assertIs(result, frame)
        self.assertTrue(np.any(frame > 0))

    def test_draw_tracks_labels_modes(self):
        detection = make_detection(10, 10, 50, 50, 1)
        obj = TrackedObject(1, detection, 1, 1, 0, (0.0, 0.0), ((10, 10),))

        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        draw_tracks(frame, [obj], labels="id")
        self.assertTrue(np.any(frame > 0))

        with self.assertRaises(ValueError):
            draw_tracks(
                np.zeros((80, 80, 3), dtype=np.uint8),
                [obj],
                labels="invalide",
            )


if __name__ == "__main__":
    unittest.main()
