import unittest

import numpy as np

from src.detector import Detection, Detector, count_by_category


class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class FakeBoxes:
    def __init__(self, boxes, ids=None):
        array = np.asarray(boxes, dtype=float)
        if array.size == 0:
            array = np.empty((0, 6))
        self.xyxy = FakeTensor(array[:, :4])
        self.conf = FakeTensor(array[:, 4])
        self.cls = FakeTensor(array[:, 5])
        self.id = FakeTensor(ids) if ids is not None else None

    def __len__(self):
        return len(self.xyxy.array)


class FakeResult:
    def __init__(self, boxes, names, ids=None):
        self.boxes = FakeBoxes(boxes, ids)
        self.names = names


class FakeModel:
    def __init__(self, results, names=None):
        self.results = results
        self.names = names or {}
        self.frames = []
        self.kwargs = None
        self.track_kwargs = None

    def __call__(self, frame, **kwargs):
        self.frames.append(frame)
        self.kwargs = kwargs
        return [self.results]

    def track(self, frame, **kwargs):
        self.frames.append(frame)
        self.track_kwargs = kwargs
        return [self.results]


def make_detector(boxes, names, ids=None, **kwargs):
    model = FakeModel(FakeResult(boxes, names, ids), names)
    return Detector(model=model, **kwargs), model


class DetectorTests(unittest.TestCase):
    def test_keeps_road_classes_and_maps_categories(self):
        boxes = [
            [10, 20, 110, 220, 0.9, 0],
            [50, 60, 150, 160, 0.8, 2],
            [0, 0, 10, 10, 0.7, 16],
        ]
        names = {0: "person", 2: "car", 16: "dog"}
        detector, _ = make_detector(boxes, names)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        detections = detector.detect(frame)

        self.assertEqual([d.category for d in detections], ["pieton", "vehicule"])
        self.assertEqual(detections[0].class_name, "person")

    def test_confidence_threshold_filters_results(self):
        boxes = [[0, 0, 10, 10, 0.2, 0], [0, 0, 10, 10, 0.9, 0]]
        names = {0: "person"}
        detector, _ = make_detector(
            boxes, names, confidence_threshold=0.5
        )

        detections = detector.detect(np.zeros((50, 50, 3), dtype=np.uint8))

        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].confidence, 0.9)

    def test_all_classes_keeps_unknown_as_autre(self):
        detector, _ = make_detector(
            [[0, 0, 5, 5, 0.9, 16]], {16: "dog"}, road_only=False
        )

        detections = detector.detect(np.zeros((20, 20, 3), dtype=np.uint8))

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].category, "autre")

    def test_coordinates_are_clamped_and_custom_class_mapping(self):
        boxes = [[-5, -5, 500, 500, 0.9, 99]]
        names = {99: "pothole"}
        detector, _ = make_detector(boxes, names)
        frame = np.zeros((100, 200, 3), dtype=np.uint8)

        detection = detector.detect(frame)[0]

        self.assertEqual(
            (detection.x1, detection.y1, detection.x2, detection.y2),
            (0, 0, 199, 99),
        )
        self.assertEqual(detection.category, "nid_de_poule")

    def test_max_detections_limits_output(self):
        boxes = [[0, 0, 10, 10, 0.9, 0] for _ in range(5)]
        detector, _ = make_detector(boxes, {0: "person"}, max_detections=2)

        detections = detector.detect(np.zeros((50, 50, 3), dtype=np.uint8))

        self.assertEqual(len(detections), 2)

    def test_empty_results_returns_empty_list(self):
        detector, _ = make_detector(np.empty((0, 6)), {0: "person"})

        self.assertEqual(
            detector.detect(np.zeros((10, 10, 3), dtype=np.uint8)), []
        )

    def test_detect_passes_frame_and_options_to_model(self):
        frame = np.zeros((30, 40, 3), dtype=np.uint8)
        detector, model = make_detector(
            np.empty((0, 6)),
            {},
            confidence_threshold=0.4,
            iou_threshold=0.5,
            max_detections=10,
            imgsz=320,
        )

        detector.detect(frame)

        self.assertIs(model.frames[0], frame)
        self.assertEqual(model.kwargs["conf"], 0.4)
        self.assertEqual(model.kwargs["iou"], 0.5)
        self.assertEqual(model.kwargs["max_det"], 10)
        self.assertEqual(model.kwargs["imgsz"], 320)
        self.assertFalse(model.kwargs["verbose"])

    def test_detect_filters_classes_at_model_level(self):
        detector, model = make_detector(
            np.empty((0, 6)), {0: "person", 2: "car", 16: "dog"}
        )

        detector.detect(np.zeros((20, 20, 3), dtype=np.uint8))

        self.assertEqual(model.kwargs["classes"], [0, 2])

    def test_all_classes_removes_model_class_filter(self):
        detector, model = make_detector(
            np.empty((0, 6)), {0: "person"}, road_only=False
        )

        detector.detect(np.zeros((20, 20, 3), dtype=np.uint8))

        self.assertNotIn("classes", model.kwargs)

    def test_track_filters_classes_at_model_level(self):
        detector, model = make_detector(
            np.empty((0, 6)), {0: "person", 2: "car", 16: "dog"}
        )

        detector.track(np.zeros((20, 20, 3), dtype=np.uint8))

        self.assertEqual(model.track_kwargs["classes"], [0, 2])

    def test_detect_rejects_empty_frame(self):
        detector, _ = make_detector(np.empty((0, 6)), {})

        with self.assertRaises(ValueError):
            detector.detect(np.array([]))

    def test_detect_sets_track_id_to_none(self):
        detector, _ = make_detector(
            [[0, 0, 10, 10, 0.9, 0]], {0: "person"}
        )

        detection = detector.detect(
            np.zeros((50, 50, 3), dtype=np.uint8)
        )[0]

        self.assertIsNone(detection.track_id)

    def test_track_propagates_ids(self):
        detector, _ = make_detector(
            [[0, 0, 10, 10, 0.9, 0], [20, 20, 30, 30, 0.8, 2]],
            {0: "person", 2: "car"},
            ids=[7, 8],
        )

        detections = detector.track(
            np.zeros((50, 50, 3), dtype=np.uint8)
        )

        self.assertEqual([d.track_id for d in detections], [7, 8])

    def test_track_keeps_low_confidence_detections(self):
        detector, _ = make_detector(
            [[0, 0, 10, 10, 0.15, 0]],
            {0: "person"},
            confidence_threshold=0.35,
        )

        detections = detector.track(
            np.zeros((50, 50, 3), dtype=np.uint8)
        )

        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].confidence, 0.15)

    def test_track_passes_ultralytics_options(self):
        frame = np.zeros((30, 40, 3), dtype=np.uint8)
        detector, model = make_detector(
            np.empty((0, 6)), {}, ids=np.empty((0,))
        )

        detector.track(frame, tracker="botsort.yaml")

        self.assertIs(model.frames[0], frame)
        self.assertTrue(model.track_kwargs["persist"])
        self.assertEqual(model.track_kwargs["tracker"], "botsort.yaml")
        self.assertEqual(model.track_kwargs["conf"], 0.1)
        self.assertFalse(model.track_kwargs["verbose"])

    def test_track_rejects_empty_frame(self):
        detector, _ = make_detector(np.empty((0, 6)), {})

        with self.assertRaises(ValueError):
            detector.track(np.array([]))

    def test_invalid_configuration_raises(self):
        with self.assertRaises(ValueError):
            Detector(confidence_threshold=1.5, model=FakeModel(None))
        with self.assertRaises(ValueError):
            Detector(iou_threshold=-0.1, model=FakeModel(None))
        with self.assertRaises(ValueError):
            Detector(max_detections=0, model=FakeModel(None))
        with self.assertRaises(ValueError):
            Detector(imgsz=0, model=FakeModel(None))

    def test_detection_geometry(self):
        detection = Detection(10, 20, 30, 60, 0.8, 0, "person", "pieton")

        self.assertEqual((detection.width, detection.height), (20, 40))
        self.assertEqual(detection.area, 800)
        self.assertEqual(detection.center, (20, 40))
        self.assertEqual(detection.bottom_center, (20, 60))

    def test_count_by_category(self):
        detections = [
            Detection(0, 0, 10, 10, 0.9, 0, "person", "pieton"),
            Detection(0, 0, 10, 10, 0.8, 0, "person", "pieton"),
            Detection(0, 0, 10, 10, 0.7, 2, "car", "vehicule"),
        ]

        self.assertEqual(
            count_by_category(detections), {"pieton": 2, "vehicule": 1}
        )


if __name__ == "__main__":
    unittest.main()
