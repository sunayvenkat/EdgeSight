
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from edgesight.detector import (
    ObjectDetector,
    draw_detections,
)


def make_box(confidence):
    return SimpleNamespace(
        xyxy=np.array([[10, 20, 100, 200]]),
        conf=np.array([confidence]),
        cls=np.array([0]),
    )


def make_detector(boxes):
    fake_model = Mock()

    fake_model.predict.return_value = [
        SimpleNamespace(
            boxes=boxes,
            names={0: "person"},
        )
    ]

    return ObjectDetector(
        model=fake_model,
        confidence=0.40,
    )


def test_detect_object():
    detector = make_detector(
        [make_box(0.90)]
    )

    frame = np.zeros(
        (480, 640, 3),
        dtype=np.uint8,
    )

    detections = detector.detect(frame)

    assert len(detections) == 1
    assert detections[0].label == "person"
    assert detections[0].bbox == (10, 20, 100, 200)
    assert abs(detections[0].confidence - 0.90) < 1e-6


def test_low_confidence_filtered():
    detector = make_detector(
        [make_box(0.20)]
    )

    frame = np.zeros(
        (480, 640, 3),
        dtype=np.uint8,
    )

    detections = detector.detect(frame)

    assert len(detections) == 0


def test_no_detections():
    detector = make_detector([])

    frame = np.zeros(
        (480, 640, 3),
        dtype=np.uint8,
    )

    detections = detector.detect(frame)

    assert detections == []

    annotated = draw_detections(
        frame,
        detections,
    )

    assert np.array_equal(annotated, frame)
