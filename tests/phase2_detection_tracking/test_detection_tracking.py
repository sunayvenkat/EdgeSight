"""Phase 2: PyTorch detection, Ultralytics tracking, and monitoring-zone events."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from edgesight.detector import ObjectDetector
from edgesight.events import ZoneMonitor, ZoneEvent
from edgesight.tracker import TrackedObject, ObjectTracker, draw_tracks

from tests.helpers import make_box


# ObjectDetector (PyTorch)

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


# ObjectTracker (Ultralytics tracking)

def make_tracker(track_id=7, empty=False):
    box = SimpleNamespace(
        cls=np.array([0]),
        conf=np.array([0.95]),
        xyxy=np.array([[10, 20, 100, 200]]),
    )

    class FakeBoxes(list):
        pass

    boxes = FakeBoxes(
        [] if empty else [box]
    )

    boxes.id = (
        None
        if empty or track_id is None
        else np.array([track_id])
    )

    detector = Mock()
    detector.predict_confidence = 0.40
    detector.passes.side_effect = lambda label, score: score >= 0.40

    detector.model.track.return_value = [
        SimpleNamespace(
            boxes=boxes,
            names={0: "person"},
        )
    ]

    return ObjectTracker(detector), detector


def test_tracking_id():
    tracker, detector = make_tracker()

    frame = np.zeros(
        (480, 640, 3), dtype=np.uint8
    )

    objects = tracker.track(frame)

    assert len(objects) == 1
    assert objects[0].track_id == 7
    assert objects[0].label == "person"
    assert objects[0].bbox == (10, 20, 100, 200)

    assert (
        detector.model.track.call_args.kwargs[
            "persist"
        ]
        is True
    )


def test_missing_tracking_id():
    tracker, _ = make_tracker(track_id=None)

    frame = np.zeros(
        (480, 640, 3), dtype=np.uint8
    )

    objects = tracker.track(frame)

    assert len(objects) == 1
    assert objects[0].track_id is None


def test_empty_tracking_results():
    tracker, _ = make_tracker(empty=True)

    frame = np.zeros(
        (480, 640, 3), dtype=np.uint8
    )

    assert tracker.track(frame) == []


def test_draw_tracks_does_not_modify_original():
    tracker, _ = make_tracker()

    frame = np.zeros(
        (480, 640, 3), dtype=np.uint8
    )

    objects = tracker.track(frame)
    annotated = draw_tracks(frame, objects)

    assert np.count_nonzero(frame) == 0
    assert np.count_nonzero(annotated) > 0


# Monitoring-zone events

FRAME_SHAPE = (480, 640, 3)


def make_object(track_id, bbox):
    return TrackedObject(
        track_id=track_id,
        label="person",
        confidence=0.90,
        bbox=bbox,
    )


def make_inside_object(zone, track_id=7):
    """Create an object inside the current zone."""

    left, top, right, bottom = zone.bounds(
        FRAME_SHAPE
    )

    center_x = (left + right) // 2
    center_y = (top + bottom) // 2

    return make_object(
        track_id,
        (
            center_x - 20,
            center_y - 60,
            center_x + 20,
            center_y,
        ),
    )


def test_enter_event_only_once():
    zone = ZoneMonitor()

    person = make_inside_object(zone)

    status, events = zone.update(
        [person], FRAME_SHAPE
    )

    assert status == "CAUTION"
    assert events == [ZoneEvent(7, "ENTER")]

    # Remaining inside should not generate
    # another ENTER event.
    status, events = zone.update(
        [person], FRAME_SHAPE
    )

    assert status == "CAUTION"
    assert events == []


def test_exit_event():
    zone = ZoneMonitor()

    person = make_inside_object(zone)

    # First, enter the zone.
    zone.update([person], FRAME_SHAPE)

    # Then, remove the object.
    status, events = zone.update(
        [], FRAME_SHAPE
    )

    assert status == "CLEAR"
    assert events == [ZoneEvent(7, "EXIT")]
