


from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from edgesight.tracker import (
    ObjectTracker,
    draw_tracks,
)


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
    detector.confidence = 0.40

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
