


from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from edgesight.tracker import (
    ObjectTracker,
    OnnxTracker,
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


class FakeOnnxDetector:
    names = {0: "rock", 1: "cone", 2: "debris"}

    def __init__(self, frames):
        # Each item is a list of (x1, y1, x2, y2, score, class_id)
        self.frames = list(frames)

    def infer(self, frame):
        rows = np.array(self.frames.pop(0), dtype=np.float32).reshape(-1, 6)

        return rows[:, :4], rows[:, 4], rows[:, 5].astype(int)


def run_onnx_tracker(frames):
    tracker = OnnxTracker(FakeOnnxDetector(frames))

    frame = np.zeros(
        (480, 640, 3), dtype=np.uint8
    )

    return [tracker.track(frame) for _ in frames]


def test_onnx_tracker_keeps_id_across_frames():
    box = (100, 100, 200, 220, 0.90, 1)

    results = run_onnx_tracker([[box], [box], [box]])

    ids = [objects[0].track_id for objects in results]

    assert all(len(objects) == 1 for objects in results)
    assert ids[0] is not None
    assert ids == [ids[0]] * 3
    assert results[0][0].label == "cone"


def test_onnx_tracker_hides_unconfirmed_track():
    box = (100, 100, 200, 220, 0.90, 0)

    results = run_onnx_tracker([[], [box], [box]])

    assert results[0] == []
    assert results[1] == []
    assert len(results[2]) == 1
    assert results[2][0].track_id is not None


def test_onnx_tracker_empty_frames():
    assert run_onnx_tracker([[], []]) == [[], []]
