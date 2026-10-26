"""Phase 5: the ONNX Runtime detector and ByteTrack tracker used by the app."""

import json
from pathlib import Path

import numpy as np
import pytest

from edgesight.detector import OnnxDetector, create_session, decode_output, letterbox, preprocess, scale_boxes
from edgesight.tracker import OnnxTracker

from tests.helpers import FakeSession, make_output


# OnnxDetector (ONNX Runtime)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


ONNX_PATH = PROJECT_ROOT / "models" / "selected.onnx"


VAL_IMAGES = PROJECT_ROOT / "datasets" / "rover_obstacles" / "images" / "val"


def test_letterbox_pads_camera_frame():
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)

    padded, gain, padding = letterbox(frame, 640)

    assert padded.shape == (640, 640, 3)
    assert gain == 1.0
    assert padding == (0, 80)
    assert np.all(padded[:80] == 114)
    assert np.all(padded[80:560] == 255)
    assert np.all(padded[560:] == 114)


def test_letterbox_downscales_large_frame():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    padded, gain, padding = letterbox(frame, 640)

    assert padded.shape == (640, 640, 3)
    assert gain == 0.5
    assert padding == (0, 140)


@pytest.mark.parametrize("shape", [(480, 640, 3), (720, 1280, 3), (514, 451, 3)])
def test_preprocess_matches_reference(shape):
    from scripts.tune_onnx import prepare_input

    frame = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)

    tensor, _, _ = preprocess(frame, 640)

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert np.allclose(tensor, prepare_input(frame), atol=1e-6)


def test_decode_filters_low_confidence():
    output = make_output([(100, 100, 20, 20, 0, 0.90), (300, 300, 20, 20, 1, 0.30)])

    boxes, scores, class_ids = decode_output(output, 0.40, 0.7)

    assert len(boxes) == 1
    assert np.allclose(boxes[0], (90, 90, 110, 110))
    assert scores[0] == pytest.approx(0.90)
    assert class_ids[0] == 0


def test_decode_suppresses_overlap_within_class():
    output = make_output([(100, 100, 40, 40, 0, 0.90), (102, 100, 40, 40, 0, 0.80)])

    boxes, scores, _ = decode_output(output, 0.40, 0.7)

    assert len(boxes) == 1
    assert scores[0] == pytest.approx(0.90)


def test_decode_keeps_overlap_across_classes():
    output = make_output([(100, 100, 40, 40, 0, 0.90), (102, 100, 40, 40, 2, 0.80)])

    _, _, class_ids = decode_output(output, 0.40, 0.7)

    assert sorted(class_ids.tolist()) == [0, 2]


def test_decode_empty_output():
    boxes, scores, class_ids = decode_output(make_output([]), 0.40, 0.7)

    assert boxes.shape == (0, 4)
    assert len(scores) == 0
    assert len(class_ids) == 0


def test_scale_boxes_removes_padding_and_clips():
    boxes = np.array([[10, 90, 110, 190], [600, 500, 700, 600]], dtype=np.float32)

    scaled = scale_boxes(boxes, 0.5, (0, 80), (720, 1280, 3))

    assert np.allclose(scaled[0], (20, 20, 220, 220))
    assert np.allclose(scaled[1], (1200, 720, 1280, 720))


def test_detector_returns_frame_coordinates():
    # Camera frames are padded by 80 px at the top
    session = FakeSession(make_output([(320, 320, 100, 50, 1, 0.85)]))
    detector = OnnxDetector(session=session, confidence=0.40)

    detections = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0].label == "cone"
    assert detections[0].bbox == (270, 215, 370, 265)
    assert detections[0].confidence == pytest.approx(0.85)
    assert session.feeds[0]["images"].shape == (1, 3, 640, 640)


def test_detector_rejects_dynamic_input():
    with pytest.raises(ValueError):
        OnnxDetector(session=FakeSession(make_output([]), shape=(1, 3, "height", "width")))


def test_detector_rejects_invalid_confidence():
    with pytest.raises(ValueError):
        OnnxDetector(session=FakeSession(make_output([])), confidence=1.5)


def test_create_session_rejects_mismatched_runtime_config(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"not the tuned model")

    config = tmp_path / "runtime_config.json"
    config.write_text(json.dumps({"onnx_sha256": "0" * 64, "configuration": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="different ONNX file"):
        create_session(model, config)


@pytest.mark.skipif(not ONNX_PATH.is_file() or not VAL_IMAGES.is_dir(), reason="Requires the exported model and validation images")
def test_matches_ultralytics_onnx_predictions():
    import cv2
    from ultralytics import YOLO

    reference = YOLO(str(ONNX_PATH), task="detect")
    detector = OnnxDetector()

    images = sorted(VAL_IMAGES.glob("*.jpg"))
    assert images

    for path in images:
        frame = cv2.imread(str(path))

        result = reference.predict(frame, imgsz=640, conf=0.40, device="cpu", verbose=False)[0]
        boxes, scores, class_ids = detector.infer(frame)

        assert len(boxes) == len(result.boxes), path.name

        if len(boxes) == 0:
            continue

        # Pair each reference box with the closest candidate box
        expected = result.boxes.xyxy.numpy()
        distance = np.abs(expected[:, None, :] - boxes[None, :, :]).max(axis=2)
        nearest = distance.argmin(axis=1)

        assert distance.min(axis=1).max() < 1.5, path.name
        assert np.array_equal(class_ids[nearest], result.boxes.cls.numpy().astype(int)), path.name
        assert np.allclose(scores[nearest], result.boxes.conf.numpy(), atol=1e-4), path.name


# OnnxTracker (ByteTrack on ONNX detections)

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
