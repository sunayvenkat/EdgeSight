"""Phase 4: CPU benchmarks, ONNX export, ONNX Runtime tuning, and native C++ inference."""

import cv2
import numpy as np
import onnxruntime as ort
import pytest

from scripts.benchmark_inference import load_images, percentile, summarize_latencies
from scripts.compare_predictions import bbox_iou, match_detections
from scripts.prepare_cpp_inputs import validate_preset
from scripts.summarize_tuning import choose_best
from scripts.tune_onnx import make_options, prepare_input, compare_outputs
from scripts.verify_cpp_outputs import compare_arrays


# PyTorch CPU benchmark

def test_percentile_calculation():
    values = [10, 20, 30, 40, 50]

    assert percentile(values, 50) == 30
    assert percentile(values, 95) == pytest.approx(48)


def test_latency_summary():
    values = [10.0, 20.0, 30.0]

    result = summarize_latencies(values)

    assert result["count"] == 3
    assert result["mean_ms"] == 20
    assert result["median_ms"] == 20
    assert result["min_ms"] == 10
    assert result["max_ms"] == 30


def test_empty_latency_rejected():
    with pytest.raises(ValueError):
        summarize_latencies([])


def test_image_loading(tmp_path):
    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    image_path = tmp_path / "test.jpg"

    assert cv2.imwrite(str(image_path), frame)

    images = load_images(tmp_path)

    assert len(images) == 1
    assert images[0][0].name == "test.jpg"
    assert images[0][1].shape == (100, 100, 3)


def test_invalid_image_rejected(tmp_path):
    image_path = tmp_path / "corrupt.jpg"

    image_path.write_bytes(b"invalid image")

    with pytest.raises(ValueError):
        load_images(tmp_path)


# ONNX export comparison

def make_detection(class_id, bbox):
    return {
        "class_id": class_id,
        "confidence": 0.90,
        "bbox": bbox,
    }


def test_identical_boxes():
    box = (10, 10, 100, 100)

    assert bbox_iou(box, box) == pytest.approx(1.0)


def test_disjoint_boxes():
    first = (0, 0, 10, 10)
    second = (20, 20, 30, 30)

    assert bbox_iou(first, second) == 0.0


def test_partial_overlap():
    first = (0, 0, 10, 10)
    second = (5, 5, 15, 15)

    assert bbox_iou(first, second) == pytest.approx(1 / 7)


def test_matching_predictions():
    reference = [make_detection(0, (10, 10, 100, 100))]

    candidate = [make_detection(0, (10, 10, 100, 100))]

    result = match_detections(reference, candidate)

    assert result["matched"] == 1
    assert result["pytorch_unmatched"] == 0
    assert result["onnx_unmatched"] == 0


def test_different_classes_not_matched():
    reference = [make_detection(0, (10, 10, 100, 100))]

    candidate = [make_detection(1, (10, 10, 100, 100))]

    result = match_detections(reference, candidate)

    assert result["matched"] == 0
    assert result["pytorch_unmatched"] == 1
    assert result["onnx_unmatched"] == 1


# ONNX Runtime tuning

def test_thread_configuration():
    options = make_options("intra4")

    assert options.intra_op_num_threads == 4
    assert options.inter_op_num_threads == 1
    assert options.graph_optimization_level == (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )


def test_spinning_disabled():
    options = make_options("intra4_nospin")

    value = options.get_session_config_entry(
        "session.intra_op.allow_spinning"
    )

    assert value == "0"


def test_preprocessing():
    # OpenCV image in BGR channel order.
    frame = np.zeros(
        (64, 64, 3),
        dtype=np.uint8,
    )

    frame[:] = (10, 20, 30)

    tensor = prepare_input(frame)

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32

    # Check BGR-to-RGB conversion.
    assert tensor[0, 0, 320, 320] == pytest.approx(
        30 / 255
    )

    assert tensor.min() >= 0
    assert tensor.max() <= 1


def test_output_consistency():
    reference = [
        np.array(
            [[1.0, 2.0]],
            dtype=np.float32,
        )
    ]

    candidate = [
        np.array(
            [[1.0001, 2.0001]],
            dtype=np.float32,
        )
    ]

    result = compare_outputs(
        reference, candidate
    )

    assert result["allclose"] is True
    assert result["max_abs_error"] < 0.001


def test_select_fastest_consistent_preset():
    results = {
        "default": {
            "median_ms": 90,
            "p95_ms": 100,
            "consistent": True,
        },
        "intra4": {
            "median_ms": 60,
            "p95_ms": 70,
            "consistent": True,
        },
        "intra2": {
            "median_ms": 40,
            "p95_ms": 50,
            "consistent": False,
        },
    }

    assert choose_best(results) == "intra4"


def test_tuning_file_names_keep_v1_names():
    from scripts.tune_onnx import tuning_file_name

    assert tuning_file_name("intra4", 2) == "tuning_intra4_run2.json"
    assert tuning_file_name("intra4", 2, "v2") == "tuning_v2_intra4_run2.json"


# Native C++ inference

def make_config(threads=4):
    return {
        "recommended_preset": "intra4",
        "configuration": {
            "intra_op_threads": threads,
            "graph_optimization": "all",
            "spinning": True,
        },
    }


def test_valid_runtime_configuration():
    assert validate_preset(make_config()) == "intra4"


def test_mismatched_runtime_configuration():
    with pytest.raises(ValueError):
        validate_preset(make_config(threads=2))


def test_matching_raw_outputs():
    reference = np.array(
        [[1.0, 2.0, 3.0]],
        dtype=np.float32,
    )

    candidate = np.array(
        [[1.0001, 2.0001, 3.0001]],
        dtype=np.float32,
    )

    consistent, error = compare_arrays(
        reference,
        candidate,
    )

    assert consistent is True
    assert error < 0.001


def test_invalid_raw_outputs():
    reference = np.array(
        [[1.0, 2.0]],
        dtype=np.float32,
    )

    candidate = np.array(
        [[5.0, 6.0]],
        dtype=np.float32,
    )

    consistent, _ = compare_arrays(
        reference,
        candidate,
    )

    assert consistent is False
