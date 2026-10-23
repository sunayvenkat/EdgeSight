


import numpy as np
import onnxruntime as ort
import pytest

from scripts.tune_onnx import (
    make_options,
    prepare_input,
    compare_outputs,
)

from scripts.summarize_tuning import choose_best


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
