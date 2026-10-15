

import numpy as np
import pytest

from scripts.prepare_cpp_inputs import validate_preset
from scripts.verify_cpp_outputs import compare_arrays


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