


import pytest

from scripts.select_model import (
    choose_candidate,
    sha256_file,
    valid_class_names,
)


def make_report(map_value, recall=0.5, images=21):
    return {
        "split": "val",
        "images": images,
        "mAP50_95": map_value,
        "recall": recall,
    }


def test_selects_better_model():
    v1 = make_report(0.30)
    v2 = make_report(0.50)

    assert choose_candidate(v1, v2) == "pilot_v2_data"


def test_retains_baseline_on_tie():
    v1 = make_report(0.40, recall=0.60)
    v2 = make_report(0.40, recall=0.60)

    assert choose_candidate(v1, v2) == "pilot_v1"


def test_rejects_different_evaluation_sizes():
    v1 = make_report(0.40, images=6)
    v2 = make_report(0.60, images=21)

    with pytest.raises(ValueError):
        choose_candidate(v1, v2)


def test_class_mapping():
    valid = {
        0: "rock",
        1: "cone",
        2: "debris",
    }

    invalid = {
        0: "cone",
        1: "rock",
        2: "debris",
    }

    assert valid_class_names(valid)
    assert not valid_class_names(invalid)


def test_checkpoint_hash_changes(tmp_path):
    checkpoint = tmp_path / "model.pt"

    checkpoint.write_bytes(b"original weights")
    original_hash = sha256_file(checkpoint)

    checkpoint.write_bytes(b"modified weights")
    modified_hash = sha256_file(checkpoint)

    assert original_hash != modified_hash
