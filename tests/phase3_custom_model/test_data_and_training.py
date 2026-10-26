"""Phase 3: dataset validation and the training pipeline."""

from pathlib import Path

import pytest
import yaml

from scripts.train_model import create_training_config
from scripts.validate_dataset import validate_label_file, validate_dataset


# Dataset validation

def create_test_dataset(tmp_path):
    """Create a temporary valid dataset."""

    root = tmp_path / "rover_obstacles"

    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split

        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)

        (image_dir / "sample.jpg").write_bytes(b"test")

        (label_dir / "sample.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n",
            encoding="utf-8",
        )

    config = {
        "nc": 3,
        "names": {
            0: "rock",
            1: "cone",
            2: "debris",
        },
    }

    (root / "data.yaml").write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )

    return root


def test_valid_dataset(tmp_path):
    root = create_test_dataset(tmp_path)

    errors, counts = validate_dataset(
        root, require_images=True
    )

    assert errors == []
    assert counts == {
        "train": 1,
        "val": 1,
        "test": 1,
    }


def test_invalid_class_id(tmp_path):
    label = tmp_path / "invalid.txt"

    label.write_text(
        "5 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )

    errors = validate_label_file(label, 3)

    assert any(
        "Invalid class ID" in error
        for error in errors
    )


def test_invalid_bounding_box(tmp_path):
    label = tmp_path / "invalid.txt"

    label.write_text(
        "0 0.95 0.5 0.3 0.2\n",
        encoding="utf-8",
    )

    errors = validate_label_file(label, 3)

    assert any(
        "Bounding box outside image" in error
        for error in errors
    )


def test_missing_label(tmp_path):
    root = create_test_dataset(tmp_path)

    label = root / "labels" / "train" / "sample.txt"
    label.unlink()

    errors, _ = validate_dataset(root)

    assert any(
        "Missing label" in error
        for error in errors
    )


def test_empty_dataset_strict_mode(tmp_path):
    root = create_test_dataset(tmp_path)

    for split in ("train", "val", "test"):
        (root / "images" / split / "sample.jpg").unlink()
        (root / "labels" / split / "sample.txt").unlink()

    errors, _ = validate_dataset(
        root, require_images=True
    )

    assert sum(
        "No images" in error
        for error in errors
    ) == 3


# Training configuration

def test_training_config_uses_absolute_path(tmp_path):
    dataset_root = tmp_path / "rover_obstacles"
    dataset_root.mkdir()

    config = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 3,
        "names": {
            0: "rock",
            1: "cone",
            2: "debris",
        },
    }

    source = dataset_root / "data.yaml"

    source.write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )

    output = tmp_path / "runs" / "training.yaml"

    create_training_config(dataset_root, output)

    result = yaml.safe_load(
        output.read_text(encoding="utf-8")
    )

    assert Path(result["path"]).is_absolute()
    assert Path(result["path"]) == dataset_root.resolve()
    assert result["nc"] == 3
    assert result["names"][0] == "rock"
    assert result["names"][1] == "cone"
    assert result["names"][2] == "debris"


def test_training_config_missing_yaml(tmp_path):
    dataset_root = tmp_path / "missing_dataset"
    dataset_root.mkdir()

    output = tmp_path / "runs" / "training.yaml"

    with pytest.raises(FileNotFoundError):
        create_training_config(dataset_root, output)


def test_training_config_path_keeps_v1_name():
    from scripts.train_model import DATASET_ROOT, RUNS_ROOT, training_config_path

    assert training_config_path(DATASET_ROOT) == RUNS_ROOT / "configs" / "rover_training.yaml"
    assert training_config_path(DATASET_ROOT.parent / "rover_obstacles_v2") == RUNS_ROOT / "configs" / "rover_obstacles_v2_training.yaml"
