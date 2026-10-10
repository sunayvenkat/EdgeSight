


import yaml

from scripts.validate_dataset import (
    validate_label_file,
    validate_dataset,
)


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
