


import pytest

from scripts.audit_splits import (build_manifest, find_cross_split_duplicates, locked_splits_match)

from scripts.compare_experiments import (calculate_changes, per_class_changes)


def make_dataset(tmp_path, duplicate=False):
    root = tmp_path / "dataset"

    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split

        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)

        data = (
            b"same-image"
            if duplicate
            else split.encode("utf-8")
        )

        (image_dir / "sample.jpg").write_bytes(data)

        (label_dir / "sample.txt").write_text("0 0.5 0.5 0.2 0.2", encoding="utf-8")

    return root


def test_locked_splits_match(tmp_path):
    root = make_dataset(tmp_path)

    original = build_manifest(root)

    assert locked_splits_match(original, original)

    label = root / "labels" / "val" / "sample.txt"
    label.write_text("1 0.5 0.5 0.3 0.3", encoding="utf-8")

    modified = build_manifest(root)

    assert not locked_splits_match(modified, original)


def test_duplicate_images_detected(tmp_path):
    root = make_dataset(tmp_path, duplicate=True)

    manifest = build_manifest(root)

    duplicates = find_cross_split_duplicates(manifest)

    assert len(duplicates) > 0


def make_report(map50, images=21):
    return {
        "split": "val",
        "images": images,
        "precision": 0.5,
        "recall": 0.4,
        "mAP50": map50,
        "mAP50_95": 0.3,
        "per_class": [
            {
                "class_id": 0,
                "class_name": "rock",
                "mAP50": map50,
            }
        ],
    }


def test_experiment_improvement():
    old = make_report(0.40)
    new = make_report(0.60)

    changes = calculate_changes(old, new)

    assert changes["mAP50"] == pytest.approx(0.20)

    classes = per_class_changes(old, new)

    assert classes[0]["class"] == "rock"
    assert classes[0]["delta"] == pytest.approx(0.20)


def test_mismatched_evaluation_rejected():
    old = make_report(0.40, images=6)
    new = make_report(0.60, images=21)

    with pytest.raises(ValueError):
        calculate_changes(old, new)
