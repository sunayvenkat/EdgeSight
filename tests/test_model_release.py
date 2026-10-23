


import pytest

from scripts.select_model import (
    archive_release,
    choose_best,
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


def test_choose_best_among_several():
    reports = {"a": make_report(0.30), "b": make_report(0.45), "c": make_report(0.40, recall=0.9)}

    assert choose_best(reports) == "b"


def test_choose_best_tie_keeps_earlier_candidate():
    reports = {"baseline": make_report(0.40, recall=0.6), "challenger": make_report(0.40, recall=0.6)}

    assert choose_best(reports) == "baseline"


def test_choose_best_rejects_mixed_datasets_or_sizes():
    first = {**make_report(0.4), "dataset": "rover_obstacles_v2", "imgsz": 640}

    with pytest.raises(ValueError):
        choose_best({"a": first, "b": {**first, "dataset": "rover_obstacles"}})

    with pytest.raises(ValueError):
        choose_best({"a": first, "b": {**first, "imgsz": 512}})


def test_archive_release_moves_files(tmp_path):
    (tmp_path / "selection.json").write_text('{"release": "edgesight-ml-v1"}', encoding="utf-8")
    (tmp_path / "selected.pt").write_bytes(b"weights")
    (tmp_path / "quantization_manifest.json").write_text("{}", encoding="utf-8")

    archive_dir, moved = archive_release(tmp_path)

    assert archive_dir == tmp_path / "archive" / "edgesight-ml-v1"
    assert sorted(moved) == ["selected.pt", "selection.json"]
    assert not (tmp_path / "selected.pt").exists()
    assert (tmp_path / "quantization_manifest.json").exists()

    with pytest.raises(FileNotFoundError):
        archive_release(tmp_path)


def test_archived_v1_can_still_win(tmp_path, monkeypatch):
    import scripts.select_model as select_model

    models = tmp_path / "models"
    models.mkdir()
    (models / "selection.json").write_text('{"release": "edgesight-ml-v1"}', encoding="utf-8")
    (models / "selected.pt").write_bytes(b"v1 weights")
    lock = tmp_path / "lock.json"
    lock.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(select_model, "MODELS_ROOT", models)
    monkeypatch.setattr(select_model, "archive_release", lambda: archive_release(models))
    monkeypatch.setattr(select_model, "verify_splits", lambda dataset, lock_path: None)
    monkeypatch.setattr(select_model, "YOLO", lambda path: type("Model", (), {"task": "detect", "names": {0: "rock", 1: "cone", 2: "debris"}})())

    candidates = {"v1": (models / "selected.pt").resolve(), "v3": tmp_path / "v3.pt"}
    full = {"precision": 0.6, "mAP50": 0.7}
    reports = {"v1": {**make_report(0.50), **full}, "v3": {**make_report(0.40), **full}}

    select_model.release_model("edgesight-ml-v2", candidates, reports, tmp_path, lock, archive=True)

    assert (models / "selected.pt").read_bytes() == b"v1 weights"
    assert (models / "archive" / "edgesight-ml-v1" / "selected.pt").is_file()
