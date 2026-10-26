

from pathlib import Path

import pytest

import scripts.build_dataset as build_dataset
from scripts.cross_validate import group_for, split_fold, summarize


GROUPS = [
    {"group": "posts", "clip": "more_cones", "ranges": [[0, 530]]},
    {"group": "crosswalk", "clip": "more_cones", "ranges": [[530, 600], [1280, 1430]]},
    {"group": "tail", "clip": "more_cones", "ranges": [[1960, None]]},
    {"group": "cones_2", "clip": "cones_2"},
]


def frame(clip, index):
    return {"file": f"{clip}_f{index:05d}.jpg", "clip": clip, "frame_index": index}


@pytest.mark.parametrize("clip, index, expected", [("more_cones", 0, "posts"), ("more_cones", 570, "crosswalk"), ("more_cones", 1300, "crosswalk"), ("more_cones", 2430, "tail"), ("cones_2", 705, "cones_2")])
def test_group_for_ranges(clip, index, expected):
    assert group_for(frame(clip, index), GROUPS) == expected


def test_group_for_requires_exactly_one_match():
    with pytest.raises(ValueError, match="0 groups"):
        group_for(frame("more_cones", 1000), GROUPS)

    with pytest.raises(ValueError, match="2 groups"):
        group_for(frame("cones_2", 0), GROUPS + [{"group": "dup", "clip": "cones_2"}])


def test_split_fold_excludes_reserved_groups():
    entries = [{"path": Path(f"{g}.jpg"), "group": g, "label": ""} for g in ("a", "b", "reserved", "c")]

    train, val = split_fold(entries, {"name": "f", "val_groups": ["b"]}, {"reserved"})

    assert [e["group"] for e in train] == ["a", "c"]
    assert [e["group"] for e in val] == ["b"]

    with pytest.raises(ValueError, match="unknown validation groups"):
        split_fold(entries, {"name": "f", "val_groups": ["zzz"]}, set())


def test_summarize_averages_completed_folds():
    def run(fold, value, fp):
        report = {"mAP50_95": value, "mAP50": value + 0.2, "recall": 0.5, "background_fp_rate": fp, "per_class": [{"class_name": "cone", "mAP50": value}]}
        return {"config": "long", "fold": fold, "best": report, "last": report, "best_epoch": 40, "epochs_run": 70, "train_seconds": 600}

    summary = summarize([run("fold1", 0.4, 0.1), run("fold2", 0.6, 0.3), {"config": "long", "fold": "fold3", "error": "boom"}])

    assert summary["long"]["folds"] == 2
    assert summary["long"]["mAP50_95"] == 0.5
    assert summary["long"]["background_fp_rate"] == 0.2
    assert summary["long"]["per_class_mAP50"]["cone"] == 0.5
    assert summary["long"]["per_class_mAP50"]["rock"] is None


def test_remove_label_lines():
    labels = {"a": "2 0.1 0.1 0.1 0.1\n1 0.5 0.5 0.1 0.1\n2 0.9 0.9 0.05 0.05\n"}

    assert build_dataset.remove_label_lines(labels, {"a": [1, 3]})["a"] == "1 0.5 0.5 0.1 0.1\n"

    with pytest.raises(ValueError, match="no line"):
        build_dataset.remove_label_lines(labels, {"a": [4]})

    with pytest.raises(ValueError, match="no labels"):
        build_dataset.remove_label_lines(labels, {"b": [1]})
