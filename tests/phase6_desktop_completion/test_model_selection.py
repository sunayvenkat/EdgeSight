"""Phase 6: choosing and scoring candidate models (6.3, 6.7, 6.8): video analysis, --model, cross-validation, and per-class cutoffs."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import main
import scripts.build_dataset as build_dataset
from edgesight.detector import load_class_confidence, ObjectDetector, OnnxDetector, create_session
from scripts.analyze_video import FRAGMENT_FRAMES, summarize_classes, summarize_expected, summarize_tracks, summarize_zone
from scripts.build_dataset import held_out_frames
from scripts.cross_validate import group_for, split_fold, summarize
from scripts.evaluate_model import operating_point
from scripts.tune_confidence import best_cutoffs, write_release

from tests.helpers import FakeSession, make_box, make_output


# Video analysis on test clips (6.3)

def make_record(frame, objects=(), status="CLEAR", events=()):
    return {
        "frame": frame,
        "processing_ms": 20.0,
        "zone_status": status,
        "events": [list(event) for event in events],
        "objects": [{"track_id": track_id, "label": label, "confidence": confidence, "bbox": [0, 0, 10, 10]} for track_id, label, confidence in objects],
    }


def test_class_presence_and_flicker():
    records = [
        make_record(0, [(1, "rock", 0.8)]),
        make_record(1),
        make_record(2, [(1, "rock", 0.6), (2, "cone", 0.9)]),
        make_record(3, [(1, "rock", 0.7)]),
    ]

    summary = summarize_classes(records)

    assert summary["rock"]["frames_present"] == 3
    assert summary["rock"]["frames_present_percent"] == 75.0
    assert summary["rock"]["mean_confidence"] == 0.7
    assert summary["rock"]["presence_toggles"] == 2
    assert summary["cone"]["detections"] == 1
    assert summary["debris"]["mean_confidence"] is None


def test_track_lifetimes_gaps_and_label_changes():
    records = [make_record(frame, [(1, "rock", 0.8)]) for frame in range(10)]
    records[4] = make_record(4)
    records[7] = make_record(7, [(1, "debris", 0.5), (2, "cone", 0.9)])

    summary = summarize_tracks(records)
    track = summary["per_track"][1]

    assert summary["track_count"] == 2
    assert track["observed_frames"] == 9
    assert track["gap_frames"] == 1
    assert track["majority_label"] == "rock"
    assert track["label_changes"] == 2
    assert summary["tracks_with_label_changes"] == 1
    assert summary["fragments"] == 1
    assert summary["per_track"][2]["observed_frames"] < FRAGMENT_FRAMES


def test_untracked_detections_counted_separately():
    summary = summarize_tracks([make_record(0, [(None, "rock", 0.8)])])

    assert summary["track_count"] == 0
    assert summary["untracked_detections"] == 1
    assert summary["median_observed_frames"] is None


def test_expected_class_hits_and_other_classes():
    records = [
        make_record(0, [(1, "cone", 0.8)]),
        make_record(1, [(1, "cone", 0.8), (2, "debris", 0.5)]),
        make_record(2),
        make_record(3, [(3, "debris", 0.5)]),
    ]

    summary = summarize_expected(records, "cone")

    assert summary["expected_frames"] == 2
    assert summary["expected_frames_percent"] == 50.0
    assert summary["other_class_detections"] == 2
    assert summary["other_class_frames"] == 2
    assert summary["other_class_counts"] == {"debris": 2}


def test_empty_scene_counts_false_positives():
    records = [make_record(0), make_record(1, [(1, "cone", 0.6), (2, "debris", 0.4)]), make_record(2, [(1, "cone", 0.6)])]

    summary = summarize_expected(records, "none")

    assert summary["false_positive_frames"] == 2
    assert summary["false_positive_detections"] == 3
    assert summary["false_positive_frames_percent"] == 66.7


def test_zone_events_and_caution_frames():
    records = [make_record(0, status="CAUTION", events=[(1, "ENTER")]), make_record(1, status="CAUTION"), make_record(2, events=[(1, "EXIT")])]

    assert summarize_zone(records) == {"caution_frames": 2, "enter_events": 1, "exit_events": 1}


# Candidate models with --model (6.7)

@pytest.mark.parametrize("argv", [["--model", "candidate.pt"], ["--backend", "pytorch", "--model", "candidate.onnx"]])
def test_model_must_match_backend(tmp_path, argv):
    argv = [str(tmp_path / value) if value.startswith("candidate") else value for value in argv]

    for value in argv:
        if value.startswith(str(tmp_path)):
            open(value, "wb").close()

    with pytest.raises(SystemExit):
        main.parse_args(argv)


def test_model_must_exist(tmp_path):
    with pytest.raises(SystemExit):
        main.parse_args(["--model", str(tmp_path / "missing.onnx")])


def test_candidate_model_skips_release_hash_check(tmp_path, monkeypatch):
    calls = {}

    def fake_create_session(**kwargs):
        calls.update(kwargs)
        return "session"

    monkeypatch.setattr("edgesight.detector.create_session", fake_create_session)
    monkeypatch.setattr("edgesight.detector.OnnxDetector", lambda session, class_confidence: (session, class_confidence))
    monkeypatch.setattr("edgesight.tracker.OnnxTracker", lambda detector: ("tracker", detector))

    # A candidate never picks up the released model's per-class cutoffs
    assert main.create_tracker("onnx", 2, model="candidate.onnx") == ("tracker", ("session", {}))
    assert calls == {"model_path": "candidate.onnx", "intra_op_threads": 2, "verify_hash": False}


def test_create_session_can_skip_hash_for_candidates(tmp_path):
    config = tmp_path / "runtime_config.json"
    config.write_text(json.dumps({"onnx_sha256": "0" * 64, "configuration": {}}), encoding="utf-8")

    # The hash check is skipped, so the failure comes later, from the missing settings
    with pytest.raises(KeyError):
        create_session(tmp_path / "candidate.onnx", config, verify_hash=False)


# Cross-validation by location (6.8)

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


def test_run_label_marks_non_default_seeds():
    from scripts.cross_validate import run_label

    assert run_label("long", 42) == "long"
    assert run_label("long", 7) == "long_seed7"


def test_summarize_keeps_seed_variants_separate():
    def run(label, fold, value):
        report = {"mAP50_95": value, "mAP50": value, "recall": 0.5, "background_fp_rate": 0.0, "per_class": []}
        return {"config": label, "fold": fold, "best": report, "last": report, "best_epoch": 30, "epochs_run": 60, "train_seconds": 60}

    summary = summarize([run("long", "fold1", 0.4), run("long_seed7", "fold1", 0.6)])

    assert summary["long"]["mAP50_95"] == 0.4
    assert summary["long_seed7"]["mAP50_95"] == 0.6


# Per-class confidence cutoffs (6.8)

def test_detector_applies_class_confidence():
    output = make_output([(320, 320, 100, 50, 1, 0.30), (100, 100, 40, 40, 0, 0.45), (500, 500, 40, 40, 2, 0.45)])
    detector = OnnxDetector(session=FakeSession(output), confidence=0.40, class_confidence={"cone": 0.25, "rock": 0.50})

    assert sorted(detection.label for detection in detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))) == ["cone", "debris"]

    with pytest.raises(ValueError, match="unknown classes"):
        OnnxDetector(session=FakeSession(output), class_confidence={"person": 0.5})


def test_pytorch_detector_applies_class_confidence():
    fake_model = Mock()
    fake_model.predict.return_value = [SimpleNamespace(boxes=[make_box(0.30)], names={0: "person"})]

    assert len(ObjectDetector(model=fake_model, confidence=0.40, class_confidence={"person": 0.25}).detect(np.zeros((480, 640, 3), dtype=np.uint8))) == 1
    assert fake_model.predict.call_args.kwargs["conf"] == 0.25
    assert ObjectDetector(model=fake_model, confidence=0.40, class_confidence={}).detect(np.zeros((480, 640, 3), dtype=np.uint8)) == []


PX = np.linspace(0, 1, 1000)


def peak(at):
    return 1 - np.abs(PX - at)


def test_best_cutoffs_keeps_classes_apart_when_one_is_missing():
    # The first run has no rock, so its curve rows are cone and debris only
    curves = [{1: peak(0.3), 2: peak(0.6)}, {0: peak(0.5), 1: peak(0.3), 2: peak(0.6)}]

    cutoffs = best_cutoffs(curves, PX)

    assert cutoffs["rock"]["cutoff"] == pytest.approx(0.5, abs=0.002) and cutoffs["rock"]["runs"] == 1
    assert cutoffs["cone"]["cutoff"] == pytest.approx(0.3, abs=0.002) and cutoffs["cone"]["runs"] == 2
    assert cutoffs["debris"]["cutoff"] == pytest.approx(0.6, abs=0.002)


def test_operating_point_reads_each_class_row():
    # Rows follow ap_class_index: row 0 is cone and row 1 is debris; rock has no row
    box = SimpleNamespace(ap_class_index=np.array([1, 2]), px=PX, p_curve=np.array([np.full(1000, 0.9), np.full(1000, 0.5)]), r_curve=np.array([np.full(1000, 0.8), np.full(1000, 0.4)]))

    result = operating_point(box, {"rock": 0.4, "cone": 0.23, "debris": 0.4})

    assert result == {"cone": {"cutoff": 0.23, "precision": 0.9, "recall": 0.8}, "debris": {"cutoff": 0.4, "precision": 0.5, "recall": 0.4}}


def test_class_confidence_applies_only_to_its_model(tmp_path):
    onnx, checkpoint, other = tmp_path / "m.onnx", tmp_path / "m.pt", tmp_path / "other.onnx"
    onnx.write_bytes(b"onnx")
    checkpoint.write_bytes(b"pt")
    other.write_bytes(b"other")
    tuning = tmp_path / "tuning.json"
    tuning.write_text(json.dumps({"classes": {"cone": {"cutoff": 0.23}, "rock": {"cutoff": 0.5}}}))
    released = tmp_path / "class_confidence.json"

    assert load_class_confidence(onnx, released) == {}

    write_release(tuning, onnx, checkpoint, released)

    assert load_class_confidence(onnx, released) == {"cone": 0.23, "rock": 0.5}
    assert load_class_confidence(checkpoint, released) == {"cone": 0.23, "rock": 0.5}

    with pytest.raises(ValueError, match="different model"):
        load_class_confidence(other, released)


def test_held_out_frames_follow_the_testset_ranges():
    manifest = {"frames": [{"file": f"more_cones_f{index:05d}.jpg", "clip": "more_cones", "frame_index": index} for index in (1415, 1430, 1945, 1960)]}
    testsets = [{"session": "session_005", "clips": ["more_cones"], "exclude_ranges": {"more_cones": [[0, 1430], [1960, None]]}}]

    assert held_out_frames("session_005", manifest, testsets) == {"more_cones_f01430.jpg", "more_cones_f01945.jpg"}
    assert held_out_frames("session_004", manifest, testsets) == set()


def test_split_fold_refuses_test_frames_outside_reserved_groups():
    entries = [{"path": Path("a.jpg"), "group": "a", "label": "", "held_out": False}, {"path": Path("fence.jpg"), "group": "fence", "label": "", "held_out": True}]

    split_fold(entries, {"name": "f", "val_groups": ["a"]}, {"fence"})

    with pytest.raises(ValueError, match="test-set frames"):
        split_fold(entries, {"name": "f", "val_groups": ["a"]}, set())
