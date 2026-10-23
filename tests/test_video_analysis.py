

from scripts.analyze_video import (
    FRAGMENT_FRAMES,
    summarize_classes,
    summarize_expected,
    summarize_tracks,
    summarize_zone,
)


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
