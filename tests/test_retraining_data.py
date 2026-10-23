

import json
import sys
import zipfile

import cv2
import numpy as np
import pytest

import scripts.build_dataset as build_dataset
import scripts.extract_frames as extract_frames
from scripts.prelabel import frames_to_label, to_yolo_lines, write_cvat_packages


def write_video(path, values, size=(160, 90), fps=30.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)

    for value in values:
        frame = np.full((size[1], size[0], 3), value, dtype=np.uint8)

        # High-contrast grid lines give every frame the same sharp edges for the blur check
        line = 0 if value > 127 else 255
        frame[::10, :] = line
        frame[:, ::10] = line
        writer.write(frame)

    writer.release()

    return path


# extract_frames

def make_candidate(index, score, value):
    return (index, score, np.full((36, 64), value, dtype=np.float32))


def test_select_frames_skips_blurry_and_duplicates():
    candidates = [
        make_candidate(0, 100.0, 0),
        make_candidate(15, 20.0, 50),    # blurry: below half the median
        make_candidate(30, 100.0, 1),    # near-duplicate of frame 0
        make_candidate(45, 100.0, 80),
    ]

    kept, skipped = extract_frames.select_frames(candidates, 0.5, 4.0)

    assert kept == [0, 45]
    assert skipped == {"blurry": 1, "duplicate": 1}


def test_select_frames_empty():
    assert extract_frames.select_frames([], 0.5, 4.0) == ([], {"blurry": 0, "duplicate": 0})


def test_sharpness_ranks_blur_lower():
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)

    assert extract_frames.sharpness(cv2.GaussianBlur(sharp, (21, 21), 0)) < extract_frames.sharpness(sharp)


def test_resize_longest_side():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    assert extract_frames.resize_longest_side(frame, 1280).shape == (720, 1280, 3)
    assert extract_frames.resize_longest_side(frame, 4000).shape == (1080, 1920, 3)


def test_extract_frames_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(extract_frames, "RAW_ROOT", tmp_path / "raw")

    clip = write_video(tmp_path / "rock_test_1.avi", [index * 6 for index in range(40)])

    monkeypatch.setattr(sys, "argv", ["extract_frames", str(clip), "--session", "s1", "--every", "10"])
    extract_frames.main()

    manifest = json.loads((tmp_path / "raw" / "s1" / "frames.json").read_text(encoding="utf-8"))

    assert manifest["clips"]["rock_test_1"]["sampled"] == 4
    assert [frame["frame_index"] for frame in manifest["frames"]] == [0, 10, 20, 30]
    assert all((tmp_path / "raw" / "s1" / frame["file"]).is_file() for frame in manifest["frames"])
    assert manifest["frames"][0]["file"] == "rock_test_1_f00000.jpg"


def test_extract_frames_refuses_existing_session(tmp_path, monkeypatch):
    monkeypatch.setattr(extract_frames, "RAW_ROOT", tmp_path / "raw")
    (tmp_path / "raw" / "s1").mkdir(parents=True)

    clip = write_video(tmp_path / "clip.avi", [0, 50])

    monkeypatch.setattr(sys, "argv", ["extract_frames", str(clip), "--session", "s1"])

    with pytest.raises(SystemExit):
        extract_frames.main()


# prelabel

def test_to_yolo_lines_normalizes_and_clips():
    lines = to_yolo_lines([(-10, 20, 110, 70), (50, 50, 50, 60)], [2, 0], (100, 100))

    # The second box has zero width and is dropped; the first is clipped to the image
    assert lines == ["2 0.500000 0.450000 1.000000 0.500000"]


def test_frames_to_label_excludes_background_clips():
    manifest = {"frames": [{"file": "a.jpg", "clip": "rock_1"}, {"file": "b.jpg", "clip": "empty_sidewalk"}]}

    assert [frame["file"] for frame in frames_to_label(manifest, ["empty*"])] == ["a.jpg"]


def test_cvat_packages_round_trip(tmp_path):
    (tmp_path / "rock_1_f00000.jpg").write_bytes(b"jpeg")
    (tmp_path / "rock_1_f00015.jpg").write_bytes(b"jpeg")

    labels = {"rock_1_f00000.jpg": ["0 0.5 0.5 0.2 0.2"], "rock_1_f00015.jpg": []}

    images_zip, annotations_zip = write_cvat_packages(tmp_path, labels)

    with zipfile.ZipFile(images_zip) as archive:
        assert sorted(archive.namelist()) == sorted(labels)

    with zipfile.ZipFile(annotations_zip) as archive:
        assert archive.read("train.txt").decode() == "data/images/train/rock_1_f00000.jpg\ndata/images/train/rock_1_f00015.jpg\n"

    # build_dataset reads the same format back, including the empty label
    assert build_dataset.read_cvat_export(annotations_zip) == {"rock_1_f00000": "0 0.5 0.5 0.2 0.2\n", "rock_1_f00015": ""}


# build_dataset

def write_cvat_export(path, names, labels):
    (path / "labels" / "train").mkdir(parents=True)
    (path / "data.yaml").write_text(json.dumps({"names": names, "path": ".", "train": "train.txt"}), encoding="utf-8")

    for stem, text in labels.items():
        (path / "labels" / "train" / f"{stem}.txt").write_text(text, encoding="utf-8")

    return path


def test_cvat_export_class_order_checked(tmp_path):
    export = write_cvat_export(tmp_path / "export", {0: "Cone", 1: "Rock", 2: "Debris"}, {})

    with pytest.raises(ValueError, match="does not match"):
        build_dataset.read_cvat_export(export)


def test_plan_frames_assigns_splits_and_backgrounds():
    manifest = {"frames": [
        {"file": "rock_1_f00000.jpg", "clip": "rock_1"},
        {"file": "empty_1_f00000.jpg", "clip": "empty_1"},
        {"file": "debris_9_f00000.jpg", "clip": "debris_9"},
    ]}

    entries, skipped = build_dataset.plan_frames(manifest, {"rock_1": "val", "empty_1": "train"}, ["empty*"], {"rock_1_f00000": "0 0.5 0.5 0.1 0.1\n"})

    assert [(entry["file"], entry["split"], entry["label"]) for entry in entries] == [
        ("rock_1_f00000.jpg", "val", "0 0.5 0.5 0.1 0.1\n"),
        ("empty_1_f00000.jpg", "train", ""),
    ]
    assert skipped == ["debris_9"]


def test_plan_frames_requires_labels_for_object_clips():
    manifest = {"frames": [{"file": "rock_1_f00000.jpg", "clip": "rock_1"}]}

    with pytest.raises(ValueError, match="No CVAT label"):
        build_dataset.plan_frames(manifest, {"rock_1": "train"}, [], {})


def test_plan_frames_rejects_test_split():
    manifest = {"frames": [{"file": "rock_1_f00000.jpg", "clip": "rock_1"}]}

    with pytest.raises(ValueError, match="train or val"):
        build_dataset.plan_frames(manifest, {"rock_1": "test"}, [], {"rock_1_f00000": ""})


def make_base_dataset(root):
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)

        image = np.full((32, 32, 3), {"train": 10, "val": 20, "test": 30}[split], dtype=np.uint8)
        cv2.imwrite(str(root / "images" / split / f"stock_{split}.jpg"), image)
        (root / "labels" / split / f"stock_{split}.txt").write_text("1 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    return root


def make_session(raw_root, name, clips):
    session = raw_root / name
    session.mkdir(parents=True)

    frames = []

    for index, clip in enumerate(clips):
        file = f"{clip}_f00000.jpg"
        cv2.imwrite(str(session / file), np.full((32, 32, 3), 100 + index * 40, dtype=np.uint8))
        frames.append({"file": file, "clip": clip, "frame_index": 0})

    manifest = {"clips": {clip: {"sha256": f"hash-{clip}"} for clip in clips}, "frames": frames}
    (session / "frames.json").write_text(json.dumps(manifest), encoding="utf-8")


def run_build(tmp_path, monkeypatch, plan, extra_args=()):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["build_dataset", "--plan", str(plan_path), "--output", str(tmp_path / "v2"), *extra_args])
    build_dataset.main()


def test_build_dataset_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(build_dataset, "RAW_ROOT", tmp_path / "raw")
    monkeypatch.setattr(build_dataset, "TEST_CLIP_RESULTS", tmp_path / "no_results")

    make_base_dataset(tmp_path / "v1")
    make_session(tmp_path / "raw", "s1", ["rock_1", "rock_2", "empty_1"])
    export = write_cvat_export(tmp_path / "export", {0: "Rock", 1: "Cone", 2: "Debris"}, {"rock_1_f00000": "0 0.5 0.5 0.2 0.2\n", "rock_2_f00000": "0 0.4 0.4 0.2 0.2\n"})

    plan = {
        "base": {"dataset": str(tmp_path / "v1"), "splits": {"train": "train", "test": "test"}},
        "sessions": [{"session": "s1", "labels": str(export), "clips": {"rock_1": "train", "rock_2": "val", "empty_1": "train"}}],
        "background_clips": ["empty*"],
    }

    run_build(tmp_path, monkeypatch, plan, ["--lock", str(tmp_path / "lock.json")])

    output = tmp_path / "v2"

    assert sorted(path.name for path in (output / "images" / "train").iterdir()) == ["empty_1_f00000.jpg", "rock_1_f00000.jpg", "stock_train.jpg"]
    assert [path.name for path in (output / "images" / "val").iterdir()] == ["rock_2_f00000.jpg"]
    assert [path.name for path in (output / "images" / "test").iterdir()] == ["stock_test.jpg"]
    assert (output / "labels" / "train" / "empty_1_f00000.txt").read_text(encoding="utf-8") == ""

    lock = json.loads((tmp_path / "lock.json").read_text(encoding="utf-8"))
    assert set(lock) == {"val", "test"}
    assert set(lock["val"]) == {"rock_2_f00000.jpg"}


def test_build_dataset_refuses_test_clips(tmp_path, monkeypatch):
    monkeypatch.setattr(build_dataset, "RAW_ROOT", tmp_path / "raw")

    results = tmp_path / "video_analysis"
    results.mkdir()
    (results / "rocks.json").write_text(json.dumps({"video_sha256": "hash-rock_1"}), encoding="utf-8")
    monkeypatch.setattr(build_dataset, "TEST_CLIP_RESULTS", results)

    make_base_dataset(tmp_path / "v1")
    make_session(tmp_path / "raw", "s1", ["rock_1"])

    plan = {
        "base": {"dataset": str(tmp_path / "v1"), "splits": {"train": "train", "test": "test"}},
        "sessions": [{"session": "s1", "clips": {"rock_1": "train"}}],
        "background_clips": ["rock*"],
    }

    with pytest.raises(ValueError, match="test clip"):
        run_build(tmp_path, monkeypatch, plan)
