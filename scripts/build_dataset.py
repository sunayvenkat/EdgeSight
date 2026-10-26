

"""Build a training dataset from the v1 data plus labeled real-world frames and background frames."""

from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

import yaml

from edgesight.detector import sha256_file
from scripts.audit_splits import build_manifest, find_cross_split_duplicates
from scripts.extract_frames import MANIFEST_NAME, RAW_ROOT
from scripts.validate_dataset import validate_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLASS_NAMES = {0: "rock", 1: "cone", 2: "debris"}

SPLITS = ("train", "val", "test")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Clips evaluated in docs/video_evaluation.md; they must never be used for training
TEST_CLIP_RESULTS = PROJECT_ROOT / "runs" / "video_analysis"


def resolve(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_cvat_xml(xml_text: str) -> dict[str, str]:
    """
    Convert a "CVAT for images 1.1" annotations.xml to {image stem: YOLO label text}.

    Classes are matched by label name, ignoring case. Images with no boxes get
    an empty label. Rotated boxes and other shape types are rejected, since
    YOLO detection labels are axis-aligned boxes.
    """
    class_ids = {name: class_id for class_id, name in CLASS_NAMES.items()}
    root = ElementTree.fromstring(xml_text)
    labels = {}

    for image in root.iter("image"):
        width, height = float(image.get("width")), float(image.get("height"))
        lines = []

        for shape in image:
            if shape.tag != "box":
                raise ValueError(f"{image.get('name')}: unsupported CVAT shape <{shape.tag}>; use rectangles only")

            if float(shape.get("rotation", "0")) != 0:
                raise ValueError(f"{image.get('name')}: rotated box; YOLO boxes must be axis-aligned")

            name = shape.get("label", "").lower()

            if name not in class_ids:
                raise ValueError(f"{image.get('name')}: unknown label {shape.get('label')!r}")

            x1, y1, x2, y2 = (float(shape.get(key)) for key in ("xtl", "ytl", "xbr", "ybr"))
            lines.append(f"{class_ids[name]} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")

        labels[Path(image.get("name")).stem] = "".join(line + "\n" for line in lines)

    return labels


def read_cvat_export(path: Path) -> dict[str, str]:
    """
    Read a CVAT export (folder or zip): "Ultralytics YOLO Detection 1.0", or
    "CVAT for images 1.1" (annotations.xml).

    Returns {image stem: label text}. For YOLO exports, the class order must
    match CLASS_NAMES, ignoring case, so classes cannot be silently swapped;
    XML exports are matched by label name.
    """
    if path.is_dir():
        if (path / "annotations.xml").is_file():
            return read_cvat_xml((path / "annotations.xml").read_text(encoding="utf-8"))

        files = {str(item.relative_to(path)).replace("\\", "/"): item.read_text(encoding="utf-8") for item in path.rglob("*") if item.is_file() and item.suffix in {".yaml", ".txt"}}
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            if "annotations.xml" in archive.namelist():
                return read_cvat_xml(archive.read("annotations.xml").decode("utf-8"))

            files = {name: archive.read(name).decode("utf-8") for name in archive.namelist() if name.endswith((".yaml", ".txt"))}
    else:
        raise FileNotFoundError(f"CVAT export not found: {path}")

    if "data.yaml" not in files:
        raise ValueError(f"{path} has no data.yaml or annotations.xml; export from CVAT as 'Ultralytics YOLO Detection 1.0' or 'CVAT for images 1.1'")

    names = yaml.safe_load(files["data.yaml"]).get("names", {})
    export_names = {int(key): str(value).lower() for key, value in names.items()}

    if export_names != CLASS_NAMES:
        raise ValueError(f"CVAT class order {export_names} does not match {CLASS_NAMES}")

    return {Path(name).stem: text for name, text in files.items() if name.startswith("labels/") and name.endswith(".txt")}


def remove_label_lines(labels, remove_boxes):
    """
    Drop specific boxes from exported labels without editing the export.

    remove_boxes maps an image stem to 1-based line numbers in that image's
    exported label text (for example, single leaves boxed as debris against
    the labeling rules). Unknown stems or line numbers raise, so a stale list
    cannot silently remove the wrong boxes.
    """
    labels = dict(labels)

    for stem, line_numbers in remove_boxes.items():
        if stem not in labels:
            raise ValueError(f"remove_boxes: no labels for {stem}")

        lines = labels[stem].splitlines()
        invalid = [number for number in line_numbers if not 1 <= number <= len(lines)]

        if invalid:
            raise ValueError(f"remove_boxes: {stem} has no line(s) {invalid}")

        drop = set(line_numbers)
        labels[stem] = "".join(line + "\n" for number, line in enumerate(lines, start=1) if number not in drop)

    return labels


def clip_label_text(text, tolerance=1e-4):
    """
    Clip YOLO boxes that overshoot the image edge by less than tolerance.

    CVAT rounds coordinates to 6 decimals, so a box snapped to the edge can
    end 0.0000005 outside the image. Larger overshoots are real labeling
    errors and raise. Returns (text, number of clipped boxes).
    """
    lines = []
    clipped = 0

    for line in text.splitlines():
        if not line.strip():
            continue

        class_id, x, y, w, h = line.split()
        x, y, w, h = map(float, (x, y, w, h))
        edges = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]

        if min(edges) < -tolerance or max(edges) > 1 + tolerance:
            raise ValueError(f"Box extends outside the image: {line}")

        # Pull overshooting edges just inside the image, so 6-decimal rounding cannot push them back out
        margin = 1e-6
        bounded = [margin if edge < 0 else 1 - margin if edge > 1 else edge for edge in edges]

        if bounded != edges:
            clipped += 1
            left, top, right, bottom = bounded
            x, y, w, h = (left + right) / 2, (top + bottom) / 2, right - left, bottom - top

        lines.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    return "".join(line + "\n" for line in lines), clipped


def plan_frames(session_manifest, clip_splits, background_patterns, labels, empty_frames=()):
    """
    Decide each frame's split and label text.

    Returns (entries, skipped_clips). Frames from background clips, and
    frames listed in empty_frames (reviewed in CVAT and left with no boxes,
    which CVAT exports as no file), get an empty label. Any other frame
    without a CVAT label is an error, so a missed export cannot silently turn
    objects into backgrounds.
    """
    frame_stems = {Path(frame["file"]).stem for frame in session_manifest["frames"]}
    unknown = sorted(set(empty_frames) - frame_stems)

    if unknown:
        raise ValueError(f"empty_frames not in this session: {', '.join(unknown)}")

    entries = []
    skipped = set()

    for frame in session_manifest["frames"]:
        split = clip_splits.get(frame["clip"])

        if split is None:
            skipped.add(frame["clip"])
            continue

        if split not in ("train", "val"):
            raise ValueError(f"Clip {frame['clip']} assigned to {split!r}; real-world clips go to train or val")

        stem = Path(frame["file"]).stem

        if any(fnmatch.fnmatch(frame["clip"], pattern) for pattern in background_patterns) or stem in empty_frames:
            label = ""
        elif stem in labels:
            label = labels[stem]
        else:
            raise ValueError(f"No CVAT label for {frame['file']} (clip {frame['clip']})")

        entries.append({"file": frame["file"], "clip": frame["clip"], "split": split, "label": label})

    return entries, sorted(skipped)


def known_test_clip_hashes() -> set[str]:
    """Hashes of clips analyzed as test clips (reports without a role predate --role and are tests)."""
    if not TEST_CLIP_RESULTS.is_dir():
        return set()

    reports = [json.loads(path.read_text(encoding="utf-8")) for path in TEST_CLIP_RESULTS.glob("*.json")]

    return {report["video_sha256"] for report in reports if report.get("role", "test") == "test"}


def count_boxes(output: Path):
    counts = {split: Counter() for split in SPLITS}
    backgrounds = Counter()

    for split in SPLITS:
        for label in (output / "labels" / split).glob("*.txt"):
            lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]

            if not lines:
                backgrounds[split] += 1

            counts[split].update(CLASS_NAMES[int(line.split()[0])] for line in lines)

    return counts, backgrounds


def copy_split(source_root: Path, source_split: str, output: Path, target_split: str):
    for image in sorted((source_root / "images" / source_split).iterdir()):
        if image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        shutil.copy2(image, output / "images" / target_split / image.name)
        shutil.copy2(source_root / "labels" / source_split / f"{image.stem}.txt", output / "labels" / target_split / f"{image.stem}.txt")


def main():
    parser = argparse.ArgumentParser(description="Build a dataset from a plan file (see docs/retraining.md).")

    parser.add_argument("--plan", type=Path, required=True, help="JSON plan: base dataset, sessions, clip splits, background clips")
    parser.add_argument("--output", type=Path, required=True, help="New dataset folder, e.g. datasets/rover_obstacles_v2")
    parser.add_argument("--lock", type=Path, default=None, help="Also write a split lock (val and test hashes) to this new file")

    args = parser.parse_args()

    plan = json.loads(resolve(args.plan).read_text(encoding="utf-8"))
    output = resolve(args.output)

    if (output / "images").exists() or (output / "labels").exists():
        parser.error(f"{output} already has images or labels; build into a new folder")

    lock_path = resolve(args.lock) if args.lock else None

    if lock_path is not None and lock_path.exists():
        parser.error(f"{lock_path} already exists; do not overwrite a split lock")

    test_hashes = known_test_clip_hashes()
    background_patterns = plan.get("background_clips", [])
    planned = []
    clipped_total = 0

    for session in plan["sessions"]:
        session_dir = RAW_ROOT / session["session"]
        manifest = json.loads((session_dir / MANIFEST_NAME).read_text(encoding="utf-8"))

        for clip in session["clips"]:
            if clip not in manifest["clips"]:
                raise ValueError(f"Clip {clip} is not in session {session['session']}")

            if manifest["clips"][clip]["sha256"] in test_hashes:
                raise ValueError(f"Clip {clip} is a test clip from docs/video_evaluation.md and cannot be used for training")

        labels = read_cvat_export(resolve(session["labels"])) if session.get("labels") else {}
        labels = remove_label_lines(labels, session.get("remove_boxes", {}))

        for stem, text in labels.items():
            labels[stem], clipped = clip_label_text(text)
            clipped_total += clipped

        entries, skipped = plan_frames(manifest, session["clips"], background_patterns, labels, set(session.get("empty_frames", [])))

        if skipped:
            print(f"Note: session {session['session']} clips not in the plan were skipped: {', '.join(skipped)}")

        planned.extend((session_dir, entry) for entry in entries)

    stems = [Path(entry["file"]).stem for _, entry in planned]

    if len(set(stems)) != len(stems):
        raise ValueError("Duplicate frame names across sessions; rename clips so their names are unique")

    # Build in a staging folder and move it into place only after every check passes
    staging = output.parent / f".{output.name}.building"

    if staging.exists():
        shutil.rmtree(staging)

    try:
        for split in SPLITS:
            (staging / "images" / split).mkdir(parents=True)
            (staging / "labels" / split).mkdir(parents=True)

        base_root = resolve(plan["base"]["dataset"])

        for source_split, target_split in plan["base"]["splits"].items():
            copy_split(base_root, source_split, staging, target_split)

        for session_dir, entry in planned:
            shutil.copy2(session_dir / entry["file"], staging / "images" / entry["split"] / entry["file"])
            (staging / "labels" / entry["split"] / f"{Path(entry['file']).stem}.txt").write_text(entry["label"], encoding="utf-8")

        data_yaml = {"train": "images/train", "val": "images/val", "test": "images/test", "nc": len(CLASS_NAMES), "names": CLASS_NAMES}
        (staging / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

        errors, image_counts = validate_dataset(staging, require_images=True)

        if errors:
            raise SystemExit("Dataset validation failed:\n" + "\n".join(errors))

        manifest = build_manifest(staging)
        duplicates = find_cross_split_duplicates(manifest)

        if duplicates:
            raise SystemExit("Cross-split duplicates found:\n" + "\n".join(duplicates))

        output.mkdir(parents=True, exist_ok=True)

        for name in ("images", "labels", "data.yaml"):
            if (output / name).exists() and name == "data.yaml":
                (output / name).unlink()

            shutil.move(str(staging / name), str(output / name))
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    if clipped_total:
        print(f"Clipped {clipped_total} box(es) that overshot the image edge by less than 0.0001 (CVAT rounding)")

    if lock_path is not None:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps({split: manifest[split] for split in ("val", "test")}, indent=2), encoding="utf-8")

    box_counts, backgrounds = count_boxes(output)

    print(f"Built {output}")

    for split in SPLITS:
        print(f"  {split}: {image_counts[split]} images ({backgrounds[split]} background), boxes {dict(box_counts[split])}")

    if lock_path is not None:
        print(f"Split lock: {lock_path}")


if __name__ == "__main__":
    main()
