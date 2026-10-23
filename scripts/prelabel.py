

"""Pre-label extracted frames with a model and package them for correction in CVAT."""

from __future__ import annotations

import argparse
import fnmatch
import json
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import yaml

from edgesight.detector import OnnxDetector, create_session
from scripts.extract_frames import MANIFEST_NAME, RAW_ROOT


# CVAT task labels, in class-ID order (matches the earlier CVAT export)
CVAT_NAMES = {0: "Rock", 1: "Cone", 2: "Debris"}

# Lower than the app's 0.40, so more candidate boxes are suggested for review
PRELABEL_CONFIDENCE = 0.25

# Clips whose frames are background examples; they get empty labels later and are not sent to CVAT
DEFAULT_EXCLUDE = ("empty*",)


def to_yolo_lines(boxes, class_ids, frame_size):
    """Convert xyxy pixel boxes to YOLO lines: class x_center y_center width height (normalized)."""
    width, height = frame_size
    lines = []

    for (x1, y1, x2, y2), class_id in zip(boxes, class_ids):
        x1, x2 = max(0.0, float(x1)), min(float(width), float(x2))
        y1, y2 = max(0.0, float(y1)), min(float(height), float(y2))

        if x2 <= x1 or y2 <= y1:
            continue

        lines.append(f"{int(class_id)} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")

    return lines


def frames_to_label(manifest, exclude_patterns):
    """Frames from clips that do not match any exclude pattern."""
    return [frame for frame in manifest["frames"] if not any(fnmatch.fnmatch(frame["clip"], pattern) for pattern in exclude_patterns)]


def write_cvat_packages(session_dir: Path, labels: dict[str, list[str]]):
    """
    Write cvat_images.zip (images to upload to a CVAT task) and
    cvat_prelabels.zip (annotations to import as "Ultralytics YOLO Detection 1.0").
    """
    images_zip = session_dir / "cvat_images.zip"
    annotations_zip = session_dir / "cvat_prelabels.zip"

    with zipfile.ZipFile(images_zip, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in labels:
            archive.write(session_dir / name, arcname=name)

    data_yaml = yaml.safe_dump({"names": CVAT_NAMES, "path": ".", "train": "train.txt"}, sort_keys=False)
    train_txt = "".join(f"data/images/train/{name}\n" for name in labels)

    with zipfile.ZipFile(annotations_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("data.yaml", data_yaml)
        archive.writestr("train.txt", train_txt)

        for name, lines in labels.items():
            archive.writestr(f"labels/train/{Path(name).stem}.txt", "".join(line + "\n" for line in lines))

    return images_zip, annotations_zip


def main():
    parser = argparse.ArgumentParser(description="Pre-label a raw session's frames for correction in CVAT.")

    parser.add_argument("--session", required=True, help="Session folder under datasets/rover_obstacles/raw/ (from scripts/extract_frames.py)")
    parser.add_argument("--confidence", type=float, default=PRELABEL_CONFIDENCE)
    parser.add_argument("--model", type=Path, default=None, help="Candidate .onnx model (default: the released models/selected.onnx)")
    parser.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDE), help="Clip-name patterns to leave out (default: %(default)s)")

    args = parser.parse_args()

    if not 0.0 < args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1")

    session_dir = RAW_ROOT / args.session
    manifest_path = session_dir / MANIFEST_NAME

    if not manifest_path.is_file():
        parser.error(f"No {MANIFEST_NAME} in {session_dir}; run scripts/extract_frames.py first")

    for name in ("cvat_images.zip", "cvat_prelabels.zip"):
        if (session_dir / name).exists():
            parser.error(f"{session_dir / name} already exists; delete it to pre-label again")

    if args.model is None:
        session = create_session()
    else:
        session = create_session(model_path=args.model, verify_hash=False)

    detector = OnnxDetector(session=session, confidence=args.confidence)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = frames_to_label(manifest, args.exclude)

    labels = {}
    counts = Counter()

    for frame_info in frames:
        frame = cv2.imread(str(session_dir / frame_info["file"]))

        if frame is None:
            raise RuntimeError(f"Could not read {frame_info['file']}")

        boxes, _, class_ids = detector.infer(frame)
        labels[frame_info["file"]] = to_yolo_lines(boxes, class_ids, (frame.shape[1], frame.shape[0]))
        counts.update(CVAT_NAMES[int(class_id)] for class_id in class_ids)

    images_zip, annotations_zip = write_cvat_packages(session_dir, labels)

    excluded = len(manifest["frames"]) - len(frames)

    print(f"Pre-labeled {len(frames)} frames ({excluded} excluded as background) at confidence {args.confidence}")
    print(f"Suggested boxes: {dict(counts)}")
    print(f"Upload to a CVAT task:  {images_zip}")
    print(f"Then import annotations (format 'Ultralytics YOLO Detection 1.0'): {annotations_zip}")


if __name__ == "__main__":
    main()
