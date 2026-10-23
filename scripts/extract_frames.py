

"""Extract labeling frames from video clips, skipping blurry and near-duplicate frames."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from edgesight.detector import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_ROOT = PROJECT_ROOT / "datasets" / "rover_obstacles" / "raw"

MANIFEST_NAME = "frames.json"


def sharpness(frame: np.ndarray) -> float:
    """Variance of the Laplacian on a 640-wide grayscale copy; higher means sharper."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scale = 640 / gray.shape[1]

    if scale < 1:
        gray = cv2.resize(gray, (640, round(gray.shape[0] * scale)), interpolation=cv2.INTER_AREA)

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def thumbnail(frame: np.ndarray) -> np.ndarray:
    """Small grayscale copy for near-duplicate checks."""
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)


def resize_longest_side(frame: np.ndarray, max_side: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = max_side / max(height, width)

    if scale >= 1:
        return frame

    return cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


def select_frames(candidates, min_sharpness_ratio, min_change):
    """
    Choose which sampled frames to keep.

    candidates is a list of (frame_index, sharpness, thumbnail). A frame is
    skipped when its sharpness is below min_sharpness_ratio times the clip's
    median, or when it differs from the last kept frame by less than
    min_change (mean absolute difference, 0–255). Returns kept indices and a
    count of each skip reason.
    """
    if not candidates:
        return [], {"blurry": 0, "duplicate": 0}

    median = statistics.median(score for _, score, _ in candidates)
    threshold = median * min_sharpness_ratio

    kept = []
    skipped = {"blurry": 0, "duplicate": 0}
    last_thumbnail = None

    for frame_index, score, thumb in candidates:
        if score < threshold:
            skipped["blurry"] += 1
            continue

        if last_thumbnail is not None and float(np.mean(np.abs(thumb - last_thumbnail))) < min_change:
            skipped["duplicate"] += 1
            continue

        kept.append(frame_index)
        last_thumbnail = thumb

    return kept, skipped


def sample_clip(path: Path, every: int):
    """Read every Nth frame; return (frame_index, sharpness, thumbnail) candidates and the clip's properties."""
    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    candidates = []
    frame_index = 0
    size = None

    try:
        while True:
            success = capture.grab()

            if not success:
                break

            if frame_index % every == 0:
                success, frame = capture.retrieve()

                if success:
                    size = (frame.shape[1], frame.shape[0])
                    candidates.append((frame_index, sharpness(frame), thumbnail(frame)))

            frame_index += 1
    finally:
        capture.release()

    return candidates, {"fps": fps, "frames": frame_index, "size": size}


def write_frames(path: Path, frame_indices, output_dir: Path, max_side: int):
    """Decode the kept frames and save them as JPEGs named <clip>_f<index>.jpg."""
    wanted = set(frame_indices)
    written = []

    capture = cv2.VideoCapture(str(path))
    frame_index = 0

    try:
        while wanted:
            success = capture.grab()

            if not success:
                break

            if frame_index in wanted:
                success, frame = capture.retrieve()

                if success:
                    name = f"{path.stem}_f{frame_index:05d}.jpg"
                    cv2.imwrite(str(output_dir / name), resize_longest_side(frame, max_side), [cv2.IMWRITE_JPEG_QUALITY, 95])
                    written.append({"file": name, "clip": path.stem, "frame_index": frame_index})

                wanted.discard(frame_index)

            frame_index += 1
    finally:
        capture.release()

    return written


def main():
    parser = argparse.ArgumentParser(description="Extract labeling frames from video clips into a raw dataset session.")

    parser.add_argument("videos", type=Path, nargs="+")
    parser.add_argument("--session", required=True, help="Session folder name under datasets/rover_obstacles/raw/")
    parser.add_argument("--every", type=int, default=15, help="Sample every Nth frame (default: %(default)s, 2 per second at 30 FPS)")
    parser.add_argument("--max-side", type=int, default=1280, help="Resize so the longest side is at most this many pixels")
    parser.add_argument("--min-sharpness-ratio", type=float, default=0.5, help="Skip frames sharper than less than this fraction of the clip's median")
    parser.add_argument("--min-change", type=float, default=4.0, help="Skip frames that differ from the last kept frame by less than this (0-255)")

    args = parser.parse_args()

    if args.every < 1 or args.max_side < 64 or not 0 <= args.min_sharpness_ratio <= 1 or args.min_change < 0:
        parser.error("Invalid extraction settings")

    stems = [video.stem for video in args.videos]

    if len(set(stems)) != len(stems):
        parser.error("Video file names must be unique; frames are named after them")

    for video in args.videos:
        if not video.is_file():
            parser.error(f"Video not found: {video}")

    output_dir = RAW_ROOT / args.session

    if output_dir.exists():
        parser.error(f"Session already exists: {output_dir}. Choose a new --session name.")

    output_dir.mkdir(parents=True)

    manifest = {
        "session": args.session,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {"every": args.every, "max_side": args.max_side, "min_sharpness_ratio": args.min_sharpness_ratio, "min_change": args.min_change},
        "clips": {},
        "frames": [],
    }

    for video in args.videos:
        candidates, properties = sample_clip(video, args.every)
        kept, skipped = select_frames(candidates, args.min_sharpness_ratio, args.min_change)
        written = write_frames(video, kept, output_dir, args.max_side)

        manifest["clips"][video.stem] = {
            "path": str(video),
            "sha256": sha256_file(video),
            **properties,
            "sampled": len(candidates),
            "kept": len(written),
            "skipped": skipped,
        }
        manifest["frames"].extend(written)

        print(f"{video.name}: sampled {len(candidates)}, kept {len(written)}, skipped {skipped['blurry']} blurry and {skipped['duplicate']} near-duplicate")

    (output_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nSaved {len(manifest['frames'])} frames to {output_dir}")


if __name__ == "__main__":
    main()
