

"""Measure detection and tracking quality on a recorded video."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

from edgesight import config
from edgesight.detector import OnnxDetector, create_session, sha256_file
from edgesight.events import ZoneMonitor
from edgesight.sources import VideoFileSource
from edgesight.tracker import OnnxTracker


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = PROJECT_ROOT / "runs" / "video_analysis"

CLASSES = ("rock", "cone", "debris")

# Tracks observed in fewer frames than this count as fragments (flicker or ID switches)
FRAGMENT_FRAMES = 5


def process_video(path, tracker, zone):
    """Run tracking and the zone monitor on every frame; return per-frame records and the video's properties."""
    source = VideoFileSource(path, realtime=False).start()

    records = []
    frame_size = None
    last_frame_id = -1

    try:
        while source.is_running:
            packet = source.read_latest(after_frame_id=last_frame_id, timeout=1.0)

            if packet is None:
                continue

            last_frame_id = packet.frame_id
            frame_size = (packet.frame.shape[1], packet.frame.shape[0])

            start = time.perf_counter()
            objects = tracker.track(packet.frame)
            processing_ms = (time.perf_counter() - start) * 1000

            status, events = zone.update(objects, packet.frame.shape)

            records.append({
                "frame": packet.frame_id,
                "processing_ms": round(processing_ms, 3),
                "zone_status": status,
                "events": [[event.track_id, event.action] for event in events],
                "objects": [{"track_id": obj.track_id, "label": obj.label, "confidence": round(obj.confidence, 4), "bbox": list(obj.bbox)} for obj in objects],
            })
    finally:
        source.stop()

    return records, source.fps, frame_size


def summarize_classes(records):
    """Per class: frames present, detections, mean confidence, and presence toggles (flicker)."""
    summary = {}

    for name in CLASSES:
        present = [any(obj["label"] == name for obj in record["objects"]) for record in records]
        confidences = [obj["confidence"] for record in records for obj in record["objects"] if obj["label"] == name]

        summary[name] = {
            "frames_present": sum(present),
            "frames_present_percent": round(100 * sum(present) / len(records), 1) if records else 0.0,
            "detections": len(confidences),
            "mean_confidence": round(statistics.mean(confidences), 3) if confidences else None,
            "presence_toggles": sum(1 for before, after in zip(present, present[1:]) if before != after),
        }

    return summary


def summarize_tracks(records):
    """Lifetime, gaps, and label consistency for every track ID."""
    tracks = {}

    for record in records:
        for obj in record["objects"]:
            if obj["track_id"] is None:
                continue

            track = tracks.setdefault(obj["track_id"], {"first": record["frame"], "last": record["frame"], "labels": []})
            track["last"] = record["frame"]
            track["labels"].append(obj["label"])

    per_track = {}

    for track_id, track in sorted(tracks.items()):
        observed = len(track["labels"])
        labels = Counter(track["labels"])

        per_track[track_id] = {
            "first_frame": track["first"],
            "last_frame": track["last"],
            "observed_frames": observed,
            "gap_frames": (track["last"] - track["first"] + 1) - observed,
            "majority_label": labels.most_common(1)[0][0],
            "label_counts": dict(labels),
            "label_changes": sum(1 for before, after in zip(track["labels"], track["labels"][1:]) if before != after),
        }

    observed = [track["observed_frames"] for track in per_track.values()]

    return {
        "track_count": len(per_track),
        "fragments": sum(1 for count in observed if count < FRAGMENT_FRAMES),
        "median_observed_frames": statistics.median(observed) if observed else None,
        "tracks_with_label_changes": sum(1 for track in per_track.values() if track["label_changes"]),
        "tracks_by_majority_label": dict(Counter(track["majority_label"] for track in per_track.values())),
        "untracked_detections": sum(1 for record in records for obj in record["objects"] if obj["track_id"] is None),
        "per_track": per_track,
    }


def summarize_expected(records, expected):
    """Score the clip against what it is known to contain ("none" for an obstacle-free scene)."""
    frames = len(records)

    if expected == "none":
        false_frames = sum(1 for record in records if record["objects"])

        return {
            "expected": "none",
            "false_positive_frames": false_frames,
            "false_positive_frames_percent": round(100 * false_frames / frames, 1) if frames else 0.0,
            "false_positive_detections": sum(len(record["objects"]) for record in records),
        }

    hit_frames = sum(1 for record in records if any(obj["label"] == expected for obj in record["objects"]))
    other = [obj for record in records for obj in record["objects"] if obj["label"] != expected]

    return {
        "expected": expected,
        "expected_frames": hit_frames,
        "expected_frames_percent": round(100 * hit_frames / frames, 1) if frames else 0.0,
        "other_class_detections": len(other),
        "other_class_frames": sum(1 for record in records if any(obj["label"] != expected for obj in record["objects"])),
        "other_class_counts": dict(Counter(obj["label"] for obj in other)),
    }


def summarize_zone(records):
    events = Counter(action for record in records for _, action in record["events"])

    return {
        "caution_frames": sum(1 for record in records if record["zone_status"] == "CAUTION"),
        "enter_events": events.get("ENTER", 0),
        "exit_events": events.get("EXIT", 0),
    }


def summarize_processing(records):
    values = [record["processing_ms"] for record in records]

    return {
        "median_ms": round(statistics.median(values), 3) if values else None,
        "p95_ms": round(statistics.quantiles(values, n=20)[-1], 3) if len(values) > 1 else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Measure detection and tracking quality on a recorded video.")

    parser.add_argument("video", type=Path)
    parser.add_argument("--expect", choices=(*CLASSES, "none"), required=True, help="Class the clip contains, or 'none' for an obstacle-free scene")
    parser.add_argument("--threads", type=int, default=config.ONNX_INTRA_OP_THREADS)
    parser.add_argument("--model", type=Path, default=None, help="Candidate .onnx model (default: the released models/selected.onnx)")
    parser.add_argument("--name", default=None, help="Output name (default: the video's file name)")
    parser.add_argument("--role", choices=["test", "diagnostic"], default="test", help="test: a held-out test clip (scripts/build_dataset.py refuses to train on it); diagnostic: training footage analyzed for reference")

    args = parser.parse_args()

    if not args.video.is_file():
        parser.error(f"Video not found: {args.video}")

    if args.model is not None and (args.model.suffix.lower() != ".onnx" or not args.model.is_file()):
        parser.error(f"--model must be an existing .onnx file: {args.model}")

    if args.model is None:
        session = create_session(intra_op_threads=args.threads)
    else:
        session = create_session(model_path=args.model, intra_op_threads=args.threads, verify_hash=False)

    model_path = config.ONNX_MODEL_PATH if args.model is None else args.model

    tracker = OnnxTracker(OnnxDetector(session=session))

    records, fps, frame_size = process_video(args.video, tracker, ZoneMonitor())

    report = {
        "video": str(args.video),
        "role": args.role,
        "video_sha256": sha256_file(args.video),
        "model": str(model_path),
        "onnx_sha256": sha256_file(model_path),
        "settings": {
            "backend": "onnx",
            "intra_op_threads": args.threads,
            "confidence": config.DETECTION_CONFIDENCE,
            "nms_iou": config.NMS_IOU_THRESHOLD,
            "tracker": config.TRACKER_CONFIG,
            "fragment_frames": FRAGMENT_FRAMES,
        },
        "frames": len(records),
        "fps": fps,
        "frame_size": frame_size,
        "expected": summarize_expected(records, args.expect),
        "classes": summarize_classes(records),
        "tracks": summarize_tracks(records),
        "zone": summarize_zone(records),
        "processing": summarize_processing(records),
        "per_frame": records,
    }

    output_path = OUTPUT_ROOT / f"{args.name or args.video.stem}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    expected = report["expected"]
    tracks = report["tracks"]

    print(f"{args.video.name}: {len(records)} frames at {fps:.1f} FPS, {frame_size[0]}x{frame_size[1]}")

    if args.expect == "none":
        print(f"  False positives: {expected['false_positive_detections']} detections in {expected['false_positive_frames']} frames ({expected['false_positive_frames_percent']}%)")
    else:
        print(f"  {args.expect} detected in {expected['expected_frames']} frames ({expected['expected_frames_percent']}%)")
        print(f"  Other classes: {expected['other_class_detections']} detections in {expected['other_class_frames']} frames {expected['other_class_counts']}")

    print(f"  Tracks: {tracks['track_count']} IDs, {tracks['fragments']} fragments (<{FRAGMENT_FRAMES} frames), {tracks['tracks_with_label_changes']} with label changes, median {tracks['median_observed_frames']} frames")
    print(f"  Zone: {report['zone']['caution_frames']} CAUTION frames, {report['zone']['enter_events']} ENTER / {report['zone']['exit_events']} EXIT")
    print(f"  Processing: median {report['processing']['median_ms']} ms, p95 {report['processing']['p95_ms']} ms")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
