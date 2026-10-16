

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from edgesight import config
from edgesight.events import ZoneMonitor
from edgesight.tracker import TrackedObject, draw_tracks

from scripts.benchmark_inference import (IMAGE_DIR, load_images, sha256_file)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "selected.pt"

OUTPUT_PATH = (PROJECT_ROOT / "runs" / "pipeline_profile" / "summary.json")

MODEL_TIMING_KEYS = ("preprocess", "inference", "postprocess")


def summarize_ms(values):
    """Summarize timing measurements in milliseconds."""

    values = np.asarray(list(values), dtype=np.float64)

    if values.size == 0:
        raise ValueError("Cannot summarize empty measurements")

    if not np.all(np.isfinite(values)):
        raise ValueError("Timing measurements must be finite")

    if np.any(values < 0):
        raise ValueError("Timing measurements cannot be negative")

    return {
        "count": int(values.size),
        "mean": round(float(np.mean(values)), 4),
        "median": round(float(np.median(values)), 4),
        "p95": round(float(np.percentile(values, 95)), 4),
        "min": round(float(np.min(values)), 4),
        "max": round(float(np.max(values)), 4),
    }


def estimate_tracking_overhead_ms(model_call_ms, speed):
    """
    Estimate time outside Ultralytics' reported model stages.

    This includes tracking callbacks and framework overhead.
    It is not a pure tracker timing measurement.
    """

    internal_ms = sum(
        float(speed.get(stage, 0.0))
        for stage in MODEL_TIMING_KEYS
    )

    return max(0.0, float(model_call_ms) - internal_ms)


def result_to_objects(result):
    """Convert Ultralytics results into EdgeSight tracked objects."""

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return []

    coordinates = boxes.xyxy.cpu().numpy()

    confidences = boxes.conf.cpu().numpy()

    class_ids = (boxes.cls.cpu().numpy().astype(int))

    if boxes.id is None:
        track_ids = [None] * len(class_ids)
    else:
        track_ids = (boxes.id.int().cpu().tolist())

    objects = []

    for bbox, confidence, class_id, track_id in zip(coordinates, confidences, class_ids, track_ids):
        names = result.names

        if isinstance(names, dict):
            label = names.get(int(class_id), str(class_id))
        else:
            label = names[int(class_id)]

        integer_bbox = tuple(
            int(round(float(value)))
            for value in bbox
        )

        objects.append(
            TrackedObject(
                track_id=(
                    int(track_id)
                    if track_id is not None
                    else None
                ),
                label=str(label),
                confidence=float(confidence),
                bbox=integer_bbox,
            )
        )

    return objects


def profile_frame(model, frame, zone_monitor):
    """Profile one frame through the processing pipeline."""

    # Stage 1: model tracking call, including preprocessing,
    # inference, postprocessing, and tracking callbacks.
    start = time.perf_counter()

    results = model.track(
        source=frame,
        persist=True,
        tracker=config.TRACKER_CONFIG,
        conf=config.DETECTION_CONFIDENCE,
        imgsz=config.INFERENCE_IMAGE_SIZE,
        device="cpu",
        verbose=False,
    )

    model_call_ms = (time.perf_counter() - start) * 1000.0

    if not results:
        raise RuntimeError("The model returned no result for a frame")

    result = results[0]

    speed = result.speed or {}

    missing = [
        stage
        for stage in MODEL_TIMING_KEYS
        if stage not in speed
    ]

    if missing:
        raise RuntimeError(f"Missing Ultralytics timing fields: {missing}")

    preprocess_ms = float(speed["preprocess"])
    inference_ms = float(speed["inference"])
    postprocess_ms = float(speed["postprocess"])

    tracking_framework_overhead_ms = (estimate_tracking_overhead_ms(model_call_ms, speed))

    # Stage 2: convert results into EdgeSight objects.
    start = time.perf_counter()

    objects = result_to_objects(result)

    object_conversion_ms = (time.perf_counter() - start) * 1000.0

    # Stage 3: update the monitoring zone and generate events.
    start = time.perf_counter()

    zone_status, events = zone_monitor.update(objects, frame.shape)

    zone_monitor_ms = (time.perf_counter() - start) * 1000.0

    # Stage 4: draw bounding boxes and tracking IDs.
    start = time.perf_counter()

    _ = draw_tracks(frame, objects)

    drawing_ms = (time.perf_counter() - start) * 1000.0

    # Actual measured processing time from the start of the
    # model call through completion of drawing.
    processing_total_ms = (
        model_call_ms
        + object_conversion_ms
        + zone_monitor_ms
        + drawing_ms
    )

    return {
        "preprocess_ms": preprocess_ms,
        "inference_ms": inference_ms,
        "postprocess_ms": postprocess_ms,
        "model_call_ms": model_call_ms,
        "tracking_framework_overhead_ms": (
            tracking_framework_overhead_ms
        ),
        "object_conversion_ms": object_conversion_ms,
        "zone_monitor_ms": zone_monitor_ms,
        "drawing_ms": drawing_ms,
        "processing_total_ms": processing_total_ms,
        "detection_count": len(objects),
        "event_count": len(events),
        "zone_status": zone_status,
    }


def main():
    parser = argparse.ArgumentParser(description="Profile the EdgeSight processing pipeline")

    parser.add_argument("--warmup", type=int, default=5, help="Warm-up frames per repetition")

    parser.add_argument("--repeats", type=int, default=3, help="Number of benchmark repetitions")

    parser.add_argument("--limit", type=int, default=0, help="Maximum images to use; 0 means all")

    args = parser.parse_args()

    if args.warmup < 0:
        parser.error("--warmup cannot be negative")

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    if args.limit < 0:
        parser.error("--limit cannot be negative")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    images = load_images(IMAGE_DIR)

    if args.limit:
        images = images[:args.limit]

    if not images:
        raise RuntimeError(f"No images found in {IMAGE_DIR}")

    warmup_count = min(args.warmup, len(images),)

    measurements = []
    model_load_times = []

    for repeat in range(1, args.repeats + 1):
        print(f"\nRepetition {repeat}/{args.repeats}")

        # Keep model initialization outside the timed frames.
        load_start = time.perf_counter()

        model = YOLO(str(MODEL_PATH))

        model_load_ms = (time.perf_counter() - load_start) * 1000.0

        model_load_times.append(model_load_ms)

        # Warm up the model and tracking pipeline.
        warmup_monitor = ZoneMonitor()

        for _, frame in images[:warmup_count]:
            profile_frame(model, frame, warmup_monitor)

        # Start each measurement pass with a fresh zone monitor.
        monitor = ZoneMonitor()

        for image_path, frame in images:
            values = profile_frame(model, frame, monitor)

            values["repeat"] = repeat
            values["image"] = Path(image_path).name

            measurements.append(values)

        print(f"Completed {len(images)} measured frames")

    stage_names = [
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
        "model_call_ms",
        "tracking_framework_overhead_ms",
        "object_conversion_ms",
        "zone_monitor_ms",
        "drawing_ms",
        "processing_total_ms",
    ]

    stages = {
        stage: summarize_ms(
            row[stage]
            for row in measurements
        )
        for stage in stage_names
    }

    median_total_ms = (stages["processing_total_ms"]["median"])

    p95_total_ms = (stages["processing_total_ms"]["p95"])

    frame_budget_ms = 1000.0 / 30.0

    report = {
        "model": str(
            MODEL_PATH.relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
        "model_sha256": sha256_file(MODEL_PATH),
        "backend": "Ultralytics YOLO / PyTorch CPU",
        "image_directory": str(IMAGE_DIR),
        "image_count": len(images),
        "warmup_frames_per_repeat": warmup_count,
        "repeats": args.repeats,
        "measurement_count": len(measurements),
        "model_load_ms": summarize_ms(
            model_load_times
        ),
        "stages_ms": stages,
        "processing_limited_fps_from_median_ms": (
            round(1000.0 / median_total_ms, 2)
            if median_total_ms > 0
            else None
        ),
        "target_30_fps_budget_ms": round(
            frame_budget_ms,
            4,
        ),
        "median_within_30_fps_budget": (
            median_total_ms <= frame_budget_ms
        ),
        "p95_within_30_fps_budget": (
            p95_total_ms <= frame_budget_ms
        ),
        "camera_capture_included": False,
        "display_included": False,
        "notes": [
            "Images are loaded into memory before timing.",
            "Camera capture and GUI display are excluded.",
            "Validation images are not a temporally ordered video.",
            "Tracking overhead includes tracker callbacks and "
            "framework overhead; it is not an isolated tracker "
            "measurement.",
        ],
        "measurements": measurements,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nEdgeSight Pipeline Profile")
    print("==========================")

    for stage in stage_names:
        stats = stages[stage]

        print(
            f"{stage:34} "
            f"median={stats['median']:8.3f} ms  "
            f"p95={stats['p95']:8.3f} ms"
        )

    print(f"\nProcessing-limited FPS estimate: {report['processing_limited_fps_from_median_ms']}")

    print(f"30 FPS frame budget: {frame_budget_ms:.2f} ms")

    print(f"\nSaved report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()