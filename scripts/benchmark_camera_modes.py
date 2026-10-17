

"""Compare synchronous and asynchronous EdgeSight camera pipelines."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from edgesight import config
from edgesight.async_camera import AsyncCamera
from edgesight.events import ZoneMonitor
from edgesight.tracker import TrackedObject
from scripts.async_camera_demo import convert_result


MODEL_PATH = Path("models/selected.pt")


def summarize_ms(values: list[float]) -> dict:
    """Summarize millisecond measurements."""
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
        }

    array = np.asarray(values, dtype=float)

    return {
        "count": len(values),
        "mean": round(float(np.mean(array)), 3),
        "median": round(float(np.median(array)), 3),
        "p95": round(float(np.percentile(array, 95)), 3),
        "min": round(float(np.min(array)), 3),
        "max": round(float(np.max(array)), 3),
    }


def process_frame(model, zone_monitor, frame, imgsz) -> float:
    """Run the same tracking and zone-event pipeline in either mode."""
    inference_size = config.INFERENCE_IMAGE_SIZE if imgsz is None else imgsz
    start = time.perf_counter()

    results = model.track(
        source=frame,
        persist=True,
        tracker=config.TRACKER_CONFIG,
        conf=config.DETECTION_CONFIDENCE,
        imgsz=inference_size,
        device="cpu",
        verbose=False,
    )

    objects = convert_result(results[0])
    zone_monitor.update(objects, frame.shape)

    return (time.perf_counter() - start) * 1000.0


def load_model_and_monitor():
    """Create fresh model and tracker state for each benchmark."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))
    zone_monitor = ZoneMonitor()
    return model, zone_monitor


def warmup_sync(capture, model, zone_monitor, seconds: float, imgsz: int | None) -> None:
    """Warm up the synchronous pipeline before timing."""
    deadline = time.monotonic() + seconds

    while time.monotonic() < deadline:
        success, frame = capture.read()

        if not success:
            raise RuntimeError("Synchronous camera failed during warmup")

        process_frame(model, zone_monitor, frame, imgsz)


def run_sync(args, source) -> dict:
    """Benchmark capture followed by inference in one thread."""
    model, zone_monitor = load_model_and_monitor()
    capture = cv2.VideoCapture(source)

    if args.width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open camera source: {source!r}")

    processing_ms = []
    age_at_start_ms = []
    age_at_end_ms = []
    processed = 0

    try:
        warmup_sync(capture, model, zone_monitor, args.warmup, args.imgsz)

        start = time.monotonic()

        while time.monotonic() - start < args.duration:
            success, frame = capture.read()

            if not success:
                raise RuntimeError("Synchronous camera stopped returning frames")

            # Timestamp is taken immediately after OpenCV returns the frame.
            captured_at = time.monotonic()
            inference_start = time.monotonic()

            age_at_start_ms.append((inference_start - captured_at) * 1000.0)

            processing_ms.append(process_frame(model, zone_monitor, frame, args.imgsz))

            finished_at = time.monotonic()
            age_at_end_ms.append((finished_at - captured_at) * 1000.0)
            processed += 1

        elapsed = time.monotonic() - start

    finally:
        capture.release()

    return {
        "mode": "synchronous",
        "elapsed_seconds": round(elapsed, 3),
        "captured_frames": processed,
        "processed_frames": processed,
        "capture_fps": round(processed / elapsed, 3),
        "processed_fps": round(processed / elapsed, 3),
        "dropped_frames": 0,
        "frame_id_skips": 0,
        "processing_ms": summarize_ms(processing_ms),
        "frame_age_at_inference_start_ms": summarize_ms(age_at_start_ms),
        "frame_age_at_processing_end_ms": summarize_ms(age_at_end_ms),
    }


def warmup_async(camera, model, zone_monitor, seconds: float, imgsz: int | None) -> int:
    """Warm up inference while camera capture runs independently."""
    deadline = time.monotonic() + seconds
    last_frame_id = -1

    while time.monotonic() < deadline:
        packet = camera.read_latest(after_frame_id=last_frame_id, timeout=0.25)

        if packet is None:
            if camera.error is not None:
                raise RuntimeError("Camera failed during warmup") from camera.error
            continue

        last_frame_id = packet.frame_id
        process_frame(model, zone_monitor, packet.frame, imgsz)

    # Ignore any frames produced during warmup when timing begins.
    return camera.frames_captured - 1


def run_async(args, source) -> dict:
    """Benchmark background capture with latest-frame inference."""
    model, zone_monitor = load_model_and_monitor()
    camera = AsyncCamera(source=source, width=args.width, height=args.height)

    processing_ms = []
    age_at_start_ms = []
    age_at_end_ms = []
    processed = 0
    frame_id_skips = 0

    camera.start()

    try:
        last_frame_id = warmup_async(
            camera, model, zone_monitor, args.warmup, args.imgsz
        )

        baseline_captured = camera.frames_captured
        baseline_dropped = camera.dropped_frames

        start = time.monotonic()

        while time.monotonic() - start < args.duration:
            packet = camera.read_latest(
                after_frame_id=last_frame_id,
                timeout=0.1,
            )

            if packet is None:
                if camera.error is not None:
                    raise RuntimeError("Asynchronous camera failed") from camera.error
                continue

            if packet.frame_id < baseline_captured:
                last_frame_id = max(last_frame_id, packet.frame_id)
                continue

            frame_id_skips += max(0, packet.frame_id - last_frame_id - 1)
            last_frame_id = packet.frame_id

            inference_start = time.monotonic()
            age_at_start_ms.append((inference_start - packet.captured_at) * 1000.0)

            processing_ms.append(process_frame(model, zone_monitor, packet.frame, args.imgsz))

            finished_at = time.monotonic()
            age_at_end_ms.append((finished_at - packet.captured_at) * 1000.0)
            processed += 1

    finally:
        camera.stop()

    elapsed = time.monotonic() - start
    captured = camera.frames_captured - baseline_captured
    dropped = camera.dropped_frames - baseline_dropped

    return {
        "mode": "asynchronous",
        "elapsed_seconds": round(elapsed, 3),
        "captured_frames": captured,
        "processed_frames": processed,
        "capture_fps": round(captured / elapsed, 3),
        "processed_fps": round(processed / elapsed, 3),
        "dropped_frames": dropped,
        "frame_id_skips": frame_id_skips,
        "processing_ms": summarize_ms(processing_ms),
        "frame_age_at_inference_start_ms": summarize_ms(age_at_start_ms),
        "frame_age_at_processing_end_ms": summarize_ms(age_at_end_ms),
    }


def sha256_file(path: Path) -> str:
    """Calculate a SHA-256 checksum for reproducibility."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def parse_source(value: str) -> int | str:
    """Treat a numeric source as a camera index."""
    return int(value) if value.isdigit() else value


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark synchronous vs asynchronous EdgeSight.")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--warmup", type=float, default=5.0)
    parser.add_argument("--source", default="0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--output", type=Path, default=Path("runs/camera_benchmark/summary.json"))
    parser.add_argument("--imgsz", type=int, default=config.INFERENCE_IMAGE_SIZE, help="YOLO inference image size")
    args = parser.parse_args()

    if args.duration <= 0:
        parser.error("--duration must be greater than zero")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if args.imgsz <= 0:
        parser.error("--imgsz must be greater than zero")

    source = parse_source(args.source)

    print("Benchmarking synchronous mode...")
    synchronous = run_sync(args, source)

    print("Benchmarking asynchronous mode...")
    asynchronous = run_async(args, source)

    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(MODEL_PATH),
        "model_sha256": sha256_file(MODEL_PATH),
        "camera_source": str(source),
        "requested_duration_seconds": args.duration,
        "warmup_seconds_per_mode": args.warmup,
        "resolution_requested": [args.width, args.height],
        "device": "cpu",
        "methodology": (
            "Both modes use the same model, tracker settings, and zone monitor. "
            "Model initialization and warmup are excluded from timed processing. "
            "Frame age begins when OpenCV returns a frame, not at sensor exposure."
        ),
        "results": {
            "synchronous": synchronous,
            "asynchronous": asynchronous,
        },
        "inference_image_size": args.imgsz,
    }

    sync_fps = synchronous["processed_fps"]
    async_fps = asynchronous["processed_fps"]

    if sync_fps > 0:
        output["processed_fps_change_percent"] = round(
            (async_fps - sync_fps) / sync_fps * 100.0,
            2,
        )
    else:
        output["processed_fps_change_percent"] = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print("\nBenchmark results")
    print("-" * 76)
    print(f"{'Metric':<34}{'Synchronous':>20}{'Asynchronous':>20}")
    print("-" * 76)
    print(
        f"{'Captured FPS':<34}"
        f"{synchronous['capture_fps']:>20.2f}"
        f"{asynchronous['capture_fps']:>20.2f}"
    )
    print(
        f"{'Processed FPS':<34}"
        f"{sync_fps:>20.2f}"
        f"{async_fps:>20.2f}"
    )
    print(
        f"{'Processed frames':<34}"
        f"{synchronous['processed_frames']:>20}"
        f"{asynchronous['processed_frames']:>20}"
    )
    print(
        f"{'Dropped/overwritten frames':<34}"
        f"{synchronous['dropped_frames']:>20}"
        f"{asynchronous['dropped_frames']:>20}"
    )
    print(
        f"{'P95 processing time (ms)':<34}"
        f"{synchronous['processing_ms']['p95']:>20.2f}"
        f"{asynchronous['processing_ms']['p95']:>20.2f}"
    )
    print(
        f"{'P95 age at inference start (ms)':<34}"
        f"{synchronous['frame_age_at_inference_start_ms']['p95']:>20.2f}"
        f"{asynchronous['frame_age_at_inference_start_ms']['p95']:>20.2f}"
    )
    print("-" * 76)
    print(
        "Asynchronous processed-FPS change: "
        f"{output['processed_fps_change_percent']}%"
    )
    print(f"JSON report saved to: {args.output}")


if __name__ == "__main__":
    main()
