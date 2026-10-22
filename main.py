

import argparse
import os
import statistics
import sys
import time
from collections import deque

import cv2

from edgesight import config
from edgesight.async_camera import AsyncCamera
from edgesight.events import ZoneMonitor
from edgesight.tracker import draw_tracks


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run EdgeSight live object detection and tracking.")

    parser.add_argument("--backend", choices=["onnx", "pytorch"], default=config.INFERENCE_BACKEND, help="Inference backend (default: %(default)s)")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="OpenCV camera index (default: %(default)s)")
    parser.add_argument("--threads", type=int, default=None, help="ONNX Runtime intra-op threads (default: edgesight/config.py)")
    parser.add_argument("--headless", action="store_true", help="Run without a video window")
    parser.add_argument("--duration", type=float, default=None, help="Stop after this many seconds")

    args = parser.parse_args(argv)

    if args.threads is not None and args.threads < 0:
        parser.error("--threads cannot be negative")

    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")

    return args


def create_tracker(backend=None, threads=None):
    # Build the detector and tracker for the configured backend
    backend = config.INFERENCE_BACKEND if backend is None else backend

    if backend == "onnx":
        from edgesight.onnx_detector import OnnxDetector, create_session
        from edgesight.tracker import OnnxTracker

        session = create_session(intra_op_threads=config.ONNX_INTRA_OP_THREADS if threads is None else threads)

        return OnnxTracker(OnnxDetector(session=session))

    if backend == "pytorch":
        from edgesight.detector import ObjectDetector
        from edgesight.tracker import ObjectTracker

        return ObjectTracker(ObjectDetector())

    raise ValueError(f"Unknown inference backend: {backend!r}")


def display_available():
    # Linux needs an X11 or Wayland session to open a window (e.g. not over plain SSH)
    if not sys.platform.startswith("linux"):
        return True

    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def draw_overlay(frame, objects, zone, status, metrics):
    # Draw tracked objects, the monitoring zone, and live metrics
    annotated = draw_tracks(frame, objects)

    left, top, right, bottom = zone.bounds(frame.shape)

    cv2.rectangle(annotated, (left, top), (right, bottom), (0, 255, 255), 2)

    for i, text in enumerate(metrics):
        color = (255, 255, 255)

        if i == 3:
            color = ((0, 255, 255) if status == "CAUTION" else (0, 255, 0))

        cv2.putText(annotated, text, (15, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return annotated


def run(tracker, camera, zone, headless=False, duration=None):
    # Process camera frames until quit, duration, or camera failure; return a summary

    # Store the last 30 processed-frame timestamps
    frame_times = deque(maxlen=30)
    processing_ms = []

    start = time.monotonic()

    try:
        process_frames(tracker, camera, zone, headless, duration, start, frame_times, processing_ms)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Exiting...")

    elapsed = time.monotonic() - start

    return {
        "elapsed_seconds": elapsed,
        "processed_frames": len(processing_ms),
        "processed_fps": len(processing_ms) / elapsed if elapsed > 0 else 0.0,
        "median_ms": statistics.median(processing_ms) if processing_ms else None,
        "p95_ms": statistics.quantiles(processing_ms, n=20)[-1] if len(processing_ms) > 1 else None,
        "captured_frames": camera.frames_captured,
        "dropped_frames": camera.dropped_frames,
    }


def process_frames(tracker, camera, zone, headless, duration, start, frame_times, processing_ms):
    # Main loop; appends each frame's inference time to processing_ms
    last_frame_id = -1

    while camera.is_running:
        if duration is not None and time.monotonic() - start >= duration:
            break

        # Wait for a frame newer than the last one processed
        packet = camera.read_latest(after_frame_id=last_frame_id, timeout=1.0)

        if packet is None:
            continue

        last_frame_id = packet.frame_id
        frame = packet.frame

        # Run detection and tracking together
        inference_start = time.perf_counter()
        objects = tracker.track(frame)
        inference_ms = (time.perf_counter() - inference_start) * 1000

        processing_ms.append(inference_ms)

        # Evaluate monitoring zone
        status, events = zone.update(objects, frame.shape)

        for event in events:
            print(f"[ZONE] Track {event.track_id}: " f"{event.action}")

        # Calculate processed FPS
        now = time.perf_counter()
        frame_times.append(now)

        fps = 0.0

        if len(frame_times) > 1:
            elapsed = frame_times[-1] - frame_times[0]

            if elapsed > 0:
                fps = (len(frame_times) - 1) / elapsed

        if headless:
            continue

        metrics = [f"FPS: {fps:.1f}", f"Inference: {inference_ms:.1f} ms", f"Objects: {len(objects)}", f"Status: {status}", f"Dropped: {camera.dropped_frames}"]

        cv2.imshow("EdgeSight - Object Detection", draw_overlay(frame, objects, zone, status, metrics))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def print_summary(summary):
    print(f"Processed {summary['processed_frames']} frames in {summary['elapsed_seconds']:.1f} s ({summary['processed_fps']:.1f} FPS)")

    if summary["median_ms"] is not None and summary["p95_ms"] is not None:
        print(f"Inference + tracking: median {summary['median_ms']:.1f} ms, p95 {summary['p95_ms']:.1f} ms")

    print(f"Camera: {summary['captured_frames']} captured, {summary['dropped_frames']} dropped")


def main(argv=None):
    args = parse_args(argv)

    headless = args.headless

    if not headless and not display_available():
        print("No display found; running headless")
        headless = True

    camera = AsyncCamera(source=args.camera, width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT, fps=config.FPS)
    summary = None

    try:
        print(f"Loading EdgeSight detection model ({args.backend})... ")

        tracker = create_tracker(args.backend, args.threads)
        zone = ZoneMonitor()

        print("EdgeSight detection model loaded successfully")

        try:
            camera.start()
        except (RuntimeError, TimeoutError) as error:
            print(f"Error opening camera: {error}")
            return

        print("Camera opened successfully")
        print("Press Ctrl+C to quit" if headless else "Press 'q' to quit")

        summary = run(tracker, camera, zone, headless=headless, duration=args.duration)

        if camera.error is not None:
            print(f"Camera error: {camera.error}")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Exiting...")

    finally:
        camera.stop()

        if not headless:
            cv2.destroyAllWindows()

        print("Camera resources released")

    if summary is not None:
        print_summary(summary)

if __name__ == "__main__":
    main()
