

import argparse
import os
import statistics
import sys
import time
from collections import deque
from pathlib import Path

import cv2

from edgesight import config
from edgesight.events import ZoneMonitor
from edgesight.recorder import CODECS, RecordingError, VideoRecorder, codec_for
from edgesight.sources import AsyncCamera, VideoFileSource
from edgesight.tracker import draw_tracks


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run EdgeSight live object detection and tracking.")

    parser.add_argument("--backend", choices=["onnx", "pytorch"], default=config.INFERENCE_BACKEND, help="Inference backend (default: %(default)s)")
    parser.add_argument("--model", default=None, metavar="PATH", help="Run a candidate model instead of the released one (.onnx for --backend onnx, .pt for --backend pytorch)")

    # Default None so an explicit --camera always conflicts with --video
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=None, help=f"OpenCV camera index (default: {config.CAMERA_INDEX})")
    source.add_argument("--video", default=None, metavar="PATH", help="Process a recorded video file instead of a camera, frame by frame")

    parser.add_argument("--threads", type=int, default=None, help="ONNX Runtime intra-op threads (default: edgesight/config.py)")
    parser.add_argument("--headless", action="store_true", help="Run without a video window")
    parser.add_argument("--duration", type=float, default=None, help="Stop after this many seconds")
    parser.add_argument("--record", default=None, metavar="PATH", help=f"Save the annotated video to a new file ({' or '.join(CODECS)})")

    args = parser.parse_args(argv)

    if args.threads is not None and args.threads < 0:
        parser.error("--threads cannot be negative")

    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")

    if args.model is not None:
        expected_suffix = ".onnx" if args.backend == "onnx" else ".pt"

        if Path(args.model).suffix.lower() != expected_suffix:
            parser.error(f"--model: the {args.backend} backend needs a {expected_suffix} file")

        if not Path(args.model).is_file():
            parser.error(f"--model: {args.model} not found")

    if args.record is not None:
        # Check the output path before loading the model, so a typo doesn't waste a run
        try:
            codec_for(args.record)
        except ValueError as error:
            parser.error(f"--record: {error}")

        record_path = Path(args.record)

        if record_path.exists():
            parser.error(f"--record: {record_path} already exists; choose a new file name")

        if not record_path.resolve().parent.is_dir():
            parser.error(f"--record: folder {record_path.resolve().parent} does not exist")

    return args


def create_tracker(backend=None, threads=None, model=None):
    # Build the detector and tracker for the configured backend; model overrides the released model
    backend = config.INFERENCE_BACKEND if backend is None else backend

    if backend == "onnx":
        from edgesight.detector import OnnxDetector, create_session, load_class_confidence
        from edgesight.tracker import OnnxTracker

        intra_op_threads = config.ONNX_INTRA_OP_THREADS if threads is None else threads

        # Released cutoffs apply to the release files and INT8 models built from them; other candidates run at the single cutoff
        if model is None:
            session = create_session(intra_op_threads=intra_op_threads)
            class_confidence = load_class_confidence(config.ONNX_MODEL_PATH)
        else:
            session = create_session(model_path=model, intra_op_threads=intra_op_threads, verify_hash=False)
            class_confidence = load_class_confidence(model, required=False)

        return OnnxTracker(OnnxDetector(session=session, class_confidence=class_confidence))

    if backend == "pytorch":
        from edgesight.detector import ObjectDetector, load_class_confidence
        from edgesight.tracker import ObjectTracker

        if model is None:
            return ObjectTracker(ObjectDetector(class_confidence=load_class_confidence(config.MODEL_PATH)))

        return ObjectTracker(ObjectDetector(model_path=model, class_confidence=load_class_confidence(model, required=False)))

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


def run(tracker, camera, zone, headless=False, duration=None, recorder=None):
    # Process camera frames until quit, duration, or camera failure; return a summary

    # Store the last 30 processed-frame timestamps
    frame_times = deque(maxlen=30)
    processing_ms = []

    start = time.monotonic()

    try:
        process_frames(tracker, camera, zone, headless, duration, start, frame_times, processing_ms, recorder)
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


def process_frames(tracker, camera, zone, headless, duration, start, frame_times, processing_ms, recorder=None):
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

        # Headless runs only draw the overlay when recording
        if headless and recorder is None:
            continue

        metrics = [f"FPS: {fps:.1f}", f"Inference: {inference_ms:.1f} ms", f"Objects: {len(objects)}", f"Status: {status}", f"Dropped: {camera.dropped_frames}"]

        annotated = draw_overlay(frame, objects, zone, status, metrics)

        if recorder is not None:
            recorder.write(annotated, packet.captured_at)

        if headless:
            continue

        cv2.imshow("EdgeSight - Object Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def print_summary(summary, source_name="Camera"):
    print(f"Processed {summary['processed_frames']} frames in {summary['elapsed_seconds']:.1f} s ({summary['processed_fps']:.1f} FPS)")

    if summary["median_ms"] is not None and summary["p95_ms"] is not None:
        print(f"Inference + tracking: median {summary['median_ms']:.1f} ms, p95 {summary['p95_ms']:.1f} ms")

    print(f"{source_name}: {summary['captured_frames']} captured, {summary['dropped_frames']} dropped")


def create_source(args, headless):
    # Recorded video (every frame, paced to its FPS when shown) or live camera (latest frame only)
    if args.video is not None:
        return VideoFileSource(args.video, realtime=not headless), "Video"

    camera_index = config.CAMERA_INDEX if args.camera is None else args.camera

    return AsyncCamera(source=camera_index, width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT, fps=config.FPS), "Camera"


def main(argv=None):
    args = parse_args(argv)

    headless = args.headless

    if not headless and not display_available():
        print("No display found; running headless")
        headless = True

    camera, source_name = create_source(args, headless)
    recorder = None
    summary = None

    try:
        print(f"Loading EdgeSight detection model ({args.backend}{', candidate ' + args.model if args.model else ''})... ")

        tracker = create_tracker(args.backend, args.threads, model=args.model)
        zone = ZoneMonitor()

        print("EdgeSight detection model loaded successfully")

        try:
            camera.start()
        except (RuntimeError, TimeoutError) as error:
            print(f"Error opening {source_name.lower()}: {error}")
            return

        if args.video is not None:
            frame_count = camera.reported_frame_count or "unknown"

            print(f"Video opened successfully ({frame_count} frames at {camera.fps:.1f} FPS)")
        else:
            print("Camera opened successfully")

        if args.record is not None:
            # Video files keep their own frame rate; live cameras are re-timed to the processed rate
            fps = camera.fps if args.video is not None else config.FPS
            recorder = VideoRecorder(args.record, fps=fps, retime=args.video is None)

            print(f"Recording to {args.record}")

        print("Press Ctrl+C to quit" if headless else "Press 'q' to quit")

        summary = run(tracker, camera, zone, headless=headless, duration=args.duration, recorder=recorder)

        if camera.error is not None:
            print(f"{source_name} error: {camera.error}")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Exiting...")

    except RecordingError as error:
        print(f"Recording error: {error}")

    finally:
        camera.stop()

        if not headless:
            cv2.destroyAllWindows()

        print(f"{source_name} resources released")

        if recorder is not None:
            playback_fps = recorder.close()

            if playback_fps is None:
                print("Recording empty: no frames were processed")
            else:
                print(f"Recorded {recorder.frames_written} frames to {args.record} ({playback_fps:.1f} FPS playback)")

    if summary is not None:
        print_summary(summary, source_name)

if __name__ == "__main__":
    main()
