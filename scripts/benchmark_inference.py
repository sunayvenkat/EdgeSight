


import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import cv2
import psutil

from edgesight.config import (INFERENCE_IMAGE_SIZE, DETECTION_CONFIDENCE)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "selected.pt"

SELECTION_PATH = (PROJECT_ROOT / "models" / "selection.json")

IMAGE_DIR = (PROJECT_ROOT / "datasets" / "rover_obstacles" / "images" / "val")

OUTPUT_ROOT = PROJECT_ROOT / "runs" / "benchmarks"

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def sha256_file(path):
    """Return the SHA-256 hash of a file."""
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def verify_model():
    """Verify the checkpoint matches its release."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    if not SELECTION_PATH.is_file():
        raise FileNotFoundError(SELECTION_PATH)

    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))

    actual_hash = sha256_file(MODEL_PATH)

    if actual_hash != selection["checkpoint_sha256"]:
        raise ValueError("Selected model differs from release manifest")

    return actual_hash


def load_images(directory):
    """Load fixed image inputs outside timed inference."""

    directory = Path(directory)

    if not directory.is_dir():
        raise FileNotFoundError(directory)

    images = []

    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in EXTENSIONS:
            continue

        frame = cv2.imread(str(path))

        if frame is None:
            raise ValueError(f"Unreadable image: {path}")

        images.append((path, frame))

    if not images:
        raise ValueError("No benchmark images found")

    return images


def percentile(values, percent):
    """Calculate a percentile using interpolation."""

    if not values:
        raise ValueError("Cannot summarize empty data")

    if not 0 <= percent <= 100:
        raise ValueError("Invalid percentile")

    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100

    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower

    return (ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def summarize_latencies(latencies):
    """Summarize repeated prediction times."""

    if not latencies or any(value <= 0 for value in latencies):
        raise ValueError("Latencies must be positive")

    return {
        "count": len(latencies),
        "mean_ms": round(statistics.mean(latencies), 3),
        "median_ms": round(statistics.median(latencies), 3),
        "p95_ms": round(percentile(latencies, 95), 3),
        "min_ms": round(min(latencies), 3),
        "max_ms": round(max(latencies), 3),
    }


def benchmark_model(model, images, iterations, warmup, imgsz, confidence):
    """Benchmark CPU prediction calls on loaded images."""

    process = psutil.Process()

    def predict(frame):
        return model.predict(
            source=frame,
            imgsz=imgsz,
            conf=confidence,
            device="cpu",
            half=False,
            rect=False,
            verbose=False,
        )

    print(f"Warming up: {warmup} predictions")

    for index in range(warmup):
        frame = images[index % len(images)][1]
        predict(frame)

    latencies = []
    stage_times = {
        "preprocess": [],
        "inference": [],
        "postprocess": [],
    }

    memory_samples = [process.memory_info().rss]

    # Prime the process CPU-utilization measurement.
    process.cpu_percent(interval=None)

    loop_start = time.perf_counter()

    for index in range(iterations):
        frame = images[index % len(images)][1]

        start = time.perf_counter()

        results = predict(frame)

        elapsed_ms = (time.perf_counter() - start) * 1000

        latencies.append(elapsed_ms)

        memory_samples.append(process.memory_info().rss)

        for stage in stage_times:
            speed = results[0].speed.get(stage)

            if speed is not None:
                stage_times[stage].append(float(speed))

        if (index + 1) % 10 == 0:
            print(f"Completed {index + 1}/{iterations}")

    loop_seconds = (time.perf_counter() - loop_start)

    cpu_percent = process.cpu_percent(interval=None)

    summary = summarize_latencies(latencies)

    summary["throughput_fps"] = round(iterations / loop_seconds, 3)

    summary["process_cpu_percent"] = round(cpu_percent, 2)

    summary["peak_sampled_rss_mb"] = round(max(memory_samples) / (1024 ** 2), 2)

    summary["stages_mean_ms"] = {
        stage: (
            round(statistics.mean(values), 3)
            if values else None
        )
        for stage, values in stage_times.items()
    }

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=INFERENCE_IMAGE_SIZE)
    parser.add_argument("--confidence", type=float, default=DETECTION_CONFIDENCE)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--name", default="pytorch_cpu_baseline")

    args = parser.parse_args()

    if (
        args.warmup < 0
        or args.iterations < 2
        or args.threads < 1
        or args.imgsz < 32
        or not 0 < args.confidence <= 1
    ):
        parser.error("Invalid benchmark parameters")

    if not args.name.replace("_", "").replace("-", "").isalnum():
        parser.error("Use letters, numbers, - or _ in --name")

    output_path = (OUTPUT_ROOT / f"{args.name}.json")

    if output_path.exists():
        raise FileExistsError(f"Results already exist: {output_path}")

    import torch
    import ultralytics
    from ultralytics import YOLO

    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)

    model_hash = verify_model()
    images = load_images(IMAGE_DIR)

    print("Loading selected model...")
    model = YOLO(str(MODEL_PATH))

    print(f"Images loaded: {len(images)}")
    print("Device: CPU")
    print(f"PyTorch threads: {args.threads}")

    summary = benchmark_model(
        model=model,
        images=images,
        iterations=args.iterations,
        warmup=args.warmup,
        imgsz=args.imgsz,
        confidence=args.confidence,
    )

    report = {
        "experiment": args.name,
        "backend": "PyTorch",
        "device": "CPU",
        "model_sha256": model_hash,
        "image_split": "val",
        "images": {
            path.name: sha256_file(path)
            for path, _ in images
        },
        "settings": {
            "imgsz": args.imgsz,
            "confidence": args.confidence,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "threads": args.threads,
            "batch": 1,
            "rect": False,
        },
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": psutil.cpu_count(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "ultralytics": ultralytics.__version__,
        },
        "metrics": summary,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nEdgeSight CPU Benchmark")
    print("-------------------------")
    print(f"Mean: {summary['mean_ms']:.2f} ms")
    print(f"Median: {summary['median_ms']:.2f} ms")
    print(f"P95: {summary['p95_ms']:.2f} ms")
    print(f"Throughput: {summary['throughput_fps']:.2f} FPS")
    print(f"CPU usage: {summary['process_cpu_percent']:.1f}%")
    print(f"Peak sampled RSS: {summary['peak_sampled_rss_mb']:.1f} MB")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
