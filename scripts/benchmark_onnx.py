


import argparse
import json
import platform
from pathlib import Path

import cv2
import onnxruntime as ort
import psutil
import torch
import ultralytics
from ultralytics import YOLO

from scripts.benchmark_inference import (IMAGE_DIR, MODEL_PATH, OUTPUT_ROOT, benchmark_model, load_images, sha256_file, verify_model)


ONNX_PATH = MODEL_PATH.with_suffix(".onnx")

ONNX_MANIFEST = (MODEL_PATH.parent / "onnx_manifest.json")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--warmup", type=int, default=5)

    parser.add_argument("--iterations", type=int, default=60)

    parser.add_argument("--name", default="onnx_cpu_run1")

    args = parser.parse_args()

    if args.warmup < 0 or args.iterations < 2:
        parser.error("Invalid benchmark parameters")

    if not args.name.replace("_", "").replace("-", "").isalnum():
        parser.error("Invalid experiment name")

    output_path = OUTPUT_ROOT / f"{args.name}.json"

    if output_path.exists():
        raise FileExistsError(output_path)

    if not ONNX_PATH.is_file():
        raise FileNotFoundError(ONNX_PATH)

    if not ONNX_MANIFEST.is_file():
        raise FileNotFoundError(ONNX_MANIFEST)

    pytorch_hash = verify_model()

    manifest = json.loads(ONNX_MANIFEST.read_text(encoding="utf-8"))

    if manifest["source_sha256"] != pytorch_hash:
        raise ValueError("ONNX source checkpoint does not match")

    if manifest["onnx_sha256"] != sha256_file(ONNX_PATH):
        raise ValueError("ONNX model hash mismatch")

    # Match the PyTorch-side thread configuration.
    torch.set_num_threads(4)
    cv2.setNumThreads(1)

    images = load_images(IMAGE_DIR)

    print("Loading ONNX Runtime model...")

    model = YOLO(str(ONNX_PATH))

    print(f"Loaded {len(images)} images")
    print("Device: CPU")
    print("ONNX providers:", ort.get_available_providers())

    summary = benchmark_model(
        model=model,
        images=images,
        iterations=args.iterations,
        warmup=args.warmup,
        imgsz=640,
        confidence=0.40,
    )

    report = {
        "experiment": args.name,
        "backend": "ONNX Runtime",
        "device": "CPU",
        "model_sha256": pytorch_hash,
        "onnx_sha256": sha256_file(ONNX_PATH),
        "image_split": "val",
        "images": {
            path.name: sha256_file(path)
            for path, _ in images
        },
        "settings": {
            "imgsz": 640,
            "confidence": 0.40,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "threads": 4,
            "batch": 1,
            "rect": False,
            "onnx_session_threads": "runtime_default",
        },
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": psutil.cpu_count(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "onnxruntime": ort.__version__,
        },
        "metrics": summary,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nONNX Runtime CPU Benchmark")
    print("-----------------------------")
    print(f"Mean: {summary['mean_ms']:.2f} ms")
    print(f"Median: {summary['median_ms']:.2f} ms")
    print(f"P95: {summary['p95_ms']:.2f} ms")
    print(
        f"Throughput: "
        f"{summary['throughput_fps']:.2f} FPS"
    )
    print(
        f"CPU usage: "
        f"{summary['process_cpu_percent']:.1f}%"
    )
    print(
        f"Peak sampled RSS: "
        f"{summary['peak_sampled_rss_mb']:.1f} MB"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
