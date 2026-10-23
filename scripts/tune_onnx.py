


import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import psutil

from ultralytics.data.augment import LetterBox

from scripts.benchmark_inference import (MODEL_PATH, IMAGE_DIR, OUTPUT_ROOT, load_images, percentile, sha256_file, verify_model)


ONNX_PATH = MODEL_PATH.with_suffix(".onnx")
MANIFEST_PATH = MODEL_PATH.parent / "onnx_manifest.json"

IMAGE_SIZE = 640

PRESETS = {
    "default": (0, "all", True),
    "intra1": (1, "all", True),
    "intra2": (2, "all", True),
    "intra4": (4, "all", True),
    "intra4_nospin": (4, "all", False),
    "intra4_basic": (4, "basic", True),
}


def tuning_file_name(preset, run, tag=None):
    """Result file name; untagged names are the original v1 tuning runs."""
    prefix = "tuning_" if tag is None else f"tuning_{tag}_"

    return f"{prefix}{preset}_run{run}.json"


def make_options(preset):
    """Create explicitly configured ONNX Runtime options."""

    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}")

    threads, optimization, spinning = PRESETS[preset]

    options = ort.SessionOptions()

    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = (ort.ExecutionMode.ORT_SEQUENTIAL)

    levels = {"all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL, "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,}

    options.graph_optimization_level = (levels[optimization])

    options.add_session_config_entry("session.intra_op.allow_spinning", "1" if spinning else "0")

    return options


def prepare_input(frame, image_size=IMAGE_SIZE):
    """Apply YOLO letterbox and normalize to FP32 NCHW."""

    letterbox = LetterBox(new_shape=(image_size, image_size), auto=False)

    image = letterbox(image=frame)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)

    return tensor / np.float32(255.0)


def compare_outputs(reference, candidate):
    """Check numerical consistency of raw model outputs."""

    if len(reference) != len(candidate):
        return {"allclose": False, "max_abs_error": None}

    maximum_error = 0.0
    equivalent = True

    for left, right in zip(reference, candidate):
        if left.shape != right.shape:
            return {"allclose": False, "max_abs_error": None}

        close = np.allclose(left, right, rtol=1e-3, atol=1e-4)

        equivalent = equivalent and bool(close)

        if left.size:
            error = float(np.max(np.abs(left - right)))
            maximum_error = max(maximum_error, error)

    return {"allclose": equivalent, "max_abs_error": maximum_error}


def verify_export():
    """Verify the ONNX file matches the released model."""

    source_hash = verify_model()

    if not ONNX_PATH.is_file():
        raise FileNotFoundError(ONNX_PATH)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    if manifest["source_sha256"] != source_hash:
        raise ValueError("ONNX source checkpoint mismatch")

    if (manifest["onnx_sha256"] != sha256_file(ONNX_PATH)):
        raise ValueError("ONNX file hash mismatch")

    if manifest["input_shape"] != [1, 3, 640, 640]:
        raise ValueError("Unexpected ONNX input shape")

    return source_hash


def benchmark_session(session, tensors, warmup,iterations):
    """Time direct ONNX Runtime inference calls."""

    input_info = session.get_inputs()[0]

    if (input_info.shape != [1, 3, 640, 640] or input_info.type != "tensor(float)"):
        raise ValueError("Expected FP32 input [1, 3, 640, 640]")

    input_name = input_info.name

    for index in range(warmup):
        tensor = tensors[index % len(tensors)]

        session.run(None, {input_name: tensor})

    process = psutil.Process()

    samples = [process.memory_info().rss]
    latencies = []

    cpu_before = process.cpu_times()
    wall_start = time.perf_counter()

    for index in range(iterations):
        tensor = tensors[index % len(tensors)]

        start = time.perf_counter()

        session.run(None, {input_name: tensor})

        elapsed_ms = (time.perf_counter() - start) * 1000

        latencies.append(elapsed_ms)
        samples.append(process.memory_info().rss)

    wall_seconds = time.perf_counter() - wall_start
    cpu_after = process.cpu_times()

    cpu_seconds = ((cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system))

    return {
        "mean_ms": round(statistics.mean(latencies), 3),
        "median_ms": round(statistics.median(latencies), 3),
        "p95_ms": round(percentile(latencies, 95), 3),
        "throughput_fps": round(iterations / wall_seconds, 3),
        "process_cpu_percent": round(cpu_seconds / wall_seconds * 100, 2),
        "peak_sampled_rss_mb": round(max(samples) / (1024 ** 2), 2),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--preset", choices=PRESETS.keys(), required=True)
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--tag", default=None, help="Prefix for a new release's result files, e.g. v2 (default: v1 file names)")

    args = parser.parse_args()

    if (args.run < 0 or args.warmup < 0 or args.iterations < 2):
        parser.error("Invalid benchmark parameters")

    output_path = OUTPUT_ROOT / tuning_file_name(args.preset, args.run, args.tag)

    if output_path.exists():
        raise FileExistsError(output_path)

    source_hash = verify_export()
    images = load_images(IMAGE_DIR)

    tensors = [prepare_input(frame) for _, frame in images]

    # Compare raw output with an untuned reference.
    reference_session = ort.InferenceSession(str(ONNX_PATH), sess_options=make_options("default"), providers=["CPUExecutionProvider"])

    input_name = reference_session.get_inputs()[0].name

    reference = reference_session.run(None, {input_name: tensors[0]})

    del reference_session

    print(f"Preset: {args.preset}")
    print(f"Images: {len(images)}")

    options = make_options(args.preset)

    start_init = time.perf_counter()

    session = ort.InferenceSession(str(ONNX_PATH), sess_options=options, providers=["CPUExecutionProvider"])

    startup_seconds = time.perf_counter() - start_init

    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("Unexpected execution provider")

    candidate = session.run(None, {session.get_inputs()[0].name: tensors[0]})

    consistency = compare_outputs(reference, candidate)

    print("Raw outputs consistent:", consistency["allclose"])

    if not consistency["allclose"]:
        raise RuntimeError(
            "Numerical output mismatch. "
            "Investigate before benchmarking."
        )

    metrics = benchmark_session(session, tensors, args.warmup, args.iterations)

    threads, graph, spinning = PRESETS[args.preset]

    report = {
        "preset": args.preset,
        "run": args.run,
        "backend": "Direct ONNX Runtime",
        "model_sha256": source_hash,
        "onnx_sha256": sha256_file(ONNX_PATH),
        "image_hashes": {
            path.name: sha256_file(path)
            for path, _ in images
        },
        "configuration": {
            "intra_op_threads": threads,
            "inter_op_threads": 1,
            "execution_mode": "sequential",
            "graph_optimization": graph,
            "spinning": spinning,
            "batch": 1,
            "imgsz": IMAGE_SIZE,
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "environment": {
            "processor": platform.processor(),
            "platform": platform.platform(),
            "onnxruntime": ort.__version__,
            "logical_cpus": psutil.cpu_count(),
        },
        "session_initialization_seconds": round(
            startup_seconds, 3
        ),
        "consistency": consistency,
        "metrics": metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nResults")
    print("-----------------------")
    print(f"Median: {metrics['median_ms']:.2f} ms")
    print(f"P95: {metrics['p95_ms']:.2f} ms")
    print(f"FPS: {metrics['throughput_fps']:.2f}")
    print(f"CPU: {metrics['process_cpu_percent']:.1f}%")
    print(f"RSS: {metrics['peak_sampled_rss_mb']:.1f} MB")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
