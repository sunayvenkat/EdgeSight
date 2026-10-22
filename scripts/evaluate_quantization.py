

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import onnxruntime as ort
import psutil

from edgesight.config import DETECTION_CONFIDENCE, NMS_IOU_THRESHOLD
from edgesight.detector import OnnxDetector, preprocess, sha256_file
from scripts.benchmark_inference import IMAGE_DIR, load_images
from scripts.compare_predictions import match_detections
from scripts.evaluate_model import build_report
from scripts.train_model import DATASET_ROOT, RUNS_ROOT, create_training_config
from scripts.tune_onnx import benchmark_session, make_options


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = RUNS_ROOT / "quantization"

MODELS = {"fp32": PROJECT_ROOT / "models" / "selected.onnx"}

# INT8 outputs of scripts/quantize_onnx.py: activation type (s8 = QInt8, u8 = QUInt8) x variant.
for activation in ("u8", "s8"):
    for variant in ("full", "no_decode", "no_head"):
        MODELS[f"{activation}_{variant}"] = PROJECT_ROOT / "models" / f"selected_int8_{activation}_{variant}.onnx"

# The app's session settings: 4 intra-op threads, 1 inter-op, sequential, all graph optimizations, spinning on.
LATENCY_PRESET = "intra4"

MATCH_IOU = 0.90


def model_path(name):
    path = MODELS[name]

    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing. Run scripts/quantize_onnx.py first.")

    return path


def make_session(path):
    """Build a session with the app's settings, bypassing create_session's hash check."""

    session = ort.InferenceSession(str(path), sess_options=make_options(LATENCY_PRESET), providers=["CPUExecutionProvider"])

    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("Unexpected execution provider")

    return session


def to_records(boxes, scores, class_ids):
    """Convert OnnxDetector.infer output into compare_predictions records."""

    return [
        {"class_id": int(class_id), "confidence": float(score), "bbox": [float(value) for value in box]}
        for box, score, class_id in zip(boxes, scores, class_ids)
    ]


def summarize_consistency(per_image):
    """Total per-image match results."""

    ious = [item["mean_matched_iou"] for item in per_image for _ in range(item["matched"]) if item["mean_matched_iou"] is not None]

    return {
        "images": len(per_image),
        "reference_detections": sum(item["pytorch_count"] for item in per_image),
        "candidate_detections": sum(item["onnx_count"] for item in per_image),
        "matched": sum(item["matched"] for item in per_image),
        "reference_unmatched": sum(item["pytorch_unmatched"] for item in per_image),
        "candidate_unmatched": sum(item["onnx_unmatched"] for item in per_image),
        "mean_matched_iou": round(statistics.mean(ious), 4) if ious else None,
    }


def run_accuracy(name):
    path = model_path(name)
    output_dir = OUTPUT_ROOT / f"accuracy_{name}.json"

    if output_dir.exists():
        raise FileExistsError(output_dir)

    from ultralytics import YOLO

    config_path = create_training_config(DATASET_ROOT, RUNS_ROOT / "configs" / "rover_training.yaml")

    metrics = YOLO(str(path), task="detect").val(
        data=str(config_path),
        split="val",
        imgsz=640,
        batch=1,
        conf=0.001,
        iou=0.7,
        device="cpu",
        workers=0,
        plots=False,
        project=str(OUTPUT_ROOT / "val"),
        name=name,
        exist_ok=False,
    )

    report = build_report(metrics, path, len(load_images(IMAGE_DIR)))
    report["model"] = name
    report["onnx_sha256"] = sha256_file(path)
    report["settings"] = {"imgsz": 640, "batch": 1, "conf": 0.001, "iou": 0.7, "device": "cpu"}

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{name}: P={report['precision']:.4f} R={report['recall']:.4f} mAP50={report['mAP50']:.4f} mAP50-95={report['mAP50_95']:.4f}")


def run_consistency(name):
    if name == "fp32":
        raise ValueError("Compare an INT8 model against fp32")

    output_path = OUTPUT_ROOT / f"consistency_{name}.json"

    if output_path.exists():
        raise FileExistsError(output_path)

    reference = OnnxDetector(session=make_session(model_path("fp32")), confidence=DETECTION_CONFIDENCE, iou_threshold=NMS_IOU_THRESHOLD)
    candidate = OnnxDetector(session=make_session(model_path(name)), confidence=DETECTION_CONFIDENCE, iou_threshold=NMS_IOU_THRESHOLD)

    per_image = []

    for path, frame in load_images(IMAGE_DIR):
        fp32 = to_records(*reference.infer(frame))
        int8 = to_records(*candidate.infer(frame))

        result = match_detections(fp32, int8, threshold=MATCH_IOU)
        result["loose_matched_iou50"] = match_detections(fp32, int8, threshold=0.5)["matched"]
        result["image"] = path.name

        per_image.append(result)

    report = {
        "reference": "fp32",
        "candidate": name,
        "confidence": DETECTION_CONFIDENCE,
        "nms_iou": NMS_IOU_THRESHOLD,
        "match_iou_threshold": MATCH_IOU,
        "note": "pytorch_* fields hold the FP32 reference and onnx_* fields the INT8 candidate (compare_predictions naming).",
        "summary": summarize_consistency(per_image),
        "matched_at_iou50": sum(item["loose_matched_iou50"] for item in per_image),
        "per_image": per_image,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(name, report["summary"], "matched@0.5:", report["matched_at_iou50"])


def run_latency(name, run, warmup, iterations):
    path = model_path(name)
    output_path = OUTPUT_ROOT / f"latency_{name}_run{run}.json"

    if output_path.exists():
        raise FileExistsError(output_path)

    images = load_images(IMAGE_DIR)
    tensors = [preprocess(frame, 640)[0] for _, frame in images]

    start = time.perf_counter()
    session = make_session(path)
    startup = time.perf_counter() - start

    metrics = benchmark_session(session, tensors, warmup, iterations)

    options = make_options(LATENCY_PRESET)

    report = {
        "model": name,
        "run": run,
        "onnx": path.relative_to(PROJECT_ROOT).as_posix(),
        "onnx_sha256": sha256_file(path),
        "file_bytes": path.stat().st_size,
        "image_split": "val",
        "images": len(images),
        "configuration": {
            "intra_op_threads": options.intra_op_num_threads,
            "inter_op_threads": options.inter_op_num_threads,
            "execution_mode": "sequential",
            "graph_optimization": "all",
            "spinning": options.get_session_config_entry("session.intra_op.allow_spinning") == "1",
            "preprocessing": "edgesight.detector.preprocess",
            "warmup": warmup,
            "iterations": iterations,
        },
        "environment": {
            "processor": platform.processor(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "onnxruntime": ort.__version__,
            "logical_cpus": psutil.cpu_count(),
        },
        "session_initialization_seconds": round(startup, 3),
        "metrics": metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{name} run{run}: median {metrics['median_ms']:.2f} ms, p95 {metrics['p95_ms']:.2f} ms, "
          f"{metrics['throughput_fps']:.1f} FPS, CPU {metrics['process_cpu_percent']:.0f}%, RSS {metrics['peak_sampled_rss_mb']:.0f} MB")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--stage", choices=["accuracy", "consistency", "latency"], required=True)
    parser.add_argument("--model", choices=MODELS.keys(), required=True)
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)

    args = parser.parse_args()

    if args.run < 1 or args.warmup < 0 or args.iterations < 2:
        parser.error("Invalid benchmark parameters")

    if args.stage == "accuracy":
        run_accuracy(args.model)
    elif args.stage == "consistency":
        run_consistency(args.model)
    else:
        run_latency(args.model, args.run, args.warmup, args.iterations)


if __name__ == "__main__":
    main()
