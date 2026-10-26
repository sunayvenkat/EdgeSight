


import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from scripts.build_dataset import CLASS_NAMES
from scripts.train_model import (
    PROJECT_ROOT,
    DATASET_ROOT,
    RUNS_ROOT,
    create_training_config,
    training_config_path,
)
from scripts.validate_dataset import validate_dataset


def resolve_checkpoint(checkpoint_path):
    # Resolve and validate the model checkpoint path

    checkpoint = Path(checkpoint_path).expanduser()

    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint

    checkpoint = checkpoint.resolve()

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")

    return checkpoint


def build_report(metrics, checkpoint, image_count, split="val", dataset=None, imgsz=None):
    # Convert Ultralytics metrics to a JSON-ready report

    box = metrics.box

    report = {
        "checkpoint": str(checkpoint),
        "split": split,
        "images": image_count,
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "mAP50": round(float(box.map50), 4),
        "mAP50_95": round(float(box.map), 4),
        "per_class": [],
    }

    # Recorded so model selection can refuse to compare unlike evaluations
    if dataset is not None:
        report["dataset"] = str(dataset)

    if imgsz is not None:
        report["imgsz"] = imgsz

    for index, class_id in enumerate(box.ap_class_index):
        class_id = int(class_id)

        precision, recall, ap50, ap = (box.class_result(index))

        report["per_class"].append({
            "class_id": class_id,
            "class_name": str(metrics.names[class_id]),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "mAP50": round(float(ap50), 4),
            "mAP50_95": round(float(ap), 4),
        })

    return report


def operating_point(box, class_confidence):
    """Per-class precision and recall at the given cutoffs, read from Ultralytics' confidence curves (IoU 0.5)."""
    names = {int(class_id): index for index, class_id in enumerate(box.ap_class_index)}
    result = {}

    for class_id, name in CLASS_NAMES.items():
        if class_id not in names or name not in class_confidence:
            continue

        cutoff = class_confidence[name]
        row = names[class_id]
        result[name] = {"cutoff": cutoff, "precision": round(float(np.interp(cutoff, box.px, box.p_curve[row])), 4), "recall": round(float(np.interp(cutoff, box.px, box.r_curve[row])), 4)}

    return result


def evaluate_model(checkpoint_path, run_name, dataset_root=DATASET_ROOT, imgsz=512, split="val", class_confidence=None):
    # Evaluates the trained model on the images

    checkpoint = resolve_checkpoint(checkpoint_path)
    dataset_root = Path(dataset_root)

    # Only the evaluated split needs images; test sets have empty train and val splits
    errors, counts = validate_dataset(dataset_root)

    if not errors and not counts.get(split):
        errors = [f"No {split} images in {dataset_root}"]

    if errors:
        raise ValueError("Dataset validation failed:\n" + "\n".join(errors))

    output_root = RUNS_ROOT / "evaluation"
    output_dir = output_root / run_name

    if output_dir.exists():
        raise FileExistsError(f"Evaluation already exists: {output_dir}\n" "Choose a new --name to keep results separate.")

    # Reuse the absolute-path configuration.
    config_path = create_training_config(dataset_root, training_config_path(dataset_root))

    print("Loading trained model...")
    model = YOLO(str(checkpoint))

    print(f"Evaluating {split} split of {dataset_root.name} at image size {imgsz}...")

    metrics = model.val(
        data=str(config_path),
        split=split,
        imgsz=imgsz,
        batch=4,
        device="cpu",
        workers=0,
        conf=0.001,
        iou=0.7,
        plots=True,
        project=str(output_root),
        name=run_name,
        exist_ok=False,
    )

    report = build_report(metrics, checkpoint, counts[split], split=split, dataset=dataset_root.name, imgsz=imgsz)

    if class_confidence:
        report["at_class_confidence"] = operating_point(metrics.box, class_confidence)

    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "summary.json"

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8",)

    print("\nEvaluation completed.")
    print(f"Precision: {report['precision']:.3f}")
    print(f"Recall: {report['recall']:.3f}")
    print(f"mAP50: {report['mAP50']:.3f}")
    print(f"mAP50-95: {report['mAP50_95']:.3f}")

    print("\nPer-class metrics:")

    for item in report["per_class"]:
        print(
            f"{item['class_name']}: "
            f"mAP50={item['mAP50']:.3f}, "
            f"mAP50-95={item['mAP50_95']:.3f}"
        )

    print(f"\nResults saved to: {output_dir}")

    return report


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", default="runs/train/pilot_v1/weights/best.pt")

    parser.add_argument("--name", default="pilot_v1_val")
    parser.add_argument("--dataset", type=Path, default=DATASET_ROOT, help="Dataset folder (default: datasets/rover_obstacles, the v1 data)")
    parser.add_argument("--imgsz", type=int, default=512, help="Evaluation image size (default: %(default)s, as used for v1; v2 uses 640)")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--confidence", type=float, default=None, help="Also report per-class precision and recall at this single cutoff")
    parser.add_argument("--class-confidence", type=Path, default=None, help="Also report them at per-class cutoffs from scripts/tune_confidence.py")

    args = parser.parse_args()

    dataset = args.dataset if args.dataset.is_absolute() else PROJECT_ROOT / args.dataset
    class_confidence = None

    if args.class_confidence is not None:
        class_confidence = {name: row["cutoff"] for name, row in json.loads(args.class_confidence.read_text(encoding="utf-8"))["classes"].items()}
    elif args.confidence is not None:
        class_confidence = {name: args.confidence for name in CLASS_NAMES.values()}

    evaluate_model(checkpoint_path=args.weights, run_name=args.name, dataset_root=dataset, imgsz=args.imgsz, split=args.split, class_confidence=class_confidence)


if __name__ == "__main__":
    main()
