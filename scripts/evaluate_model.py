


import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from scripts.train_model import (
    PROJECT_ROOT,
    DATASET_ROOT,
    RUNS_ROOT,
    create_training_config,
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


def build_report(metrics, checkpoint, image_count):
    # Convert Ultralytics metrics to a JSON-ready report

    box = metrics.box

    report = {
        "checkpoint": str(checkpoint),
        "split": "val",
        "images": image_count,
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "mAP50": round(float(box.map50), 4),
        "mAP50_95": round(float(box.map), 4),
        "per_class": [],
    }

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


def evaluate_model(checkpoint_path, run_name):
    # Evaluates the trained model on the images

    checkpoint = resolve_checkpoint(checkpoint_path)

    errors, counts = validate_dataset(DATASET_ROOT, require_images=True)

    if errors:
        raise ValueError("Dataset validation failed:\n" + "\n".join(errors))

    output_root = RUNS_ROOT / "evaluation"
    output_dir = output_root / run_name

    if output_dir.exists():
        raise FileExistsError(f"Evaluation already exists: {output_dir}\n" "Choose a new --name to keep results separate.")

    # Reuse the absolute-path configuration.
    config_path = create_training_config(DATASET_ROOT, RUNS_ROOT / "configs" / "rover_training.yaml")

    print("Loading trained model...")
    model = YOLO(str(checkpoint))

    print("Evaluating validation dataset...")

    metrics = model.val(
        data=str(config_path),
        split="val",
        imgsz=512,
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

    report = build_report(metrics, checkpoint, counts["val"])

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

    args = parser.parse_args()

    evaluate_model(checkpoint_path=args.weights, run_name=args.name)


if __name__ == "__main__":
    main()
