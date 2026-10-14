


import json
from pathlib import Path

from ultralytics import YOLO

from scripts.audit_splits import (build_manifest, find_cross_split_duplicates, locked_splits_match)
from scripts.evaluate_model import build_report
from scripts.select_model import (LOCK_PATH, MODELS_ROOT, RUNS_ROOT, sha256_file)
from scripts.train_model import (DATASET_ROOT, create_training_config)
from scripts.validate_dataset import validate_dataset


def main():
    checkpoint = MODELS_ROOT / "selected.pt"
    selection_path = MODELS_ROOT / "selection.json"

    selection = json.loads(selection_path.read_text(encoding="utf-8"))

    if sha256_file(checkpoint) != selection["checkpoint_sha256"]:
        raise ValueError("Model checkpoint differs from manifest")

    if sha256_file(LOCK_PATH) != selection["split_lock_sha256"]:
        raise ValueError("Dataset lock was modified")

    current = build_manifest(DATASET_ROOT)

    locked = json.loads(LOCK_PATH.read_text(encoding="utf-8"))

    if not locked_splits_match(current, locked):
        raise ValueError("Test/validation files changed")

    if find_cross_split_duplicates(current):
        raise ValueError("Cross-split duplicates found")

    errors, counts = validate_dataset(DATASET_ROOT, require_images=True)

    if errors:
        raise ValueError("Dataset validation failed:\n" + "\n".join(errors))

    output_root = RUNS_ROOT / "final_test"
    output_dir = output_root / "selected"

    if output_dir.exists():
        raise FileExistsError(
            "Final test results already exist. "
            "Do not rerun the held-out test to tune the model."
        )

    config_path = create_training_config(DATASET_ROOT, RUNS_ROOT / "configs" / "rover_training.yaml")

    model = YOLO(str(checkpoint))

    metrics = model.val(
        data=str(config_path),
        split="test",
        imgsz=512,
        batch=4,
        device="cpu",
        workers=0,
        conf=0.001,
        iou=0.7,
        plots=True,
        project=str(output_root),
        name="selected",
        exist_ok=False,
    )

    report = build_report(metrics, str(checkpoint), counts["test"])

    report["split"] = "test"
    report["model_sha256"] = sha256_file(checkpoint)

    actual_output = Path(metrics.save_dir)

    (actual_output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nFinal held-out test complete!")
    print(f"Images: {report['images']}")
    print(f"mAP50: {report['mAP50']}")
    print(f"mAP50-95: {report['mAP50_95']}")
    print(f"Report: {actual_output / 'summary.json'}")


if __name__ == "__main__":
    main()
