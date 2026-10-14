


import argparse
import hashlib
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

from scripts.audit_splits import (build_manifest, find_cross_split_duplicates, locked_splits_match)
from scripts.train_model import DATASET_ROOT, PROJECT_ROOT


EXPECTED_CLASSES = ("rock", "cone", "debris")

RUNS_ROOT = PROJECT_ROOT / "runs"
MODELS_ROOT = PROJECT_ROOT / "models"
LOCK_PATH = PROJECT_ROOT / "docs" / "split_lock.json"


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def choose_candidate(v1, v2):
    # Select a model using validation metrics only

    if (
        v1["split"] != "val"
        or v2["split"] != "val"
        or v1["images"] != v2["images"]
        or v1["images"] < 1
    ):
        raise ValueError(
            "Models must use the same nonempty "
            "validation split."
        )

    score_v1 = (v1["mAP50_95"], v1["recall"])
    score_v2 = (v2["mAP50_95"], v2["recall"])

    if score_v2 > score_v1:
        return "pilot_v2_data"

    return "pilot_v1"


def valid_class_names(names):
    # Verify the exact EdgeSight class mapping.

    if isinstance(names, dict):
        if set(names) != set(range(len(EXPECTED_CLASSES))):
            return False

        ordered = [names[i] for i in range(len(names))]
    else:
        ordered = list(names)

    return tuple(ordered) == EXPECTED_CLASSES


def resolve_weights(path):
    path = Path(path).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(path)

    return path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--v1-weights", required=True)
    parser.add_argument("--v2-weights", required=True)

    args = parser.parse_args()

    v1_report = RUNS_ROOT / ("evaluation/pilot_v1_commonval/summary.json")

    v2_report = RUNS_ROOT / ("evaluation/pilot_v2_commonval/summary.json")

    v1 = json.loads(v1_report.read_text(encoding="utf-8"))

    v2 = json.loads(v2_report.read_text(encoding="utf-8"))

    if not LOCK_PATH.is_file():
        raise FileNotFoundError("Create the dataset split lock first.")

    current = build_manifest(DATASET_ROOT)
    locked = json.loads(LOCK_PATH.read_text(encoding="utf-8"))

    if not locked_splits_match(current, locked):
        raise ValueError("Evaluation data changed")

    if find_cross_split_duplicates(current):
        raise ValueError("Cross-split duplicates found")

    winner = choose_candidate(v1, v2)

    weights = {"pilot_v1": resolve_weights(args.v1_weights), "pilot_v2_data": resolve_weights(args.v2_weights)}

    source = weights[winner]
    model = YOLO(str(source))

    if model.task != "detect":
        raise ValueError("Expected a detection model")

    if not valid_class_names(model.names):
        raise ValueError(f"Incorrect model classes: {model.names}")

    MODELS_ROOT.mkdir(parents=True, exist_ok=True)

    destination = MODELS_ROOT / "selected.pt"
    manifest_path = MODELS_ROOT / "selection.json"

    if destination.exists() or manifest_path.exists():
        raise FileExistsError(
            "A selected model already exists. "
            "Review the release before replacing it."
        )

    shutil.copy2(source, destination)

    selected_report = (v2 if winner == "pilot_v2_data" else v1)

    manifest = {
        "release": "edgesight-ml-v1",
        "selected_experiment": winner,
        "classes": list(EXPECTED_CLASSES),
        "checkpoint_sha256": sha256_file(destination),
        "split_lock_sha256": sha256_file(LOCK_PATH),
        "selection_metric": "validation_mAP50_95",
        "validation": {
            "images": selected_report["images"],
            "precision": selected_report["precision"],
            "recall": selected_report["recall"],
            "mAP50": selected_report["mAP50"],
            "mAP50_95": selected_report["mAP50_95"],
        },
    }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Selected model: {winner}")
    print(f"Saved checkpoint: {destination}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
