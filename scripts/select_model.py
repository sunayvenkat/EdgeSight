



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

# Files that make up a release; --archive moves them to models/archive/<release>/
RELEASE_FILES = ("selected.pt", "selected.onnx", "selection.json", "onnx_manifest.json", "runtime_config.json")


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def choose_best(reports):
    # Select a model using validation metrics only; on a tie, the earlier candidate wins

    reports = list(reports.items())
    first = reports[0][1]

    for _, report in reports:
        if (
            report["split"] != "val"
            or report["images"] != first["images"]
            or report["images"] < 1
            or report.get("dataset") != first.get("dataset")
            or report.get("imgsz") != first.get("imgsz")
        ):
            raise ValueError(
                "Models must use the same nonempty "
                "validation split, dataset, and image size."
            )

    best_index = max(range(len(reports)), key=lambda index: (reports[index][1]["mAP50_95"], reports[index][1]["recall"], -index))

    return reports[best_index][0]


def choose_candidate(v1, v2):
    # The original pilot_v1 vs pilot_v2_data selection
    return choose_best({"pilot_v1": v1, "pilot_v2_data": v2})


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


def archive_release(models_root=MODELS_ROOT):
    # Move the current release files to models/archive/<release>/ so a new release never overwrites them
    selection_path = models_root / "selection.json"

    if not selection_path.is_file():
        raise FileNotFoundError("No current release to archive (models/selection.json is missing)")

    release = json.loads(selection_path.read_text(encoding="utf-8"))["release"]
    archive_dir = models_root / "archive" / release

    if archive_dir.exists():
        raise FileExistsError(f"{archive_dir} already exists")

    archive_dir.mkdir(parents=True)

    moved = []

    for name in RELEASE_FILES:
        if (models_root / name).exists():
            shutil.move(str(models_root / name), str(archive_dir / name))
            moved.append(name)

    return archive_dir, moved


def verify_splits(dataset_root, lock_path):
    if not lock_path.is_file():
        raise FileNotFoundError("Create the dataset split lock first.")

    current = build_manifest(dataset_root)
    locked = json.loads(lock_path.read_text(encoding="utf-8"))

    if not locked_splits_match(current, locked):
        raise ValueError("Evaluation data changed")

    if find_cross_split_duplicates(current):
        raise ValueError("Cross-split duplicates found")


def release_model(release, candidates, reports, dataset_root, lock_path, archive):
    # Copy the winning checkpoint to models/selected.pt and record why it won
    verify_splits(dataset_root, lock_path)

    winner = choose_best(reports)
    source = candidates[winner]
    model = YOLO(str(source))

    if model.task != "detect":
        raise ValueError("Expected a detection model")

    if not valid_class_names(model.names):
        raise ValueError(f"Incorrect model classes: {model.names}")

    MODELS_ROOT.mkdir(parents=True, exist_ok=True)

    if archive:
        archive_dir, moved = archive_release()
        print(f"Archived {', '.join(moved)} to {archive_dir}")

        # The winner may be the release just archived (e.g. v1 kept its place)
        if source.parent == MODELS_ROOT.resolve() and source.name in moved:
            source = archive_dir / source.name

    destination = MODELS_ROOT / "selected.pt"
    manifest_path = MODELS_ROOT / "selection.json"

    if destination.exists() or manifest_path.exists():
        raise FileExistsError(
            "A selected model already exists. "
            "Review the release before replacing it, or pass --archive."
        )

    shutil.copy2(source, destination)

    selected_report = reports[winner]

    manifest = {
        "release": release,
        "selected_experiment": winner,
        "classes": list(EXPECTED_CLASSES),
        "checkpoint_sha256": sha256_file(destination),
        "split_lock_sha256": sha256_file(lock_path),
        "selection_metric": "validation_mAP50_95",
        "validation": {
            "images": selected_report["images"],
            "precision": selected_report["precision"],
            "recall": selected_report["recall"],
            "mAP50": selected_report["mAP50"],
            "mAP50_95": selected_report["mAP50_95"],
        },
    }

    if len(reports) > 2 or selected_report.get("dataset") is not None:
        manifest["dataset"] = selected_report.get("dataset")
        manifest["imgsz"] = selected_report.get("imgsz")
        manifest["candidates"] = {name: {"mAP50_95": report["mAP50_95"], "recall": report["recall"]} for name, report in reports.items()}

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Selected model: {winner}")
    print(f"Saved checkpoint: {destination}")
    print(f"Saved manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Select and release the best candidate model by validation mAP50-95.")

    parser.add_argument("--v1-weights", help="Original v1 release: pilot_v1 checkpoint")
    parser.add_argument("--v2-weights", help="Original v1 release: pilot_v2_data checkpoint")
    parser.add_argument("--candidate", nargs=3, action="append", metavar=("NAME", "WEIGHTS", "SUMMARY"), help="A candidate: name, checkpoint, and evaluation summary.json (repeat for each)")
    parser.add_argument("--release", default="edgesight-ml-v1", help="Release name recorded in selection.json")
    parser.add_argument("--dataset", type=Path, default=DATASET_ROOT)
    parser.add_argument("--lock", type=Path, default=LOCK_PATH, help="Split lock for --dataset")
    parser.add_argument("--archive", action="store_true", help="Move the current release to models/archive/<release>/ first")

    args = parser.parse_args()

    if args.candidate:
        if args.v1_weights or args.v2_weights:
            parser.error("Use either --candidate or --v1-weights/--v2-weights, not both")

        if len(args.candidate) < 2:
            parser.error("Give at least two --candidate entries")

        candidates = {name: resolve_weights(weights) for name, weights, _ in args.candidate}
        reports = {name: json.loads(Path(summary).read_text(encoding="utf-8")) for name, _, summary in args.candidate}

        dataset = args.dataset if args.dataset.is_absolute() else PROJECT_ROOT / args.dataset
        lock = args.lock if args.lock.is_absolute() else PROJECT_ROOT / args.lock

        release_model(args.release, candidates, reports, dataset, lock, args.archive)
        return

    if not (args.v1_weights and args.v2_weights):
        parser.error("Give --candidate entries, or both --v1-weights and --v2-weights")

    v1_report = RUNS_ROOT / ("evaluation/pilot_v1_commonval/summary.json")

    v2_report = RUNS_ROOT / ("evaluation/pilot_v2_commonval/summary.json")

    reports = {
        "pilot_v1": json.loads(v1_report.read_text(encoding="utf-8")),
        "pilot_v2_data": json.loads(v2_report.read_text(encoding="utf-8")),
    }

    candidates = {"pilot_v1": resolve_weights(args.v1_weights), "pilot_v2_data": resolve_weights(args.v2_weights)}

    release_model(args.release, candidates, reports, DATASET_ROOT, LOCK_PATH, args.archive)


if __name__ == "__main__":
    main()
