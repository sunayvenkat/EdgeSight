

"""Compare training settings with cross-validation by location (grouped k-fold) on real footage."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import statistics
import time
import traceback
from pathlib import Path

import yaml

from scripts.build_dataset import CLASS_NAMES, IMAGE_EXTENSIONS, clip_label_text, known_test_clip_hashes, read_cvat_export, remove_label_lines
from scripts.evaluate_model import build_report
from scripts.extract_frames import MANIFEST_NAME, RAW_ROOT
from scripts.validate_dataset import validate_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CV_ROOT = PROJECT_ROOT / "runs" / "cv"

DEFAULT_PLAN = PROJECT_ROOT / "datasets" / "cv_plan.json"

# Settings shared by every run; matches scripts/train_model.py apart from the per-config values below
COMMON = {"imgsz": 640, "batch": 4, "optimizer": "AdamW", "lr0": 0.001, "seed": 42, "workers": 0, "mixup": 0.0, "copy_paste": 0.0, "device": "cpu"}

# Run in this order, so partial results are still a useful comparison
CONFIGS = {
    "baseline": {"stock": True, "train": {"epochs": 50, "patience": 10, "mosaic": 0.5}},
    "long": {"stock": True, "train": {"epochs": 100, "patience": 30, "mosaic": 1.0}},
    "long_aug": {"stock": True, "train": {"epochs": 100, "patience": 30, "mosaic": 1.0, "scale": 0.7, "hsv_v": 0.5, "translate": 0.15}},
    "long_nostock": {"stock": False, "train": {"epochs": 100, "patience": 30, "mosaic": 1.0}},
}

BACKGROUND_CONFIDENCE = 0.40


def resolve(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def group_for(frame, groups):
    """Return the one group whose clip and frame ranges contain this frame; error on zero or several."""
    matches = []

    for group in groups:
        if group["clip"] != frame["clip"]:
            continue

        ranges = group.get("ranges") or [[0, None]]

        if any(start <= frame["frame_index"] and (end is None or frame["frame_index"] < end) for start, end in ranges):
            matches.append(group["group"])

    if len(matches) != 1:
        raise ValueError(f"{frame['file']} matches {len(matches)} groups {matches}; each labeled frame needs exactly one")

    return matches[0]


def assign_groups(plan):
    """Every labeled real-world frame with its group and final label text."""
    test_hashes = known_test_clip_hashes()
    entries = []

    for session in plan["sessions"]:
        session_dir = RAW_ROOT / session["session"]
        manifest = json.loads((session_dir / MANIFEST_NAME).read_text(encoding="utf-8"))

        for clip, info in manifest["clips"].items():
            if info["sha256"] in test_hashes:
                raise ValueError(f"Clip {clip} is a test clip and cannot be used for cross-validation")

        labels = remove_label_lines(read_cvat_export(resolve(session["labels"])), session.get("remove_boxes", {}))
        labels = {stem: clip_label_text(text)[0] for stem, text in labels.items()}
        empty_frames = set(session.get("empty_frames", []))

        for frame in manifest["frames"]:
            stem = Path(frame["file"]).stem

            if any(fnmatch.fnmatch(frame["clip"], pattern) for pattern in plan.get("background_clips", [])) or stem in empty_frames:
                label = ""
            elif stem in labels:
                label = labels[stem]
            else:
                raise ValueError(f"No label for {frame['file']}")

            entries.append({"path": session_dir / frame["file"], "group": group_for(frame, session["groups"]), "label": label})

    return entries


def split_fold(entries, fold, reserved):
    """Training and validation entries for one fold; reserved groups are left out of both."""
    known = {entry["group"] for entry in entries}
    missing = sorted(set(fold["val_groups"]) - known)

    if missing:
        raise ValueError(f"{fold['name']}: unknown validation groups {missing}")

    val = [entry for entry in entries if entry["group"] in fold["val_groups"]]
    train = [entry for entry in entries if entry["group"] not in fold["val_groups"] and entry["group"] not in reserved]

    return train, val


def build_fold_dataset(root: Path, train, val, stock_root: Path | None):
    """Write a YOLO dataset for one fold; returns the data.yaml path."""
    if root.exists():
        shutil.rmtree(root)

    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)

    if stock_root is not None:
        for image in sorted((stock_root / "images" / "train").iterdir()):
            if image.suffix.lower() in IMAGE_EXTENSIONS:
                shutil.copy2(image, root / "images" / "train" / image.name)
                shutil.copy2(stock_root / "labels" / "train" / f"{image.stem}.txt", root / "labels" / "train" / f"{image.stem}.txt")

    for split, entries in (("train", train), ("val", val)):
        for entry in entries:
            shutil.copy2(entry["path"], root / "images" / split / entry["path"].name)
            (root / "labels" / split / f"{entry['path'].stem}.txt").write_text(entry["label"], encoding="utf-8")

    data_yaml = {"path": root.as_posix(), "train": "images/train", "val": "images/val", "test": "images/test", "nc": len(CLASS_NAMES), "names": CLASS_NAMES}
    (root / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    errors, _ = validate_dataset(root)

    if errors:
        raise ValueError("Fold dataset invalid:\n" + "\n".join(errors[:10]))

    return root / "data.yaml"


def best_epoch(results_csv: Path):
    import csv

    rows = [{key.strip(): value for key, value in row.items()} for row in csv.DictReader(open(results_csv))]
    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))

    return int(float(best["epoch"])), len(rows)


def background_fp_rate(model, images):
    """Share of background (no-object) validation images with any detection at the app's confidence."""
    if not images:
        return None

    flagged = 0

    for result in model.predict(source=[str(path) for path in images], conf=BACKGROUND_CONFIDENCE, imgsz=640, device="cpu", verbose=False, stream=True):
        flagged += int(result.boxes is not None and len(result.boxes) > 0)

    return round(flagged / len(images), 4)


def run_label(config_name, seed):
    """Results folder name: the config, plus the seed when it is not the default 42."""
    return config_name if seed == COMMON["seed"] else f"{config_name}_seed{seed}"


def run_one(config_name, config, fold, data_yaml, val_entries, seed=COMMON["seed"]):
    from ultralytics import YOLO

    label = run_label(config_name, seed)
    run_dir = CV_ROOT / label / fold["name"]
    result_path = run_dir / "cv_result.json"

    if result_path.is_file():
        print(f"[skip] {label}/{fold['name']} already done")
        return json.loads(result_path.read_text(encoding="utf-8"))

    if run_dir.exists():
        shutil.rmtree(run_dir)

    settings = {**COMMON, "seed": seed, **config["train"]}
    start = time.time()

    YOLO(str(PROJECT_ROOT / "yolo26n.pt")).train(data=str(data_yaml), project=str(CV_ROOT / label), name=fold["name"], exist_ok=False, plots=False, save=True, **settings)

    train_seconds = round(time.time() - start)
    epoch, epochs_run = best_epoch(run_dir / "results.csv")
    backgrounds = [entry["path"] for entry in val_entries if not entry["label"].strip()]

    result = {"config": label, "base_config": config_name, "seed": seed, "fold": fold["name"], "settings": {**settings, "stock": config["stock"]}, "best_epoch": epoch, "epochs_run": epochs_run, "train_seconds": train_seconds, "val_images": len(val_entries), "val_backgrounds": len(backgrounds)}

    for which in ("best", "last"):
        weights = run_dir / "weights" / f"{which}.pt"
        model = YOLO(str(weights))
        metrics = model.val(data=str(data_yaml), split="val", imgsz=640, batch=4, conf=0.001, iou=0.7, device="cpu", workers=0, plots=False, project=str(run_dir), name=f"val_{which}", exist_ok=True, verbose=False)
        report = build_report(metrics, weights, len(val_entries), split="val", dataset=f"cv:{fold['name']}", imgsz=640)
        report["background_fp_rate"] = background_fp_rate(model, backgrounds)
        result[which] = report

    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


def summarize(results):
    """Average each config's (and seed variant's) validation metrics over its completed folds."""
    summary = {}
    labels = list(dict.fromkeys(result.get("config") for result in results if result.get("config")))

    for config_name in labels:
        runs = [result for result in results if result.get("config") == config_name and "best" in result]

        if not runs:
            continue

        def mean(values):
            values = [value for value in values if value is not None]
            return round(statistics.mean(values), 4) if values else None

        per_class = {name: mean([next((item["mAP50"] for item in run["best"]["per_class"] if item["class_name"] == name), None) for run in runs]) for name in CLASS_NAMES.values()}

        summary[config_name] = {
            "folds": len(runs),
            "mAP50_95": mean([run["best"]["mAP50_95"] for run in runs]),
            "mAP50": mean([run["best"]["mAP50"] for run in runs]),
            "recall": mean([run["best"]["recall"] for run in runs]),
            "per_class_mAP50": per_class,
            "background_fp_rate": mean([run["best"]["background_fp_rate"] for run in runs]),
            "last_mAP50_95": mean([run["last"]["mAP50_95"] for run in runs]),
            "best_epochs": [run["best_epoch"] for run in runs],
            "epochs_run": [run["epochs_run"] for run in runs],
            "train_minutes": round(sum(run["train_seconds"] for run in runs) / 60, 1),
        }

    return summary


def write_summary(results):
    # Combine every completed run on disk (all seeds and invocations) with this invocation's errors
    completed = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(CV_ROOT.glob("*/*/cv_result.json"))]
    order = {name: index for index, name in enumerate(CONFIGS)}
    completed.sort(key=lambda result: (order.get(result.get("base_config", result["config"]), 99), result.get("seed", COMMON["seed"]), result["fold"]))
    results = completed + [result for result in results if "error" in result]

    summary = summarize(results)
    (CV_ROOT / "summary.json").write_text(json.dumps({"summary": summary, "runs": results}, indent=2, default=str), encoding="utf-8")

    lines = ["| Config | Folds | mAP50-95 | mAP50 | Recall | Rock | Cone | Debris | Background FP | last.pt mAP50-95 | Best epochs |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]

    for name, row in summary.items():
        pc = row["per_class_mAP50"]
        lines.append(f"| {name} | {row['folds']} | {row['mAP50_95']} | {row['mAP50']} | {row['recall']} | {pc['rock']} | {pc['cone']} | {pc['debris']} | {row['background_fp_rate']} | {row['last_mAP50_95']} | {row['best_epochs']} of {row['epochs_run']} |")

    (CV_ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Cross-validate training settings by location (see datasets/cv_plan.json).")

    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--stage-dir", type=Path, default=Path(os.environ.get("TEMP", "/tmp")) / "edgesight_cv", help="Where fold datasets are built (outside OneDrive)")
    parser.add_argument("--seed", type=int, default=COMMON["seed"], help="Training seed; non-default seeds get their own results folders (e.g. long_seed7)")

    args = parser.parse_args()

    plan = json.loads(resolve(args.plan).read_text(encoding="utf-8"))
    entries = assign_groups(plan)
    reserved = set(plan.get("reserved_groups", []))
    stock_root = resolve(plan["base"]["dataset"])

    CV_ROOT.mkdir(parents=True, exist_ok=True)

    counts = {}

    for entry in entries:
        counts[entry["group"]] = counts.get(entry["group"], 0) + 1

    print("Frames per group:", counts, "| reserved:", sorted(reserved))

    folds = {}

    for fold in plan["folds"]:
        train, val = split_fold(entries, fold, reserved)
        folds[fold["name"]] = (fold, train, val)
        print(f"{fold['name']}: {len(train)} real training frames, {len(val)} validation frames ({sum(1 for e in val if not e['label'].strip())} background)")

    datasets = {}
    results = []

    for config_name in args.configs:
        config = CONFIGS[config_name]

        for fold_name, (fold, train, val) in folds.items():
            key = (fold_name, config["stock"])

            try:
                if key not in datasets:
                    datasets[key] = build_fold_dataset(args.stage_dir / f"{fold_name}_{'stock' if config['stock'] else 'nostock'}", train, val, stock_root if config["stock"] else None)

                print(f"\n=== {run_label(config_name, args.seed)} / {fold_name} ===", flush=True)
                result = run_one(config_name, config, fold, datasets[key], val, seed=args.seed)
                print(f"best epoch {result['best_epoch']} of {result['epochs_run']}, val mAP50-95 {result['best']['mAP50_95']}, background FP {result['best']['background_fp_rate']}", flush=True)
            except Exception as error:
                traceback.print_exc()
                result = {"config": config_name, "fold": fold_name, "error": repr(error)}

            results.append(result)
            write_summary(results)

    print("\n" + (CV_ROOT / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
