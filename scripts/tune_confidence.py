"""Choose per-class confidence cutoffs from cross-validation predictions on held-out locations, before any test scoring."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from edgesight import config
from edgesight.detector import sha256_file
from scripts.build_dataset import CLASS_NAMES, PROJECT_ROOT
from scripts.cross_validate import BACKGROUND_CONFIDENCE, CV_ROOT, DEFAULT_PLAN, assign_groups, build_fold_dataset, resolve, run_label, split_fold


def best_cutoffs(curves, px):
    """
    Average each class's F1-vs-confidence curve over runs and return {class: {cutoff, f1, f1_at_default}}.

    curves is a list of {class_id: F1 curve}; a class missing from a run's
    validation labels has no curve for that run.
    """
    result = {}

    for class_id, name in CLASS_NAMES.items():
        runs = [run[class_id] for run in curves if class_id in run]

        if not runs:
            continue

        mean = np.mean(runs, axis=0)
        best = int(np.argmax(mean))
        result[name] = {"cutoff": round(float(px[best]), 3), "f1": round(float(mean[best]), 4), "f1_at_default": round(float(np.interp(BACKGROUND_CONFIDENCE, px, mean)), 4), "runs": len(runs)}

    return result


def write_release(tuning_path, onnx_path=config.ONNX_MODEL_PATH, checkpoint_path=config.MODEL_PATH, output=config.CLASS_CONFIDENCE_PATH):
    """Record the tuned cutoffs with the hashes of the released model files, so the app uses them only with those files."""
    tuning = json.loads(Path(tuning_path).read_text(encoding="utf-8"))
    release = {
        "source": Path(tuning_path).relative_to(PROJECT_ROOT).as_posix() if Path(tuning_path).is_relative_to(PROJECT_ROOT) else Path(tuning_path).as_posix(),
        "onnx_sha256": sha256_file(onnx_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "classes": {name: row["cutoff"] for name, row in tuning["classes"].items()},
    }

    Path(output).write_text(json.dumps(release, indent=2), encoding="utf-8")
    print(f"Saved {output}: {release['classes']}")

    return release


def main():
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description="Per-class confidence cutoffs from cross-validation (see docs/retraining.md).")
    parser.add_argument("--config", default="long_aug")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7])
    parser.add_argument("--stage-dir", type=Path, default=Path(os.environ.get("TEMP", "/tmp")) / "edgesight_cv")
    parser.add_argument("--release", action="store_true", help="Instead of tuning, write the saved cutoffs to models/class_confidence.json for the released model files")
    args = parser.parse_args()

    tuning_path = CV_ROOT / f"confidence_tuning_{args.config}.json"

    if args.release:
        write_release(tuning_path)
        return

    plan = json.loads(DEFAULT_PLAN.read_text(encoding="utf-8"))
    entries = assign_groups(plan)
    reserved = set(plan.get("reserved_groups", []))
    curves = []
    px = None

    for fold in plan["folds"]:
        train, val = split_fold(entries, fold, reserved)
        data_yaml = build_fold_dataset(args.stage_dir / f"{fold['name']}_stock", train, val, resolve(plan["base"]["dataset"]))

        for seed in args.seeds:
            weights = CV_ROOT / run_label(args.config, seed) / fold["name"] / "weights" / "best.pt"
            metrics = YOLO(str(weights)).val(data=str(data_yaml), split="val", imgsz=640, batch=4, conf=0.001, iou=0.7, device="cpu", workers=0, plots=False, project=str(args.stage_dir / "tune_confidence"), name=f"{fold['name']}_{seed}", exist_ok=True, verbose=False)
            px = metrics.box.px
            curves.append({int(class_id): metrics.box.f1_curve[index] for index, class_id in enumerate(metrics.box.ap_class_index)})

    cutoffs = best_cutoffs(curves, px)
    output = tuning_path
    output.write_text(json.dumps({"config": args.config, "seeds": args.seeds, "weights": "best.pt per fold", "default_confidence": BACKGROUND_CONFIDENCE, "classes": cutoffs}, indent=2), encoding="utf-8")

    for name, row in cutoffs.items():
        print(f"{name}: cutoff {row['cutoff']} (F1 {row['f1']}), F1 at {BACKGROUND_CONFIDENCE}: {row['f1_at_default']} over {row['runs']} runs")

    print(f"Saved {output}")


if __name__ == "__main__":
    main()
