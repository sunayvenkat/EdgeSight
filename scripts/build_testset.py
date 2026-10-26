"""Build labeled test datasets (scoring only, never training) from a plan file such as datasets/testset_plan.json."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

from scripts.build_dataset import CLASS_NAMES, SPLITS, clip_label_text, count_boxes, plan_frames, read_cvat_export, resolve
from scripts.extract_frames import MANIFEST_NAME, RAW_ROOT
from scripts.validate_dataset import validate_dataset


def build_testset(testset, background_patterns):
    """Write one test dataset; every frame goes to the test split. Returns the output folder."""
    output = resolve(testset["output"])

    if output.exists():
        raise FileExistsError(f"{output} already exists; test sets are built once")

    session_dir = RAW_ROOT / testset["session"]
    manifest = json.loads((session_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    labels = {stem: clip_label_text(text)[0] for stem, text in read_cvat_export(resolve(testset["labels"])).items()}

    # plan_frames only accepts train and val, so plan as "val" and write everything to test
    entries, _ = plan_frames(manifest, {clip: "val" for clip in testset["clips"]}, background_patterns, labels, set(testset.get("empty_frames", [])), testset.get("exclude_ranges"))

    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)

    for entry in entries:
        shutil.copy2(session_dir / entry["file"], output / "images" / "test" / entry["file"])
        (output / "labels" / "test" / f"{Path(entry['file']).stem}.txt").write_text(entry["label"], encoding="utf-8")

    data_yaml = {"train": "images/train", "val": "images/val", "test": "images/test", "nc": len(CLASS_NAMES), "names": CLASS_NAMES}
    (output / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    errors, _ = validate_dataset(output)

    if errors:
        shutil.rmtree(output)
        raise SystemExit(f"{testset['name']} failed validation:\n" + "\n".join(errors))

    return output


def main():
    parser = argparse.ArgumentParser(description="Build the labeled test sets (see docs/retraining.md).")
    parser.add_argument("--plan", type=Path, default=Path("datasets/testset_plan.json"))
    args = parser.parse_args()

    plan = json.loads(resolve(args.plan).read_text(encoding="utf-8"))

    for testset in plan["testsets"]:
        output = build_testset(testset, plan.get("background_clips", []))
        boxes, backgrounds = count_boxes(output)
        images = len(list((output / "images" / "test").iterdir()))
        print(f"{testset['name']}: {images} images ({backgrounds['test']} background), boxes {dict(boxes['test'])} -> {output}")


if __name__ == "__main__":
    main()
