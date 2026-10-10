

from pathlib import Path
import argparse

import yaml

DATASET_ROOT = (Path(__file__).resolve().parents[1] / "datasets" / "rover_obstacles")

def validate_label_file(label_path, class_count):
    # Validata one YOLO formatted file

    errors = []

    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()

        if len(parts) != 5:
            errors.append(f"{label_path}: {line_number}: " "Expected 5 values")
            continue

        try:
            class_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            errors.append(f"{label_path}: {line_number}: " "Invalid number format")
            continue

        if not 0 <= class_id < class_count:
            errors.append(f"{label_path}: {line_number}: " "Invalid class ID {class_id}")

        valid_bbox = (
            0 < w <= 1
            and 0 < h <= 1
            and 0 <= x - w / 2 <= 1
            and 0 <= x + w / 2 <= 1
            and 0 <= y - h / 2 <= 1
            and 0 <= y + h / 2 <= 1
        )

        if not valid_bbox:
            errors.append(f"{label_path}: {line_number}: " "Bounding box outside image")

    return errors

def validate_dataset(root, require_images=False):
    # Validate dataset layout and annotations

    root = Path(root)
    errors = []
    counts = {}

    config_path = root / "data.yaml"

    if not config_path.is_file():
        return ["Missing data.yaml file"], counts

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if not isinstance(config, dict):
        return ["Invalid data.yaml format"], counts

    names = config.get("names", {})
    class_count = len(names)

    if (not isinstance(names, dict) 
        or list(names.keys()) != list(range(class_count)) 
        or not all(isinstance(v, str) for v in names.values()) 
        or class_count == 0 
        or config.get("nc") != class_count
    ):

        return ["Invalid class configuration"], counts

    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split

        if not image_dir.is_dir():
            errors.append(f"Missing image directory: {image_dir}")
            continue
        
        if not label_dir.is_dir():
            errors.append(f"Missing label directory: {label_dir}")
            continue

        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})

        counts[split] = len(images)

        if require_images and not images:
            errors.append(f"No images found in {image_dir}")

        image_stems = {image.stem for image in images}

        if len(image_stems) != len(images):
            errors.append(f"Duplicate image stems found in {image_dir}")

        for image in images:
            label_path = label_dir / (image.stem + ".txt")
            
            if not label_path.is_file():
                errors.append(f"Missing label file for image: {label_path}")
                continue

            errors.extend(validate_label_file(label_path, class_count))

        for label_path in label_dir.glob("*.txt"):
            if label_path.stem not in image_stems:
                errors.append(f"Orphan label file without corresponding image: {label_path}")

    return errors, counts

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=Path,default=DATASET_ROOT)

    parser.add_argument("--require-images", action="store_true")

    args = parser.parse_args()

    errors, counts = validate_dataset(args.root, require_images=args.require_images)

    print("EdgeSight Dataset Validation")
    print("----------------------------")

    for split, count in counts.items():
        print(f"{split}: {count} images")

    if errors:
        print(f"\nFound {len(errors)} issue(s):")

        for error in errors:
            print(f"ERROR: {error}")

        raise SystemExit(1)

    print("\nDataset structure and labels validated!")


if __name__ == "__main__":
    main()


    