


import argparse
import hashlib
import json
from pathlib import Path

from scripts.train_model import DATASET_ROOT, PROJECT_ROOT


EXTENSIONS = {".jpg", ".jpeg", ".png"}

LOCK_PATH = (PROJECT_ROOT / "docs" / "split_lock.json")


def file_hash(path):
    # Compute a SHA-256 fingerprint 
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def build_manifest(root):
    # Fingerpring images and matching annotations
    root = Path(root)
    manifest = {}

    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split

        entries = {}

        for image in sorted(image_dir.iterdir()):
            if image.suffix.lower() not in EXTENSIONS:
                continue

            label = label_dir / f"{image.stem}.txt"

            if not label.is_file():
                raise FileNotFoundError(label)

            entries[image.name] = {
                "image": file_hash(image),
                "label": file_hash(label),
            }

        manifest[split] = entries

    return manifest


def find_cross_split_duplicates(manifest):
    # Find identical image bytes in different splits
    seen = {}
    duplicates = []

    for split, entries in manifest.items():
        for filename, hashes in entries.items():
            digest = hashes["image"]

            if digest in seen:
                previous_split, previous_name = seen[digest]

                if previous_split != split:
                    duplicates.append(
                        f"{previous_split}/{previous_name} "
                        f"and {split}/{filename}"
                    )
            else:
                seen[digest] = (split, filename)

    return duplicates


def locked_splits_match(current, locked):
    return all(
        current[split] == locked[split]
        for split in ("val", "test")
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--create-lock", action="store_true")

    args = parser.parse_args()

    manifest = build_manifest(DATASET_ROOT)

    duplicates = find_cross_split_duplicates(manifest)

    if duplicates:
        raise SystemExit("Cross-split duplicates found:\n" + "\n".join(duplicates))

    if args.create_lock:
        if LOCK_PATH.exists():
            raise SystemExit("Split lock already exists. " "Do not overwrite it during an experiment.")

        locked = {
            split: manifest[split]
            for split in ("val", "test")
        }

        LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)

        LOCK_PATH.write_text(json.dumps(locked, indent=2), encoding="utf-8")

        print("Validation and test splits locked.")

    else:
        if not LOCK_PATH.is_file():
            raise SystemExit(
                "No split lock found. "
                "Run --create-lock first."
            )

        locked = json.loads(LOCK_PATH.read_text(encoding="utf-8"))

        if not locked_splits_match(manifest, locked):
            raise SystemExit("ERROR: Validation or test files changed!")

        print("Locked validation/test splits verified.")

    print("No exact cross-split duplicates detected.")


if __name__ == "__main__":
    main()
