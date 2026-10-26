

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

from scripts.validate_dataset import validate_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (PROJECT_ROOT / "datasets" / "rover_obstacles")

RUNS_ROOT = PROJECT_ROOT / "runs"

def create_training_config(dataset_root, output_path):
    # Create a YAML file with an absolute dataset path

    dataset_root = Path(dataset_root)
    output_path = Path(output_path)

    source_yaml = dataset_root / "data.yaml"

    with open(source_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    if not isinstance(config, dict):
        raise ValueError(f"Invalid dataset config: {source_yaml}")

    #Ensures YOLO finds images
    config["path"] = dataset_root.resolve().as_posix()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return output_path

def training_config_path(dataset_root):
    # The v1 dataset keeps its original config name; other datasets are named after their folder
    dataset_root = Path(dataset_root)

    if dataset_root.resolve() == DATASET_ROOT.resolve():
        return RUNS_ROOT / "configs" / "rover_training.yaml"

    return RUNS_ROOT / "configs" / f"{dataset_root.name}_training.yaml"

def train_model(epochs, imgsz, batch, device, name, dataset_root=DATASET_ROOT, seed=42, settings=None):
    # Validate the dataset and fine-tune YOLO
    errors, counts = validate_dataset(dataset_root, require_images=True)

    if errors:
        print("Dataset validation failed!")

        for error in errors:
            print(f" - {error}")

        raise SystemExit(1)

    print("Dataset validated successfully.")
    print("Images:", counts)

    training_yaml = create_training_config(dataset_root, training_config_path(dataset_root))

    print("Loading pretrained YOLO26n model... ")

    model = YOLO("yolo26n.pt")

    print("Starting transfer learning... ")

    # settings (augmentation, patience) override the v1 defaults below
    train_settings = {"mosaic": 0.5, "patience": 10, **(settings or {})}

    model.train(
        data=str(training_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=0,
        seed=seed,
        optimizer="AdamW",
        lr0=0.001,
        mixup=0.0,
        copy_paste=0.0,
        plots=True,
        save=True,
        project=str(RUNS_ROOT / "train"),
        name=name,
        exist_ok=False,
        **train_settings,
    )

    print("Training completed!")
    print("Check runs/train for model checkpoints.")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=30)

    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--name", default="pilot_v1")
    parser.add_argument("--dataset", type=Path, default=DATASET_ROOT, help="Dataset folder (default: datasets/rover_obstacles, the v1 data)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default=None, help="Augmentation settings from a scripts/cross_validate.py config, e.g. long_aug (its epochs and patience are not used)")
    parser.add_argument("--fixed-epochs", action="store_true", help="Train every epoch without early stopping; use last.pt, since stock-photo validation picks best.pt poorly")

    args = parser.parse_args()

    if args.epochs < 1 or args.batch < 1 or args.imgsz < 32:
        parser.error("Invalid training parameters")

    settings = {}

    if args.config:
        # Imported here: cross_validate imports evaluate_model, which imports this module
        from scripts.cross_validate import CONFIGS

        if args.config not in CONFIGS:
            parser.error(f"--config must be one of {', '.join(CONFIGS)}")

        settings = {key: value for key, value in CONFIGS[args.config]["train"].items() if key not in ("epochs", "patience")}

    if args.fixed_epochs:
        settings["patience"] = 0

    train_model(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        dataset_root=args.dataset if args.dataset.is_absolute() else PROJECT_ROOT / args.dataset,
        seed=args.seed,
        settings=settings,
    )

if __name__ == "__main__":
    main()