

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

def train_model(epochs, imgsz, batch, device, name):
    # Validate the dataset and fine-tune YOLO
    errors, counts = validate_dataset(DATASET_ROOT, require_images=True)

    if errors:
        print("Dataset validation failed!")

        for error in errors:
            print(f" - {error}")

        raise SystemExit(1)

    print("Dataset validated successfully.")
    print("Images:", counts)

    training_yaml = create_training_config(DATASET_ROOT, RUNS_ROOT / "configs" / "rover_training.yaml")

    print("Loading pretrained YOLO26n model... ")

    model = YOLO("yolo26n.pt")

    print("Starting transfer learning... ")

    model.train(
        data=str(training_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=0,
        seed=42,
        optimizer="AdamW",
        lr0=0.001,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        patience=10,
        plots=True,
        save=True,
        project=str(RUNS_ROOT / "train"),
        name=name,
        exist_ok=False,
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

    args = parser.parse_args()

    if args.epochs < 1 or args.batch < 1 or args.imgsz < 32:
        parser.error("Invalid training parameters")

    train_model(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name
    )

if __name__ == "__main__":
    main()