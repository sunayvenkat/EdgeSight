


from pathlib import Path

import cv2
import yaml


DATASET_ROOT = (Path(__file__).resolve().parents[1] / "datasets" / "rover_obstacles")


def preview_dataset():
    config_path = DATASET_ROOT / "data.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    names = config["names"]

    for split in ("train", "val", "test"):
        image_dir = DATASET_ROOT / "images" / split
        label_dir = DATASET_ROOT / "labels" / split

        images = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower()
            in {".jpg", ".jpeg", ".png"}
        )

        for image_path in images:
            frame = cv2.imread(str(image_path))

            if frame is None:
                raise ValueError(f"Cannot read image: {image_path}")

            height, width = frame.shape[:2]
            label_path = label_dir / (image_path.stem + ".txt")

            lines = label_path.read_text(encoding="utf-8").splitlines()

            for line in lines:
                if not line.strip():
                    continue

                values = line.split()

                class_id = int(values[0])
                x, y, w, h = map(float, values[1:])

                x1 = int((x - w / 2) * width)
                y1 = int((y - h / 2) * height)
                x2 = int((x + w / 2) * width)
                y2 = int((y + h / 2) * height)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(frame, str(names[class_id]), (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Resize large images for display.
            scale = min(1.0, 1100 / width, 700 / height)

            if scale < 1.0:
                frame = cv2.resize(frame, None, fx=scale, fy=scale)

            print(f"Viewing: {split}/{image_path.name}")

            cv2.imshow("EdgeSight Dataset Preview", frame)

            print("Space: next | Q: quit")

            key = cv2.waitKey(0) & 0xFF

            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("Dataset preview completed!")


if __name__ == "__main__":
    preview_dataset()
