


import json
import statistics
from pathlib import Path

from ultralytics import YOLO

from scripts.benchmark_inference import (IMAGE_DIR, MODEL_PATH, load_images)

ONNX_PATH = MODEL_PATH.with_suffix(".onnx")

OUTPUT_PATH = (
    MODEL_PATH.parent.parent / "runs" / "onnx" / "prediction_consistency.json")


def bbox_iou(a, b):
    """Calculate intersection over union."""

    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = (max(0, x2 - x1) * max(0, y2 - y1))

    area_a = (max(0, a[2] - a[0]) * max(0, a[3] - a[1]))

    area_b = (max(0, b[2] - b[0]) * max(0, b[3] - b[1]))

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def extract_detections(result):
    """Extract class, confidence, and box coordinates."""

    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        detections.append({
            "class_id": int(box.cls[0].item()),
            "confidence": float(box.conf[0].item()),
            "bbox": box.xyxy[0].tolist(),
        })

    return detections


def match_detections(reference, candidate, threshold=0.90):
    """Greedily match same-class detections by IoU."""

    used = set()
    matched_ious = []

    for ref in reference:
        best_index = None
        best_iou = 0.0

        for index, pred in enumerate(candidate):
            if index in used:
                continue

            if ref["class_id"] != pred["class_id"]:
                continue

            overlap = bbox_iou(ref["bbox"], pred["bbox"])

            if overlap > best_iou:
                best_iou = overlap
                best_index = index

        if best_index is not None and best_iou >= threshold:
            used.add(best_index)
            matched_ious.append(best_iou)

    return {
        "pytorch_count": len(reference),
        "onnx_count": len(candidate),
        "matched": len(matched_ious),
        "pytorch_unmatched": (len(reference) - len(matched_ious)),
        "onnx_unmatched": (len(candidate) - len(matched_ious)),
        "mean_matched_iou": (round(statistics.mean(matched_ious), 4) if matched_ious else None),
    }


def main():
    if not ONNX_PATH.is_file():
        raise FileNotFoundError(ONNX_PATH)

    pytorch = YOLO(str(MODEL_PATH))
    onnx = YOLO(str(ONNX_PATH))

    images = load_images(IMAGE_DIR)

    report = []
    kwargs = {
        "imgsz": 640,
        "conf": 0.40,
        "device": "cpu",
        "half": False,
        "rect": False,
        "verbose": False,
    }

    for image_path, frame in images:
        pt_result = pytorch.predict(source=frame, **kwargs)[0]

        onnx_result = onnx.predict(source=frame, **kwargs)[0]

        pt_objects = extract_detections(pt_result)
        onnx_objects = extract_detections(onnx_result)

        comparison = match_detections(pt_objects, onnx_objects)

        comparison["image"] = image_path.name
        report.append(comparison)

        print(image_path.name, comparison)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    total_matches = sum(item["matched"] for item in report)

    unmatched_pt = sum(item["pytorch_unmatched"] for item in report)

    unmatched_onnx = sum(item["onnx_unmatched"] for item in report)

    print("\nPrediction Consistency Summary")
    print("------------------------------")
    print(f"Images: {len(report)}")
    print(f"Matched detections: {total_matches}")
    print(f"Unmatched PyTorch: {unmatched_pt}")
    print(f"Unmatched ONNX: {unmatched_onnx}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
