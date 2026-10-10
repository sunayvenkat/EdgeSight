

from dataclasses import dataclass

import cv2
from ultralytics import YOLO

from edgesight.config import MODEL_PATH, DETECTION_CONFIDENCE, INFERENCE_IMAGE_SIZE

@dataclass(frozen=True)
class Detection:
    # A single object detected in an image

    label: str
    confidence: float
    bbox: tuple[int, int, int, int]

class ObjectDetector:
    # Runs YOLO inference on video frames

    def __init__(self, model = None, model_path = MODEL_PATH, confidence = DETECTION_CONFIDENCE):
        self.model = (model if model is not None else YOLO(model_path))

        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        self.confidence = confidence

    def detect(self, frame):
        # Return structured detections for a frame

        results = self.model.predict(source=frame, conf=self.confidence, imgsz=INFERENCE_IMAGE_SIZE, device="cpu", verbose=False)

        result = results[0]
        detections = []

        if result.boxes is None:
            return detections

        for box in result.boxes:
            score = float(box.conf[0].item())

            if score < self.confidence:
                continue

            class_id = int(box.cls[0].item())
            label = result.names[class_id]

            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())

            detections.append(Detection(label=label, confidence=score, bbox=(x1, y1, x2, y2)))

        return detections

def draw_detections(frame, detections):
    # Draws bounding boxes and labels on the frame

    annotated = frame.copy()

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox

        color = (0, 255, 0)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label_text = (f"{detection.label} {detection.confidence:.2f}")
        
        cv2.putText(annotated, label_text, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return annotated

    