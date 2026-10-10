

from dataclasses import dataclass

import cv2

from edgesight.config import INFERENCE_IMAGE_SIZE, TRACKER_CONFIG

@dataclass(frozen=True)
class TrackedObject:
    # Object detection with optional tracking ID

    track_id: int | None
    label: str
    bbox: tuple[int, int, int, int]
    confidence: float

class ObjectTracker:
    def __init__(self, detector):
        self.model = detector.model
        self.confidence = detector.confidence

    def track(self, frame):
        # Detect and associate objects across frames

        results = self.model.track(source=frame, persist=True, tracker=TRACKER_CONFIG, conf=self.confidence, imgsz=INFERENCE_IMAGE_SIZE, device="cpu", verbose=False)

        result = results[0]

        if result.boxes is None:
            return []

        track_ids = result.boxes.id
        tracked_objects = []

        for index, box in enumerate(result.boxes):
            track_id = None

            if track_ids is not None:
                track_id = int(track_ids[index].item())

            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())

            tracked_objects.append(TrackedObject(track_id=track_id, label=result.names[class_id], confidence=confidence, bbox=(x1, y1, x2, y2)))

        return tracked_objects

def draw_tracks(frame, objects):
    # Draw tracked objects and their IDs

    annotated = frame.copy()

    for obj in objects:
        x1, y1, x2, y2 = obj.bbox

        color = (0, 255, 0)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        identifier = (str(obj.track_id) if obj.track_id is not None else "?")

        text = (f"ID {identifier} | " f"{obj.label} {obj.confidence:.2f}")

        cv2.putText(annotated, text, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    return annotated


