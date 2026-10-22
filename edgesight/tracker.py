

from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

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

class OnnxTracker:
    # Runs ByteTrack on detections from an OnnxDetector

    def __init__(self, detector, tracker_config=TRACKER_CONFIG):
        self.detector = detector

        settings = IterableSimpleNamespace(**YAML.load(check_yaml(tracker_config)))

        if settings.tracker_type != "bytetrack":
            raise ValueError("OnnxTracker only supports ByteTrack")

        self.tracker = BYTETracker(args=settings)

    def track(self, frame):
        # Detect and associate objects across frames

        boxes, scores, class_ids = self.detector.infer(frame)

        data = np.column_stack([boxes, scores, class_ids]).astype(np.float32).reshape(-1, 6)

        tracks = self.tracker.update(Boxes(data, frame.shape[:2]), frame)

        # Mirror Ultralytics' model.track(): hide new tracks until confirmed,
        # and fall back to untracked detections when nothing is being tracked
        if len(tracks) == 0:
            if any(not track.is_activated for track in self.tracker.tracked_stracks):
                return []

            return [self._make_object(None, row[:4], row[4], row[5]) for row in data]

        # Track rows are [x1, y1, x2, y2, track_id, score, class_id, index]
        return [self._make_object(int(row[4]), row[:4], row[5], row[6]) for row in tracks]

    def _make_object(self, track_id, bbox, confidence, class_id):
        return TrackedObject(track_id=track_id, label=self.detector.names[int(class_id)], confidence=float(confidence), bbox=tuple(int(value) for value in bbox))

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


