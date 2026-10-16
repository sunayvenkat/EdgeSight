

"""Run EdgeSight inference with asynchronous camera capture."""

from pathlib import Path

import cv2
from ultralytics import YOLO

from edgesight import config
from edgesight.async_camera import AsyncCamera
from edgesight.events import ZoneMonitor
from edgesight.tracker import TrackedObject, draw_tracks


MODEL_PATH = Path("models/selected.pt")


def convert_result(result) -> list[TrackedObject]:
    """Convert Ultralytics detections to EdgeSight tracked objects."""
    objects = []
    boxes = result.boxes

    if boxes is None:
        return objects

    coordinates = boxes.xyxy.cpu().tolist()
    classes = boxes.cls.cpu().tolist()
    confidences = boxes.conf.cpu().tolist()

    track_ids = (
        boxes.id.int().cpu().tolist()
        if boxes.id is not None
        else [None] * len(coordinates)
    )

    for bbox, class_id, confidence, track_id in zip(
        coordinates, classes, confidences, track_ids
    ):
        objects.append(
            TrackedObject(
                track_id=track_id,
                label=result.names[int(class_id)],
                confidence=float(confidence),
                bbox=tuple(round(value) for value in bbox),
            )
        )

    return objects


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    model = YOLO(str(MODEL_PATH))
    zone_monitor = ZoneMonitor()
    last_frame_id = -1
    processed_frames = 0

    camera = AsyncCamera(source=0, width=640, height=480)

    try:
        camera.start()
        print("EdgeSight asynchronous camera started.")
        print("Press Q in the video window to quit.")

        while camera.is_running:
            packet = camera.read_latest(
                after_frame_id=last_frame_id,
                timeout=1.0,
            )

            if packet is None:
                if camera.error is not None:
                    print(f"Camera error: {camera.error}")
                    break
                continue

            last_frame_id = packet.frame_id
            frame = packet.frame

            results = model.track(
                source=frame,
                persist=True,
                tracker=config.TRACKER_CONFIG,
                conf=config.DETECTION_CONFIDENCE,
                imgsz=config.INFERENCE_IMAGE_SIZE,
                device="cpu",
                verbose=False,
            )

            result = results[0]
            objects = convert_result(result)

            zone_status, events = zone_monitor.update(
                objects,
                frame.shape,
            )

            display_frame = draw_tracks(frame, objects)

            cv2.putText(
                display_frame,
                f"Zone: {zone_status}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                display_frame,
                f"Frame: {packet.frame_id} | "
                f"Dropped: {camera.dropped_frames}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            for event in events:
                print(
                    f"Frame {packet.frame_id}: "
                    f"track={event.track_id}, action={event.action}"
                )

            cv2.imshow("EdgeSight Async Vision", display_frame)
            processed_frames += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Stopping EdgeSight...")

    finally:
        camera.stop()
        cv2.destroyAllWindows()

    print(f"Processed frames: {processed_frames}")
    print(f"Overwritten frames: {camera.dropped_frames}")


if __name__ == "__main__":
    main()