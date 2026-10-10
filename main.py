

import cv2
import time
from collections import deque

from edgesight.camera import CameraStream, CameraError
from edgesight.detector import ObjectDetector
from edgesight.events import ZoneMonitor
from edgesight.tracker import ObjectTracker, draw_tracks


def main():
    camera = CameraStream()

    # Store last 3 frames 
    frame_times = deque(maxlen=30)

    try:
        print("Loading EdgeSight detection model... ")

        detector = ObjectDetector()
        tracker = ObjectTracker(detector)
        zone = ZoneMonitor()

        print("EdgeSight detection model loaded successfully")

        camera.open()

        print("Camera opened successfully")
        print("Press 'q' to quit")

        while True:
            # Capture one frame
            frame = camera.read()

            # Run detection and tracking together
            start = time.perf_counter()
            objects = tracker.track(frame)
            inference_ms = (time.perf_counter() - start) * 1000

            # Evaluate monitoring zone
            status, events = zone.update(objects, frame.shape)

            for event in events:
                print(f"[ZONE] Track {event.track_id}: " f"{event.action}")

            # Draw tracked objects
            annotated = draw_tracks(frame, objects)

            left, top, right, bottom = zone.bounds(frame.shape)

            cv2.rectangle(annotated, (left, top), (right, bottom), (0, 255, 255), 2)

            # Calculate FPS
            now = time.perf_counter()
            frame_times.append(now)

            fps = 0.0

            if len(frame_times) > 1:
                elapsed = frame_times[-1] - frame_times[0]

                if elapsed > 0:
                    fps = (len(frame_times) - 1) / elapsed

            # Draw metrics
            metrics = [f"FPS: {fps:.1f}", f"Inference: {inference_ms:.1f} ms", f"Objects: {len(objects)}", f"Status: {status}"]

            for i, text in enumerate(metrics):
                color = (255, 255, 255)

                if i == 3:
                    color = ((0, 255, 255) if status == "CAUTION" else (0, 255, 0))

                cv2.putText(annotated, text, (15, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("EdgeSight - Object Detection", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    
    except CameraError as error:
        print(f"Error opening camera: {error}")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Exiting...")

    finally:
        camera.close()
        cv2.destroyAllWindows()
        print("Camera resources released")

if __name__ == "__main__":
    main()