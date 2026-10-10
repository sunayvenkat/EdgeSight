

import cv2
import time
from collections import deque

from edgesight.config import (FRAME_WIDTH, FRAME_HEIGHT, FPS)
from edgesight.camera import CameraStream, CameraError
from edgesight.detector import ObjectDetector, draw_detections

def main():
    camera = CameraStream()

    # Store last 3 frames 
    frame_times = deque(maxlen=30)

    try:
        print("Loading EdgeSight detection model... ")

        detector = ObjectDetector()

        print("EdgeSight detection model loaded successfully")

        camera.open()

        print("Camera opened successfully")
        print("Press 'q' to quit")

        while True:
            # Capture one frame
            frame = camera.read()

            # Run neural network inference
            inference_start = time.perf_counter()

            detections = detector.detect(frame)

            inference_ms = (time.perf_counter() - inference_start) * 1000

            annotated = draw_detections(frame, detections)

            current_time = time.perf_counter()
            frame_times.append(current_time)

            fps = 0.0

            if len(frame_times) > 1:
                elapsed = frame_times[-1] - frame_times[0]

                if elapsed > 0:
                    fps = (len(frame_times) - 1) / elapsed

            # Retrieve frame dimensions
            height, width = frame.shape[:2]

            # Draw performance information
            cv2.putText(annotated, f"FPS: {fps:.1f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"Inference: {inference_ms:.1f} ms", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated, f"Objects: {len(detections)}", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display the video
            cv2.imshow("EdgeSight - Object Detection", annotated)

            # Exit when user presses 'q'
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