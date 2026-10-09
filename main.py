

import cv2
import time
from collections import deque

from edgesight.config import (FRAME_WIDTH, FRAME_HEIGHT, FPS)
from edgesight.camera import CameraStream, CameraError

def main():
    camera = CameraStream()

    # Store last 3 frames 
    frame_times = deque(maxlen=30)

    try:
        camera.open()

        print("Camera opened successfully")
        print("Press 'q' to quit")

        while True:
            # Capture one frame
            frame = camera.read()

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
            cv2.putText(frame, f"EdgeSight | FPS: {fps:.1f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Resolution: {width}x{height}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Display the video
            cv2.imshow("EdgeSight - Live Feed", frame)

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