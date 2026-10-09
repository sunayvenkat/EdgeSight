

import cv2
from edgesight.config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FPS

class CameraError(RuntimeError):
    # Raised when camera access or capture fails
    pass

class CameraStream:
    def __init__(self, camera_index = CAMERA_INDEX):
        self.camera_index = camera_index
        self.cap = None

    def open(self):
        # Initalize the webcam
        self.close()

        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            self.close()
            raise CameraError(f"Cannot open camera {self.camera_index}")

        # Request desire camera settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)

        return self

    def read(self):
        # Capture and return a camera frame
        if self.cap is None or not self.cap.isOpened():
            raise CameraError("Camera is not opened")

        success, frame = self.cap.read()

        if not success or frame is None:
            raise CameraError("Failed to capture a frame")

        return frame

    def close(self):
        # Release the camera resource
        if self.cap is not None:
            self.cap.release()
            self.cap = None