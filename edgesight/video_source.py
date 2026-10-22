"""Recorded video input with the same interface as AsyncCamera."""

from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np

from edgesight import config
from edgesight.async_camera import FramePacket


# Reported frame rates outside this range are treated as missing
MIN_VALID_FPS = 1.0
MAX_VALID_FPS = 1000.0


def valid_fps(fps: float | None, fallback: float | None = None) -> float:
    """Return fps if it is a plausible frame rate, otherwise fallback (default config.FPS)."""
    if fps is None or not math.isfinite(fps) or not MIN_VALID_FPS <= fps <= MAX_VALID_FPS:
        return float(config.FPS if fallback is None else fallback)

    return float(fps)


class VideoFileSource:
    """
    Read a video file frame by frame, in order, without dropping frames.

    Frames are decoded on the caller's thread when read_latest is called, so
    every frame is delivered exactly once and tracking sees an unbroken
    sequence. One frame is read ahead, so is_running turns False as soon as
    the last frame has been delivered.

    With realtime=True, delivery is paced to at most the file's frame rate
    (config.FPS when the file reports none). With realtime=False, frames are
    delivered as fast as the caller asks for them.
    """

    def __init__(self, path: str | Path, realtime: bool = True) -> None:
        self.path = Path(path)
        self.realtime = realtime
        self.fps = float(config.FPS)
        self.reported_frame_count = 0

        self._capture: cv2.VideoCapture | None = None
        self._pending: np.ndarray | None = None
        self._started = False
        self._stopped = False
        self._frames_delivered = 0
        self._last_delivery: float | None = None
        self._error: Exception | None = None

    def start(self) -> "VideoFileSource":
        """Open the file and read its first frame."""
        if self._started:
            raise RuntimeError("This video source has already been started")

        self._started = True

        if not self.path.is_file():
            raise RuntimeError(f"Video file not found: {self.path}")

        capture = cv2.VideoCapture(str(self.path))

        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open video file: {self.path}. Check that it is a video format OpenCV can decode.")

        self._capture = capture
        self.fps = valid_fps(capture.get(cv2.CAP_PROP_FPS))
        self.reported_frame_count = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 0)

        self._read_next()

        if self._pending is None:
            self.stop()
            raise RuntimeError(f"Video file contains no readable frames: {self.path}")

        return self

    def _read_next(self) -> None:
        # Decode the next frame into the read-ahead slot; release the file at the end
        if self._capture is None:
            self._pending = None
            return

        success, frame = self._capture.read()

        if success and frame is not None and frame.size > 0:
            self._pending = frame
            return

        self._pending = None
        self._capture.release()
        self._capture = None

    def read_latest(self, after_frame_id: int = -1, timeout: float | None = 1.0) -> FramePacket | None:
        """
        Return the next frame in file order.

        after_frame_id is accepted for compatibility with AsyncCamera; frames
        are never skipped. Returns None at end of file, after stop(), or when
        pacing would wait longer than timeout (the frame is kept for the next
        call).
        """
        if timeout is not None and timeout < 0:
            raise ValueError("timeout cannot be negative")

        if not self.is_running:
            return None

        if self.realtime and self._last_delivery is not None:
            wait = self._last_delivery + 1.0 / self.fps - time.monotonic()

            if wait > 0:
                if timeout is not None and wait > timeout:
                    time.sleep(timeout)
                    return None

                time.sleep(wait)

        frame = self._pending
        frame_id = self._frames_delivered
        captured_at = time.monotonic()

        self._frames_delivered += 1
        self._last_delivery = captured_at

        self._read_next()

        return FramePacket(frame=frame, frame_id=frame_id, captured_at=captured_at)

    @property
    def dropped_frames(self) -> int:
        """Always 0: every frame is delivered."""
        return 0

    @property
    def frames_captured(self) -> int:
        """Count frames delivered to the caller."""
        return self._frames_delivered

    @property
    def is_running(self) -> bool:
        return self._started and not self._stopped and self._pending is not None

    @property
    def error(self) -> Exception | None:
        return self._error

    def stop(self) -> None:
        """Release the file. Safe to call more than once."""
        self._stopped = True
        self._pending = None

        if self._capture is not None:
            self._capture.release()
            self._capture = None
