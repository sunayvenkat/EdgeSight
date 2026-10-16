

"""Asynchronous camera capture with a latest-frame buffer."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FramePacket:
    """A captured frame and its metadata."""

    frame: np.ndarray
    frame_id: int
    captured_at: float


class LatestFrameBuffer:
    """Thread-safe buffer that retains only the newest pending frame."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest: FramePacket | None = None
        self._next_frame_id = 0
        self._last_delivered_id = -1
        self._dropped_frames = 0
        self._closed = False

    def publish(
        self,
        frame: np.ndarray,
        captured_at: float | None = None,
    ) -> bool:
        """Publish a frame, replacing any older pending frame."""
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame must be a non-empty NumPy array")

        frame_copy = frame.copy()
        timestamp = (time.monotonic() if captured_at is None else captured_at)

        with self._condition:
            if self._closed:
                return False

            if (self._latest is not None and self._latest.frame_id > self._last_delivered_id):
                self._dropped_frames += 1

            packet = FramePacket(frame=frame_copy, frame_id=self._next_frame_id, captured_at=timestamp)
            self._next_frame_id += 1
            self._latest = packet
            self._condition.notify_all()

        return True

    def get_latest(self, after_frame_id: int = -1, timeout: float | None = None) -> FramePacket | None:
        """
        Wait for a frame newer than after_frame_id.

        Returns None if the buffer closes or the timeout expires.
        """
        if timeout is not None and timeout < 0:
            raise ValueError("timeout cannot be negative")

        deadline = (None if timeout is None else time.monotonic() + timeout)

        with self._condition:
            while (not self._closed and (self._latest is None or self._latest.frame_id <= after_frame_id)):
                if deadline is None:
                    self._condition.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._condition.wait(remaining)

            if (self._latest is None or self._latest.frame_id <= after_frame_id):
                return None

            self._last_delivered_id = self._latest.frame_id
            return self._latest

    @property
    def dropped_frames(self) -> int:
        """Count frames overwritten before the consumer retrieved them."""
        with self._condition:
            return self._dropped_frames

    def close(self) -> None:
        """Wake any consumer waiting for a frame."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class AsyncCamera:
    """Capture camera frames on a dedicated background thread."""

    def __init__(
        self,
        source: int | str = 0,
        width: int | None = 640,
        height: int | None = 480,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height

        self.buffer = LatestFrameBuffer()
        self._stop_event = threading.Event()
        self._started_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._opened = False
        self._error: Exception | None = None

    def start(self, timeout: float = 5.0) -> "AsyncCamera":
        """Start capture and wait for the camera to initialize."""
        if self._thread is not None:
            raise RuntimeError("This camera instance has already been started")

        self._thread = threading.Thread(
            target=self._capture_loop,
            name="EdgeSightCamera",
            daemon=True,
        )
        self._thread.start()

        if not self._started_event.wait(timeout):
            self.stop()
            raise TimeoutError("Camera initialization timed out")

        if self._error is not None:
            raise RuntimeError("Camera initialization failed") from self._error

        if not self._opened:
            raise RuntimeError(
                f"Could not open camera source: {self.source!r}. "
                "Check the camera index, permissions, and connections."
            )

        return self

    def _capture_loop(self) -> None:
        capture = None

        try:
            capture = cv2.VideoCapture(self.source)

            if self.width is not None:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)

            if self.height is not None:
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            self._opened = capture.isOpened()
            self._started_event.set()

            if not self._opened:
                return

            while not self._stop_event.is_set():
                success, frame = capture.read()

                if not success:
                    if not self._stop_event.is_set():
                        self._error = RuntimeError(
                            "Camera stopped returning frames"
                        )
                    break

                self.buffer.publish(frame, time.monotonic())

        except Exception as exc:
            self._error = exc
            self._started_event.set()

        finally:
            if capture is not None:
                capture.release()

            self.buffer.close()

    def read_latest(
        self,
        after_frame_id: int = -1,
        timeout: float | None = 1.0,
    ) -> FramePacket | None:
        """Retrieve the newest available frame."""
        return self.buffer.get_latest(after_frame_id, timeout)

    @property
    def dropped_frames(self) -> int:
        return self.buffer.dropped_frames

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
 
    @property
    def error(self) -> Exception | None:
        return self._error

    def stop(self, timeout: float = 2.0) -> None:
        """Request shutdown and wait briefly for capture to finish."""
        self._stop_event.set()
        self.buffer.close()

        if self._thread is not None:
            self._thread.join(timeout)