

"""
Frame sources for the EdgeSight pipeline.

AsyncCamera captures a live camera on a background thread and keeps only the
newest frame. VideoFileSource reads a recorded file in order without dropping
frames. Both deliver FramePacket objects through the same interface: start(),
stop(), is_running, read_latest(), error, dropped_frames, and frames_captured.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from edgesight import config


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

    def publish(self, frame: np.ndarray, captured_at: float | None = None) -> bool:
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

    @property
    def frames_published(self) -> int:
        with self._condition:
            return self._next_frame_id

    def close(self) -> None:
        """Wake any consumer waiting for a frame."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class AsyncCamera:
    """Capture camera frames on a dedicated background thread."""

    def __init__(self, source: int | str = 0, width: int | None = 640, height: int | None = 480, fps: int | None = None) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps

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

        self._thread = threading.Thread(target=self._capture_loop, name="EdgeSightCamera", daemon=True)
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

            if self.fps is not None:
                capture.set(cv2.CAP_PROP_FPS, self.fps)

            self._opened = capture.isOpened()
            self._started_event.set()

            if not self._opened:
                return

            while not self._stop_event.is_set():
                success, frame = capture.read()

                if not success:
                    if not self._stop_event.is_set():
                        self._error = RuntimeError("Camera stopped returning frames")
                    break

                self.buffer.publish(frame, time.monotonic())

        except Exception as exc:
            self._error = exc
            self._started_event.set()

        finally:
            if capture is not None:
                capture.release()

            self.buffer.close()

    def read_latest(self, after_frame_id: int = -1, timeout: float | None = 1.0) -> FramePacket | None:
        """Retrieve the newest available frame."""
        return self.buffer.get_latest(after_frame_id, timeout)

    @property
    def dropped_frames(self) -> int:
        return self.buffer.dropped_frames

    @property
    def frames_captured(self) -> int:
        return self.buffer.frames_published

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
