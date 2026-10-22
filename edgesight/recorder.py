

"""Save annotated frames to a video file."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


# File extension to OpenCV FourCC codec
CODECS = {".mp4": "mp4v", ".avi": "MJPG"}

# Re-time the file when the measured frame rate differs from the nominal rate by more than this
RETIME_TOLERANCE = 0.05


class RecordingError(RuntimeError):
    """Raised when the output video cannot be written."""


def codec_for(path: str | Path) -> str:
    """Return the FourCC codec for a recording path, based on its extension."""
    suffix = Path(path).suffix.lower()

    if suffix not in CODECS:
        raise ValueError(f"Unsupported recording format {suffix or '(none)'!r}; use one of: {', '.join(CODECS)}")

    return CODECS[suffix]


class VideoRecorder:
    """
    Write frames to a video file.

    The file is opened on the first frame so its size matches the frames.

    With retime=True (live cameras), the file's frame rate is corrected on
    close to the measured rate at which frames arrived, so playback runs in
    real time even when processing is slower than the camera. With
    retime=False (recorded video), every output frame stands for one source
    frame and the nominal rate is kept.
    """

    def __init__(self, path: str | Path, fps: float, retime: bool = False) -> None:
        if not fps > 0:
            raise ValueError("fps must be positive")

        self.path = Path(path)
        self.fourcc = codec_for(self.path)
        self.fps = float(fps)
        self.retime = retime
        self.frames_written = 0

        self._writer: cv2.VideoWriter | None = None
        self._size: tuple[int, int] | None = None
        self._first_timestamp: float | None = None
        self._last_timestamp: float | None = None
        self._closed = False

    def _open(self, size: tuple[int, int], fps: float, path: Path) -> cv2.VideoWriter:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*self.fourcc), fps, size)

        if not writer.isOpened():
            writer.release()
            raise RecordingError(f"Could not open {path} for writing")

        return writer

    def write(self, frame: np.ndarray, timestamp: float) -> None:
        """Append a frame captured at timestamp (seconds, monotonic clock)."""
        if self._closed:
            raise RecordingError("Recorder is already closed")

        if self._writer is None:
            height, width = frame.shape[:2]

            self._size = (width, height)
            self._writer = self._open(self._size, self.fps, self.path)
            self._first_timestamp = timestamp

        # A camera can change resolution mid-run; the file's size is fixed
        if (frame.shape[1], frame.shape[0]) != self._size:
            frame = cv2.resize(frame, self._size)

        self._writer.write(frame)

        self._last_timestamp = timestamp
        self.frames_written += 1

    @property
    def measured_fps(self) -> float | None:
        """Average rate at which frames arrived, or None with fewer than two frames."""
        if self.frames_written < 2 or self._last_timestamp <= self._first_timestamp:
            return None

        return (self.frames_written - 1) / (self._last_timestamp - self._first_timestamp)

    def close(self) -> float | None:
        """Finish the file and return the frame rate it plays at (None if nothing was written)."""
        if self._closed:
            return self.fps if self.frames_written else None

        self._closed = True

        if self._writer is None:
            return None

        self._writer.release()
        self._writer = None

        measured = self.measured_fps

        if self.retime and measured is not None and abs(measured - self.fps) / self.fps > RETIME_TOLERANCE:
            self._rewrite(measured)
            self.fps = measured

        return self.fps

    def _rewrite(self, fps: float) -> None:
        # Re-encode the finished file at the measured rate (OpenCV cannot change it in place)
        temporary = self.path.with_name(f".{self.path.stem}.retime{self.path.suffix}")

        reader = cv2.VideoCapture(str(self.path))
        writer = self._open(self._size, fps, temporary)

        try:
            while True:
                success, frame = reader.read()

                if not success:
                    break

                writer.write(frame)
        finally:
            reader.release()
            writer.release()

        os.replace(temporary, self.path)
