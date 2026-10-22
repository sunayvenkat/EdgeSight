

import time

import numpy as np
import pytest

import main
from edgesight.async_camera import FramePacket
from edgesight.events import ZoneMonitor
from edgesight.tracker import TrackedObject


class FakeCamera:
    def __init__(self, frame_count):
        self.frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(frame_count)]
        self.frames_captured = frame_count
        self.dropped_frames = 0
        self.error = None

    @property
    def is_running(self):
        return bool(self.frames)

    def read_latest(self, after_frame_id=-1, timeout=1.0):
        return FramePacket(frame=self.frames.pop(0), frame_id=after_frame_id + 1, captured_at=time.monotonic())


class FakeTracker:
    def __init__(self):
        self.calls = 0

    def track(self, frame):
        self.calls += 1

        # Bottom-center inside the default monitoring zone
        return [TrackedObject(track_id=1, label="rock", bbox=(300, 300, 340, 400), confidence=0.9)]


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        main.create_tracker("tensorrt")


def test_parse_args_defaults():
    args = main.parse_args([])

    assert args.backend == "onnx"
    assert args.headless is False
    assert args.duration is None
    assert args.threads is None


@pytest.mark.parametrize("argv", [["--threads", "-1"], ["--duration", "0"], ["--backend", "tensorrt"]])
def test_parse_args_rejects_invalid_values(argv):
    with pytest.raises(SystemExit):
        main.parse_args(argv)


def test_headless_run_processes_every_frame(monkeypatch, capsys):
    def fail_imshow(*args):
        raise AssertionError("headless mode must not open a window")

    monkeypatch.setattr(main.cv2, "imshow", fail_imshow)

    tracker = FakeTracker()
    summary = main.run(tracker, FakeCamera(5), ZoneMonitor(), headless=True)

    assert tracker.calls == 5
    assert summary["processed_frames"] == 5
    assert summary["captured_frames"] == 5
    assert summary["median_ms"] is not None
    assert "[ZONE] Track 1: ENTER" in capsys.readouterr().out


def test_run_stops_after_duration():
    tracker = FakeTracker()
    summary = main.run(tracker, FakeCamera(1000), ZoneMonitor(), headless=True, duration=0.0001)

    assert summary["processed_frames"] < 1000


def test_display_missing_on_linux_without_session(monkeypatch):
    monkeypatch.setattr(main.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    assert main.display_available() is False

    monkeypatch.setenv("DISPLAY", ":0")

    assert main.display_available() is True
