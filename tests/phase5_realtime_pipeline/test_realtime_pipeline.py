"""Phase 5: pipeline profiling, asynchronous camera capture, and main.py."""

import time

import numpy as np
import pytest

import main
from edgesight.events import ZoneMonitor
from edgesight.sources import LatestFrameBuffer
from scripts.benchmark_camera_modes import parse_source, summarize_ms as summarize_camera_ms
from scripts.profile_pipeline import summarize_ms, estimate_tracking_overhead_ms

from tests.helpers import FakeCamera, FakeTracker


# Pipeline stage profile

def test_summarize_timings():
    result = summarize_ms([10.0, 20.0, 30.0])

    assert result["count"] == 3
    assert result["mean"] == pytest.approx(20.0)
    assert result["median"] == pytest.approx(20.0)
    assert result["p95"] == pytest.approx(29.0)
    assert result["min"] == pytest.approx(10.0)
    assert result["max"] == pytest.approx(30.0)


def test_summarize_rejects_empty_input():
    with pytest.raises(ValueError):
        summarize_ms([])


def test_tracking_overhead_estimate():
    speed = {
        "preprocess": 2.0,
        "inference": 18.0,
        "postprocess": 3.0,
    }

    result = estimate_tracking_overhead_ms(
        25.0,
        speed,
    )

    assert result == pytest.approx(2.0)


def test_tracking_overhead_never_negative():
    speed = {
        "preprocess": 3.0,
        "inference": 20.0,
        "postprocess": 4.0,
    }

    result = estimate_tracking_overhead_ms(
        20.0,
        speed,
    )

    assert result == pytest.approx(0.0)


# Camera mode benchmark

def test_summarize_camera_timings():
    result = summarize_camera_ms([10.0, 20.0, 30.0])

    assert result["count"] == 3
    assert result["mean"] == pytest.approx(20.0)
    assert result["median"] == pytest.approx(20.0)
    assert result["p95"] == pytest.approx(29.0)
    assert result["min"] == pytest.approx(10.0)
    assert result["max"] == pytest.approx(30.0)


def test_summarize_empty_timings():
    result = summarize_camera_ms([])

    assert result["count"] == 0
    assert result["mean"] is None
    assert result["p95"] is None


def test_parse_camera_index():
    assert parse_source("0") == 0
    assert parse_source("1") == 1


def test_parse_video_path():
    assert parse_source("videos/test.mp4") == "videos/test.mp4"


# LatestFrameBuffer (live camera)

def test_publish_and_read_frame():
    buffer = LatestFrameBuffer()
    frame = np.zeros((20, 30, 3), dtype=np.uint8)

    assert buffer.publish(frame, captured_at=123.0)

    packet = buffer.get_latest(timeout=0.1)

    assert packet is not None
    assert packet.frame_id == 0
    assert packet.captured_at == 123.0
    assert packet.frame.shape == (20, 30, 3)


def test_latest_frame_replaces_stale_frame():
    buffer = LatestFrameBuffer()

    for value in range(3):
        frame = np.full((5, 5, 3), value, dtype=np.uint8)
        buffer.publish(frame)

    packet = buffer.get_latest(timeout=0.1)

    assert packet is not None
    assert packet.frame_id == 2
    assert np.all(packet.frame == 2)
    assert buffer.dropped_frames == 2


def test_read_after_frame_id_waits_for_newer_frame():
    buffer = LatestFrameBuffer()
    buffer.publish(np.zeros((5, 5, 3), dtype=np.uint8))

    first = buffer.get_latest(timeout=0.1)
    assert first is not None

    start = time.monotonic()
    result = buffer.get_latest(
        after_frame_id=first.frame_id,
        timeout=0.05,
    )
    elapsed = time.monotonic() - start

    assert result is None
    assert elapsed >= 0.03


def test_closed_buffer_rejects_new_frames():
    buffer = LatestFrameBuffer()
    buffer.close()

    frame = np.zeros((5, 5, 3), dtype=np.uint8)

    assert buffer.publish(frame) is False
    assert buffer.get_latest(timeout=0.01) is None


def test_empty_frame_is_rejected():
    buffer = LatestFrameBuffer()

    with pytest.raises(ValueError):
        buffer.publish(np.array([]))


def test_frames_published_counter():
    buffer = LatestFrameBuffer()
    frame = np.zeros((5, 5, 3), dtype=np.uint8)

    assert buffer.frames_published == 0

    buffer.publish(frame)
    buffer.publish(frame)

    assert buffer.frames_published == 2


# main.py: backends, options, headless runs

def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        main.create_tracker("tensorrt")


def test_parse_args_defaults():
    args = main.parse_args([])

    assert args.backend == "onnx"
    assert args.headless is False
    assert args.duration is None
    assert args.threads is None
    assert args.video is None


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
