

import time

import cv2
import numpy as np
import pytest

import main
from edgesight.async_camera import AsyncCamera, FramePacket
from edgesight.events import ZoneMonitor
from edgesight.tracker import TrackedObject
from edgesight.video_source import VideoFileSource


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


def write_video(path, frame_count, size=(640, 480)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, size)

    for index in range(frame_count):
        writer.write(np.full((size[1], size[0], 3), index * 10, dtype=np.uint8))

    writer.release()

    return path


@pytest.mark.parametrize("camera", ["0", "1"])
def test_video_and_camera_together_rejected(camera):
    with pytest.raises(SystemExit):
        main.parse_args(["--video", "clip.avi", "--camera", camera])


def test_create_source_chooses_video_or_camera():
    source, name = main.create_source(main.parse_args(["--video", "clip.avi"]), headless=True)

    assert isinstance(source, VideoFileSource)
    assert name == "Video"
    assert source.realtime is False

    source, name = main.create_source(main.parse_args(["--video", "clip.avi"]), headless=False)

    assert source.realtime is True

    source, name = main.create_source(main.parse_args([]), headless=True)

    assert isinstance(source, AsyncCamera)
    assert name == "Camera"
    assert source.source == main.config.CAMERA_INDEX


def test_headless_run_processes_every_video_frame(tmp_path):
    source = VideoFileSource(write_video(tmp_path / "clip.avi", 12), realtime=False).start()
    tracker = FakeTracker()

    summary = main.run(tracker, source, ZoneMonitor(), headless=True)

    assert tracker.calls == 12
    assert summary["processed_frames"] == 12
    assert summary["captured_frames"] == 12
    assert summary["dropped_frames"] == 0
    assert source.is_running is False


def test_main_runs_video_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(main, "create_tracker", lambda backend, threads: FakeTracker())

    main.main(["--video", str(write_video(tmp_path / "clip.avi", 4)), "--headless"])

    out = capsys.readouterr().out

    assert "Video opened successfully (4 frames at 30.0 FPS)" in out
    assert "Processed 4 frames" in out
    assert "Video: 4 captured, 0 dropped" in out


def test_main_reports_missing_video(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(main, "create_tracker", lambda backend, threads: FakeTracker())

    main.main(["--video", str(tmp_path / "missing.avi"), "--headless"])

    out = capsys.readouterr().out

    assert "Error opening video: Video file not found" in out
    assert "Processed" not in out


def test_record_accepts_new_file(tmp_path):
    args = main.parse_args(["--record", str(tmp_path / "demo.mp4")])

    assert args.record == str(tmp_path / "demo.mp4")


@pytest.mark.parametrize("name", ["demo.mkv", "missing_folder/demo.mp4"])
def test_record_rejects_bad_paths(tmp_path, name):
    with pytest.raises(SystemExit):
        main.parse_args(["--record", str(tmp_path / name)])


def test_record_refuses_to_overwrite(tmp_path):
    existing = tmp_path / "take1.mp4"
    existing.write_bytes(b"previous take")

    with pytest.raises(SystemExit):
        main.parse_args(["--record", str(existing)])


def test_headless_run_records_annotated_frames():
    class FakeRecorder:
        def __init__(self):
            self.frames = []

        def write(self, frame, timestamp):
            self.frames.append(frame)

    recorder = FakeRecorder()
    summary = main.run(FakeTracker(), FakeCamera(3), ZoneMonitor(), headless=True, recorder=recorder)

    assert summary["processed_frames"] == 3
    assert len(recorder.frames) == 3

    # Source frames are black; the overlay (zone, box, metrics) must be drawn
    assert all(np.count_nonzero(frame) > 0 for frame in recorder.frames)


def test_main_records_video_to_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(main, "create_tracker", lambda backend, threads: FakeTracker())

    output = tmp_path / "annotated.avi"

    main.main(["--video", str(write_video(tmp_path / "clip.avi", 5)), "--headless", "--record", str(output)])

    out = capsys.readouterr().out

    assert f"Recorded 5 frames to {output} (30.0 FPS playback)" in out

    capture = cv2.VideoCapture(str(output))
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 5
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(30.0, rel=0.01)
    capture.release()
