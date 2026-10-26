"""Phase 6: video-file input (6.2) and annotated recording (6.6)."""

import time

import cv2
import numpy as np
import pytest

import main
from edgesight import config
from edgesight.events import ZoneMonitor
from edgesight.recorder import RecordingError, VideoRecorder, codec_for
from edgesight.sources import AsyncCamera, VideoFileSource, valid_fps

from tests.helpers import FakeCamera, FakeTracker, write_video


# VideoFileSource (6.2)

def read_all(source):
    packets = []

    while source.is_running:
        packet = source.read_latest(after_frame_id=packets[-1].frame_id if packets else -1, timeout=1.0)

        if packet is not None:
            packets.append(packet)

    return packets


def test_every_frame_delivered_in_order(tmp_path):
    source = VideoFileSource(write_video(tmp_path / "clip.avi", 10), realtime=False).start()
    packets = read_all(source)

    assert [packet.frame_id for packet in packets] == list(range(10))
    assert [round(packet.frame.mean() / 20) for packet in packets] == list(range(10))
    assert packets[0].frame.shape == (48, 64, 3)
    assert source.frames_captured == 10
    assert source.dropped_frames == 0
    assert source.error is None


def test_stops_running_after_last_frame(tmp_path):
    source = VideoFileSource(write_video(tmp_path / "clip.avi", 3), realtime=False).start()

    for _ in range(2):
        assert source.read_latest(timeout=1.0) is not None
        assert source.is_running

    assert source.read_latest(timeout=1.0).frame_id == 2
    assert source.is_running is False
    assert source.read_latest(timeout=1.0) is None


def test_reports_file_fps(tmp_path):
    source = VideoFileSource(write_video(tmp_path / "clip.avi", 2, fps=12.0)).start()

    assert source.fps == pytest.approx(12.0)
    source.stop()


@pytest.mark.parametrize("reported", [0.0, -5.0, float("nan"), float("inf"), 90000.0, None])
def test_nonsense_fps_falls_back_to_config(reported):
    assert valid_fps(reported) == config.FPS


def test_realtime_paces_to_file_fps(tmp_path):
    source = VideoFileSource(write_video(tmp_path / "clip.avi", 5, fps=20.0), realtime=True).start()

    start = time.monotonic()
    packets = read_all(source)
    elapsed = time.monotonic() - start

    # Four 50 ms gaps between five frames
    assert len(packets) == 5
    assert elapsed >= 0.19


def test_realtime_wait_longer_than_timeout_keeps_frame(tmp_path):
    source = VideoFileSource(write_video(tmp_path / "clip.avi", 2, fps=2.0), realtime=True).start()

    assert source.read_latest(timeout=1.0).frame_id == 0
    assert source.read_latest(timeout=0.01) is None
    assert source.read_latest(timeout=1.0).frame_id == 1


def test_missing_file_raises(tmp_path):
    source = VideoFileSource(tmp_path / "missing.avi")

    with pytest.raises(RuntimeError, match="not found"):
        source.start()

    assert source.is_running is False


def test_unreadable_file_raises(tmp_path):
    path = tmp_path / "not_a_video.avi"
    path.write_bytes(b"this is not a video")

    with pytest.raises(RuntimeError):
        VideoFileSource(path).start()


def test_start_twice_rejected(tmp_path):
    source = VideoFileSource(write_video(tmp_path / "clip.avi", 2)).start()

    with pytest.raises(RuntimeError, match="already been started"):
        source.start()

    source.stop()


def test_stop_ends_delivery(tmp_path):
    source = VideoFileSource(write_video(tmp_path / "clip.avi", 5), realtime=False).start()

    assert source.read_latest(timeout=1.0) is not None

    source.stop()
    source.stop()

    assert source.is_running is False
    assert source.read_latest(timeout=1.0) is None
    assert source.frames_captured == 1


# main.py --video (6.2)

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
    monkeypatch.setattr(main, "create_tracker", lambda backend, threads, model=None: FakeTracker())

    main.main(["--video", str(write_video(tmp_path / "clip.avi", 4)), "--headless"])

    out = capsys.readouterr().out

    assert "Video opened successfully (4 frames at 30.0 FPS)" in out
    assert "Processed 4 frames" in out
    assert "Video: 4 captured, 0 dropped" in out


def test_main_reports_missing_video(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(main, "create_tracker", lambda backend, threads, model=None: FakeTracker())

    main.main(["--video", str(tmp_path / "missing.avi"), "--headless"])

    out = capsys.readouterr().out

    assert "Error opening video: Video file not found" in out
    assert "Processed" not in out


# VideoRecorder (6.6)

def make_frame(value, size=(640, 480)):
    return np.full((size[1], size[0], 3), value, dtype=np.uint8)


def read_video(path):
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = []

    while True:
        success, frame = capture.read()

        if not success:
            break

        frames.append(frame)

    capture.release()

    return frames, fps


def record(path, values, timestamps, fps=30.0, retime=False):
    recorder = VideoRecorder(path, fps=fps, retime=retime)

    for value, timestamp in zip(values, timestamps):
        recorder.write(make_frame(value), timestamp)

    return recorder, recorder.close()


@pytest.mark.parametrize("name, codec", [("clip.mp4", "mp4v"), ("clip.avi", "MJPG"), ("CLIP.AVI", "MJPG")])
def test_codec_for_supported_formats(name, codec):
    assert codec_for(name) == codec


@pytest.mark.parametrize("name", ["clip.mkv", "clip"])
def test_codec_for_rejects_other_formats(name):
    with pytest.raises(ValueError):
        codec_for(name)


@pytest.mark.parametrize("suffix", [".avi", ".mp4"])
def test_writes_every_frame_in_order(tmp_path, suffix):
    path = tmp_path / f"out{suffix}"
    values = [20, 80, 140, 200]

    recorder, playback_fps = record(path, values, [index / 30 for index in range(len(values))])

    frames, fps = read_video(path)

    assert recorder.frames_written == 4
    assert playback_fps == pytest.approx(30.0)
    assert len(frames) == 4
    assert frames[0].shape == (480, 640, 3)
    assert fps == pytest.approx(30.0, rel=0.01)

    # Lossy codecs: check brightness order, not exact pixels
    means = [frame.mean() for frame in frames]
    assert means == sorted(means)


def test_retime_matches_slow_processing(tmp_path):
    path = tmp_path / "slow.avi"

    # Frames arrive at 5 FPS while the nominal rate is 30
    _, playback_fps = record(path, [50] * 6, [index * 0.2 for index in range(6)], retime=True)

    frames, fps = read_video(path)

    assert playback_fps == pytest.approx(5.0)
    assert fps == pytest.approx(5.0, rel=0.01)
    assert len(frames) == 6
    assert list(tmp_path.iterdir()) == [path]


def test_no_retime_within_tolerance(tmp_path):
    path = tmp_path / "steady.avi"

    _, playback_fps = record(path, [50] * 6, [index / 29.5 for index in range(6)], retime=True)

    assert playback_fps == pytest.approx(30.0)


def test_video_input_keeps_nominal_rate(tmp_path):
    path = tmp_path / "video.avi"

    # Fast headless processing must not change the source video's timing
    _, playback_fps = record(path, [50] * 6, [index * 0.001 for index in range(6)], fps=25.0, retime=False)

    _, fps = read_video(path)

    assert playback_fps == pytest.approx(25.0)
    assert fps == pytest.approx(25.0, rel=0.01)


def test_resized_frames_keep_file_size(tmp_path):
    path = tmp_path / "sizes.avi"
    recorder = VideoRecorder(path, fps=30.0)

    recorder.write(make_frame(50), 0.0)
    recorder.write(make_frame(50, size=(320, 240)), 0.1)
    recorder.close()

    frames, _ = read_video(path)

    assert [frame.shape for frame in frames] == [(480, 640, 3)] * 2


def test_empty_recording_creates_no_file(tmp_path):
    path = tmp_path / "empty.avi"

    assert VideoRecorder(path, fps=30.0).close() is None
    assert not path.exists()


def test_write_after_close_rejected(tmp_path):
    recorder, _ = record(tmp_path / "closed.avi", [50], [0.0])

    with pytest.raises(RecordingError):
        recorder.write(make_frame(50), 1.0)

    assert recorder.close() == pytest.approx(30.0)


def test_invalid_fps_rejected(tmp_path):
    with pytest.raises(ValueError):
        VideoRecorder(tmp_path / "bad.avi", fps=0)


# main.py --record (6.6)

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
    monkeypatch.setattr(main, "create_tracker", lambda backend, threads, model=None: FakeTracker())

    output = tmp_path / "annotated.avi"

    main.main(["--video", str(write_video(tmp_path / "clip.avi", 5)), "--headless", "--record", str(output)])

    out = capsys.readouterr().out

    assert f"Recorded 5 frames to {output} (30.0 FPS playback)" in out

    capture = cv2.VideoCapture(str(output))
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 5
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(30.0, rel=0.01)
    capture.release()
