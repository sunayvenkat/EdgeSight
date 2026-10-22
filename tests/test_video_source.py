import time

import cv2
import numpy as np
import pytest

from edgesight import config
from edgesight.video_source import VideoFileSource, valid_fps


def write_video(path, frame_count, fps=30.0, size=(64, 48)):
    # Solid frames whose brightness encodes their index, so order survives MJPG compression
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)

    assert writer.isOpened()

    for index in range(frame_count):
        writer.write(np.full((size[1], size[0], 3), index * 20, dtype=np.uint8))

    writer.release()

    return path


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
