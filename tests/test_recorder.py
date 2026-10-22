

import cv2
import numpy as np
import pytest

from edgesight.recorder import RecordingError, VideoRecorder, codec_for


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
