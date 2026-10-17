

import time

import numpy as np
import pytest

from edgesight.async_camera import LatestFrameBuffer


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