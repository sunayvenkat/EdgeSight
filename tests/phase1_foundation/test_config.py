"""Phase 1: project configuration."""

from edgesight.config import FRAME_WIDTH, FRAME_HEIGHT, FPS


def test_resolution_is_positive():
    assert FRAME_WIDTH > 0
    assert FRAME_HEIGHT > 0


def test_fps_is_positive():
    assert FPS > 0
