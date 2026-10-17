

import pytest

from scripts.benchmark_camera_modes import parse_source, summarize_ms


def test_summarize_timings():
    result = summarize_ms([10.0, 20.0, 30.0])

    assert result["count"] == 3
    assert result["mean"] == pytest.approx(20.0)
    assert result["median"] == pytest.approx(20.0)
    assert result["p95"] == pytest.approx(29.0)
    assert result["min"] == pytest.approx(10.0)
    assert result["max"] == pytest.approx(30.0)


def test_summarize_empty_timings():
    result = summarize_ms([])

    assert result["count"] == 0
    assert result["mean"] is None
    assert result["p95"] is None


def test_parse_camera_index():
    assert parse_source("0") == 0
    assert parse_source("1") == 1


def test_parse_video_path():
    assert parse_source("videos/test.mp4") == "videos/test.mp4"