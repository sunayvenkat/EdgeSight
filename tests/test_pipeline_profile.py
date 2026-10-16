

import pytest

from scripts.profile_pipeline import (
    summarize_ms,
    estimate_tracking_overhead_ms,
)


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