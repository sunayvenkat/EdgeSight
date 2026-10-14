


import cv2
import numpy as np
import pytest

from scripts.benchmark_inference import (
    load_images,
    percentile,
    summarize_latencies,
)


def test_percentile_calculation():
    values = [10, 20, 30, 40, 50]

    assert percentile(values, 50) == 30
    assert percentile(values, 95) == pytest.approx(48)


def test_latency_summary():
    values = [10.0, 20.0, 30.0]

    result = summarize_latencies(values)

    assert result["count"] == 3
    assert result["mean_ms"] == 20
    assert result["median_ms"] == 20
    assert result["min_ms"] == 10
    assert result["max_ms"] == 30


def test_empty_latency_rejected():
    with pytest.raises(ValueError):
        summarize_latencies([])


def test_image_loading(tmp_path):
    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    image_path = tmp_path / "test.jpg"

    assert cv2.imwrite(str(image_path), frame)

    images = load_images(tmp_path)

    assert len(images) == 1
    assert images[0][0].name == "test.jpg"
    assert images[0][1].shape == (100, 100, 3)


def test_invalid_image_rejected(tmp_path):
    image_path = tmp_path / "corrupt.jpg"

    image_path.write_bytes(b"invalid image")

    with pytest.raises(ValueError):
        load_images(tmp_path)
