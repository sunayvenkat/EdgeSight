


import pytest

from scripts.compare_predictions import (
    bbox_iou,
    match_detections,
)


def make_detection(class_id, bbox):
    return {
        "class_id": class_id,
        "confidence": 0.90,
        "bbox": bbox,
    }


def test_identical_boxes():
    box = (10, 10, 100, 100)

    assert bbox_iou(box, box) == pytest.approx(1.0)


def test_disjoint_boxes():
    first = (0, 0, 10, 10)
    second = (20, 20, 30, 30)

    assert bbox_iou(first, second) == 0.0


def test_partial_overlap():
    first = (0, 0, 10, 10)
    second = (5, 5, 15, 15)

    assert bbox_iou(first, second) == pytest.approx(1 / 7)


def test_matching_predictions():
    reference = [make_detection(0, (10, 10, 100, 100))]

    candidate = [make_detection(0, (10, 10, 100, 100))]

    result = match_detections(reference, candidate)

    assert result["matched"] == 1
    assert result["pytorch_unmatched"] == 0
    assert result["onnx_unmatched"] == 0


def test_different_classes_not_matched():
    reference = [make_detection(0, (10, 10, 100, 100))]

    candidate = [make_detection(1, (10, 10, 100, 100))]

    result = match_detections(reference, candidate)

    assert result["matched"] == 0
    assert result["pytorch_unmatched"] == 1
    assert result["onnx_unmatched"] == 1
