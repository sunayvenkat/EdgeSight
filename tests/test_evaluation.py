


from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.evaluate_model import (
    build_report,
    resolve_checkpoint,
)


def make_fake_metrics():
    fake_box = SimpleNamespace(
        mp=0.55,
        mr=0.45,
        map50=0.60,
        map=0.35,
        ap_class_index=[0, 2],
        class_result=Mock(
            side_effect=[
                (0.70, 0.60, 0.75, 0.45),
                (0.40, 0.30, 0.45, 0.25),
            ]
        ),
    )

    return SimpleNamespace(
        box=fake_box,
        names={
            0: "rock",
            1: "cone",
            2: "debris",
        },
    )


def test_evaluation_report():
    metrics = make_fake_metrics()

    report = build_report(
        metrics,
        checkpoint="best.pt",
        image_count=6,
    )

    assert report["split"] == "val"
    assert report["images"] == 6
    assert report["precision"] == 0.55
    assert report["recall"] == 0.45
    assert report["mAP50"] == 0.60
    assert report["mAP50_95"] == 0.35

    assert len(report["per_class"]) == 2
    assert report["per_class"][0]["class_name"] == "rock"
    assert report["per_class"][1]["class_name"] == "debris"


def test_missing_checkpoint(tmp_path):
    missing_path = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(missing_path)
