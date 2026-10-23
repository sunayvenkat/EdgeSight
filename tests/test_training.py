


from pathlib import Path

import pytest
import yaml

from scripts.train_model import create_training_config


def test_training_config_uses_absolute_path(tmp_path):
    dataset_root = tmp_path / "rover_obstacles"
    dataset_root.mkdir()

    config = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 3,
        "names": {
            0: "rock",
            1: "cone",
            2: "debris",
        },
    }

    source = dataset_root / "data.yaml"

    source.write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )

    output = tmp_path / "runs" / "training.yaml"

    create_training_config(dataset_root, output)

    result = yaml.safe_load(
        output.read_text(encoding="utf-8")
    )

    assert Path(result["path"]).is_absolute()
    assert Path(result["path"]) == dataset_root.resolve()
    assert result["nc"] == 3
    assert result["names"][0] == "rock"
    assert result["names"][1] == "cone"
    assert result["names"][2] == "debris"


def test_training_config_missing_yaml(tmp_path):
    dataset_root = tmp_path / "missing_dataset"
    dataset_root.mkdir()

    output = tmp_path / "runs" / "training.yaml"

    with pytest.raises(FileNotFoundError):
        create_training_config(dataset_root, output)


def test_training_config_path_keeps_v1_name():
    from scripts.train_model import DATASET_ROOT, RUNS_ROOT, training_config_path

    assert training_config_path(DATASET_ROOT) == RUNS_ROOT / "configs" / "rover_training.yaml"
    assert training_config_path(DATASET_ROOT.parent / "rover_obstacles_v2") == RUNS_ROOT / "configs" / "rover_obstacles_v2_training.yaml"
