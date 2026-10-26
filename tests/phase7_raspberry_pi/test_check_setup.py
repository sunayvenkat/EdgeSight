"""Phase 7: the setup check run on the Raspberry Pi after installing (7.4)."""

import json

import pytest

from edgesight.detector import sha256_file
from scripts.check_setup import check_int8, check_release_files, run


def make_release(models):
    models.mkdir()
    (models / "selected.onnx").write_bytes(b"onnx")
    (models / "selected.pt").write_bytes(b"pt")
    (models / "selection.json").write_text(json.dumps({"release": "edgesight-ml-test", "checkpoint_sha256": sha256_file(models / "selected.pt")}))
    (models / "runtime_config.json").write_text(json.dumps({"onnx_sha256": sha256_file(models / "selected.onnx")}))


def test_release_files_pass_when_hashes_match(tmp_path):
    models = tmp_path / "models"
    make_release(models)

    assert check_release_files(models) == "edgesight-ml-test, cutoffs single cutoff"


def test_release_files_fail_on_missing_or_changed_files(tmp_path):
    models = tmp_path / "models"
    make_release(models)
    (models / "selected.pt").write_bytes(b"retrained")

    with pytest.raises(ValueError, match="selected.pt does not match"):
        check_release_files(models)

    (models / "selected.onnx").unlink()

    with pytest.raises(FileNotFoundError, match="selected.onnx is missing"):
        check_release_files(models)


def test_int8_is_optional_but_must_match(tmp_path):
    models = tmp_path / "models"
    make_release(models)
    (models / "selected_int8_u8_no_head.onnx").write_bytes(b"int8")
    released = {"onnx_sha256": sha256_file(models / "selected.onnx"), "checkpoint_sha256": sha256_file(models / "selected.pt"), "classes": {"cone": 0.23}}
    (models / "class_confidence.json").write_text(json.dumps({**released, "derived_sha256": {"selected_int8_u8_no_head": sha256_file(models / "selected_int8_u8_no_head.onnx"), "selected_int8_s8_no_head": "0" * 64}}))

    assert check_int8(models) == "1 of 2 present: selected_int8_u8_no_head"

    (models / "selected_int8_u8_no_head.onnx").write_bytes(b"stale")
    rows = run([("INT8 models", lambda: check_int8(models))])

    assert rows[0][1] == "FAIL" and "selected_int8_u8_no_head" in rows[0][2]

    (models / "selected_int8_u8_no_head.onnx").unlink()

    assert run([("INT8 models", lambda: check_int8(models))])[0][1] == "SKIP"
