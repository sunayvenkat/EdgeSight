"""Phase 6: INT8 quantization (6.4)."""

import json
from pathlib import Path

import cv2
import numpy as np
import onnx
import pytest
from onnx import helper

from scripts.evaluate_quantization import summarize_consistency, to_records
from scripts.quantize_onnx import ImageCalibrationReader, copy_metadata, count_quantized, list_calibration_images, select_nodes_to_exclude, update_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


FP32_PATH = PROJECT_ROOT / "models" / "selected.onnx"


INT8_PATH = PROJECT_ROOT / "models" / "selected_int8_u8_no_head.onnx"


def write_images(directory, count=2, shape=(48, 64, 3)):
    paths = []

    for index in range(count):
        path = directory / f"img_{index}.jpg"
        cv2.imwrite(str(path), np.full(shape, 40 * (index + 1), dtype=np.uint8))
        paths.append(path)

    return paths


def make_model(node_names, metadata=None):
    nodes = [helper.make_node("Identity", [f"t{index}"], [f"t{index + 1}"], name=name) for index, name in enumerate(node_names)]

    graph = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("t0", onnx.TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info(f"t{len(node_names)}", onnx.TensorProto.FLOAT, [1])],
    )

    model = helper.make_model(graph)

    for key, value in (metadata or {}).items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value

    return model


def test_lists_only_images_sorted(tmp_path):
    write_images(tmp_path, count=2)
    (tmp_path / "notes.txt").write_text("skip", encoding="utf-8")

    paths = list_calibration_images(tmp_path)

    assert [path.name for path in paths] == ["img_0.jpg", "img_1.jpg"]


def test_empty_calibration_directory_rejected(tmp_path):
    with pytest.raises(ValueError):
        list_calibration_images(tmp_path)


def test_calibration_reader_yields_app_tensors(tmp_path):
    paths = write_images(tmp_path, count=2)

    reader = ImageCalibrationReader(paths, input_name="images", image_size=640)

    batches = []

    while (batch := reader.get_next()) is not None:
        batches.append(batch)

    assert len(batches) == 2

    for batch in batches:
        assert set(batch) == {"images"}
        assert batch["images"].shape == (1, 3, 640, 640)
        assert batch["images"].dtype == np.float32
        assert 0.0 <= batch["images"].min() and batch["images"].max() <= 1.0

    # Letterbox padding uses 114 / 255.
    assert batches[0]["images"][0, 0, 0, 0] == pytest.approx(114 / 255)

    reader.rewind()
    assert reader.get_next() is not None


def test_metadata_copied_and_overwritten():
    source = make_model(["a"], {"names": "{0: 'rock', 1: 'cone', 2: 'debris'}", "stride": "32", "imgsz": "[640, 640]"})
    target = make_model(["a"], {"stride": "16", "onnx.quant.pre_process": "onnxruntime.quant"})

    copy_metadata(source, target)

    metadata = {prop.key: prop.value for prop in target.metadata_props}

    assert metadata["names"] == "{0: 'rock', 1: 'cone', 2: 'debris'}"
    assert metadata["stride"] == "32"
    assert metadata["imgsz"] == "[640, 640]"
    assert metadata["onnx.quant.pre_process"] == "onnxruntime.quant"
    assert len(target.metadata_props) == 4


def test_variant_node_selection():
    names = [
        "/model.22/cv2/conv/Conv",
        "/model.23/cv2.0/cv2.0.0/conv/Conv",
        "/model.23/cv3.2/cv3.2.2/Conv",
        "/model.23/Reshape",
        "/model.23/Concat_3",
        "/model.23/Sigmoid",
    ]
    model = make_model(names)

    assert select_nodes_to_exclude(model, "full") == []
    assert select_nodes_to_exclude(model, "no_head") == names[1:]
    assert select_nodes_to_exclude(model, "no_decode") == ["/model.23/Reshape", "/model.23/Concat_3", "/model.23/Sigmoid"]

    with pytest.raises(ValueError):
        select_nodes_to_exclude(model, "unknown")


def test_count_quantized_nodes():
    graph = helper.make_graph(
        [
            helper.make_node("QuantizeLinear", ["x", "s"], ["q"]),
            helper.make_node("DequantizeLinear", ["q", "s"], ["y"]),
            helper.make_node("DequantizeLinear", ["w", "s"], ["z"]),
        ],
        "qdq",
        [],
        [],
    )

    assert count_quantized(helper.make_model(graph)) == {"QuantizeLinear": 1, "DequantizeLinear": 2}


def test_manifest_merges_variants(tmp_path):
    path = tmp_path / "manifest.json"
    calibration = {"split": "train", "images": {"a.jpg": "hash"}}

    update_manifest(path, "selected_int8_u8_full", {"variant": "full"}, calibration)
    update_manifest(path, "selected_int8_u8_no_head", {"variant": "no_head"}, calibration)

    manifest = json.loads(path.read_text(encoding="utf-8"))

    assert set(manifest["variants"]) == {"selected_int8_u8_full", "selected_int8_u8_no_head"}
    assert manifest["calibration"] == calibration

    with pytest.raises(ValueError):
        update_manifest(path, "other", {}, {"split": "val"})


def test_consistency_summary_weights_iou_by_matches():
    per_image = [
        {"pytorch_count": 2, "onnx_count": 2, "matched": 2, "pytorch_unmatched": 0, "onnx_unmatched": 0, "mean_matched_iou": 1.0},
        {"pytorch_count": 2, "onnx_count": 1, "matched": 1, "pytorch_unmatched": 1, "onnx_unmatched": 0, "mean_matched_iou": 0.7},
        {"pytorch_count": 0, "onnx_count": 0, "matched": 0, "pytorch_unmatched": 0, "onnx_unmatched": 0, "mean_matched_iou": None},
    ]

    summary = summarize_consistency(per_image)

    assert summary["matched"] == 3
    assert summary["reference_unmatched"] == 1
    assert summary["mean_matched_iou"] == pytest.approx(0.9)


def test_records_from_detector_output():
    boxes = np.array([[1, 2, 3, 4]], dtype=np.float32)
    records = to_records(boxes, np.array([0.5], dtype=np.float32), np.array([2]))

    assert records == [{"class_id": 2, "confidence": 0.5, "bbox": [1.0, 2.0, 3.0, 4.0]}]


@pytest.mark.skipif(not (FP32_PATH.is_file() and INT8_PATH.is_file()), reason="Quantized model not generated")
def test_quantized_model_keeps_interface_and_metadata():
    import onnxruntime as ort

    from edgesight.detector import OnnxDetector

    source = {prop.key: prop.value for prop in onnx.load(str(FP32_PATH)).metadata_props}
    quantized = {prop.key: prop.value for prop in onnx.load(str(INT8_PATH)).metadata_props}

    assert all(quantized[key] == value for key, value in source.items())

    session = ort.InferenceSession(str(INT8_PATH), providers=["CPUExecutionProvider"])
    detector = OnnxDetector(session=session)

    assert detector.names == {0: "rock", 1: "cone", 2: "debris"}

    output = session.run(None, {"images": np.zeros((1, 3, 640, 640), dtype=np.float32)})[0]

    assert output.shape == (1, 7, 8400)
    assert output.dtype == np.float32
