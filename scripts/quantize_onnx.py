

import argparse
import json
import platform
import re
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

from edgesight.onnx_detector import preprocess, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_PATH = PROJECT_ROOT / "models" / "selected.onnx"
ONNX_MANIFEST_PATH = PROJECT_ROOT / "models" / "onnx_manifest.json"
MANIFEST_PATH = PROJECT_ROOT / "models" / "quantization_manifest.json"

# Calibration uses the training split only; val and test stay unseen.
CALIBRATION_DIR = PROJECT_ROOT / "datasets" / "rover_obstacles" / "images" / "train"

EXTENSIONS = {".jpg", ".jpeg", ".png"}

IMAGE_SIZE = 640

HEAD_PREFIX = "/model.23/"

# Nodes inside the head's per-scale conv branches (box: cv2, class: cv3).
HEAD_BRANCH = re.compile(r"^/model\.23/cv[23]\.")

VARIANTS = {
    "full": "Quantize every supported node",
    "no_decode": "Keep the head's output-decode nodes (reshape, box decode, sigmoid, concat) in FP32",
    "no_head": "Keep the whole detection head (/model.23/) in FP32",
}

CALIBRATION_METHODS = {
    "minmax": CalibrationMethod.MinMax,
    "entropy": CalibrationMethod.Entropy,
    "percentile": CalibrationMethod.Percentile,
}


def list_calibration_images(directory):
    """Return sorted image paths in a calibration directory."""

    directory = Path(directory)

    if not directory.is_dir():
        raise FileNotFoundError(directory)

    paths = [path for path in sorted(directory.iterdir()) if path.suffix.lower() in EXTENSIONS]

    if not paths:
        raise ValueError(f"No calibration images in {directory}")

    return paths


class ImageCalibrationReader(CalibrationDataReader):
    """Feed images through the app's exact preprocessing for calibration."""

    def __init__(self, image_paths, input_name="images", image_size=IMAGE_SIZE):
        self.image_paths = list(image_paths)
        self.input_name = input_name
        self.image_size = image_size
        self._index = 0

    def __len__(self):
        return len(self.image_paths)

    def get_next(self):
        if self._index >= len(self.image_paths):
            return None

        path = self.image_paths[self._index]
        self._index += 1

        frame = cv2.imread(str(path))

        if frame is None:
            raise ValueError(f"Unreadable image: {path}")

        tensor, _, _ = preprocess(frame, self.image_size)

        return {self.input_name: tensor}

    def rewind(self):
        self._index = 0


def copy_metadata(source, target):
    """Copy custom metadata_props (names, stride, imgsz, ...) onto a model."""

    existing = {prop.key: prop for prop in target.metadata_props}

    for prop in source.metadata_props:
        if prop.key in existing:
            existing[prop.key].value = prop.value
        else:
            entry = target.metadata_props.add()
            entry.key = prop.key
            entry.value = prop.value

    return target


def select_nodes_to_exclude(model, variant):
    """Return node names to keep in FP32 for a quantization variant."""

    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")

    names = [node.name for node in model.graph.node]

    if variant == "full":
        return []

    if variant == "no_head":
        return [name for name in names if name.startswith(HEAD_PREFIX)]

    # no_decode: head nodes that are not part of the conv branches.
    return [name for name in names if name.startswith(HEAD_PREFIX) and not HEAD_BRANCH.match(name)]


def count_quantized(model):
    """Count QuantizeLinear / DequantizeLinear nodes in a model."""

    quantize = sum(node.op_type == "QuantizeLinear" for node in model.graph.node)
    dequantize = sum(node.op_type == "DequantizeLinear" for node in model.graph.node)

    return {"QuantizeLinear": quantize, "DequantizeLinear": dequantize}


def verify_source(source_path):
    """Check the source ONNX file against its export manifest."""

    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    actual = sha256_file(source_path)

    if source_path.resolve() == SOURCE_PATH.resolve() and ONNX_MANIFEST_PATH.is_file():
        manifest = json.loads(ONNX_MANIFEST_PATH.read_text(encoding="utf-8"))

        if manifest["onnx_sha256"] != actual:
            raise ValueError("models/selected.onnx differs from models/onnx_manifest.json")

    return actual


def update_manifest(path, key, entry, calibration):
    """Merge one model's record into the manifest file."""

    path = Path(path)

    manifest = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    if manifest.get("calibration", calibration) != calibration:
        raise ValueError("Calibration set differs from the one recorded in the manifest")

    manifest["calibration"] = calibration
    manifest.setdefault("variants", {})[key] = entry

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return manifest


def quantize(source_path, output_path, variant, calibration_paths, calibrate_method, activation_type, weight_type):
    """Pre-process, calibrate, and statically quantize one variant."""

    source = onnx.load(str(source_path))
    input_name = source.graph.input[0].name

    with tempfile.TemporaryDirectory() as workdir:
        prepared_path = Path(workdir) / "prepared.onnx"

        quant_pre_process(str(source_path), str(prepared_path))

        prepared = onnx.load(str(prepared_path))
        excluded = select_nodes_to_exclude(prepared, variant)

        reader = ImageCalibrationReader(calibration_paths, input_name=input_name)

        quantize_static(
            str(prepared_path),
            str(output_path),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            activation_type=activation_type,
            weight_type=weight_type,
            nodes_to_exclude=excluded,
            calibrate_method=calibrate_method,
        )

    quantized = onnx.load(str(output_path))
    copy_metadata(source, quantized)
    onnx.save(quantized, str(output_path))

    return excluded, count_quantized(quantized)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--variant", choices=VARIANTS.keys(), default="full")
    parser.add_argument("--source", default=str(SOURCE_PATH))
    parser.add_argument("--output", default=None, help="Defaults to models/selected_int8_<s8|u8>_<variant>.onnx")
    parser.add_argument("--calibrate-method", choices=CALIBRATION_METHODS.keys(), default="minmax")
    # x86 ONNX Runtime only fuses QDQ Conv into QLinearConv for uint8 activations (U8S8).
    parser.add_argument("--activation-type", choices=["qint8", "quint8"], default="quint8")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output model")

    args = parser.parse_args()

    source_path = Path(args.source).resolve()
    activation_tag = "s8" if args.activation_type == "qint8" else "u8"
    output_path = Path(args.output) if args.output else PROJECT_ROOT / "models" / f"selected_int8_{activation_tag}_{args.variant}.onnx"
    output_path = output_path.resolve()

    if output_path == source_path:
        parser.error("Output must differ from the source model")

    if output_path.exists() and not args.force:
        raise FileExistsError(f"{output_path} exists. Pass --force to overwrite.")

    source_hash = verify_source(source_path)
    calibration_paths = list_calibration_images(CALIBRATION_DIR)

    activation_type = QuantType.QInt8 if args.activation_type == "qint8" else QuantType.QUInt8

    print(f"Variant: {args.variant} ({VARIANTS[args.variant]})")
    print(f"Calibration images: {len(calibration_paths)} from {CALIBRATION_DIR.relative_to(PROJECT_ROOT).as_posix()}")

    start = time.perf_counter()

    excluded, qdq_counts = quantize(
        source_path,
        output_path,
        args.variant,
        calibration_paths,
        CALIBRATION_METHODS[args.calibrate_method],
        activation_type,
        QuantType.QInt8,
    )

    seconds = time.perf_counter() - start

    # Confirm the output loads and runs on the CPU provider.
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    reader = ImageCalibrationReader(calibration_paths[:1], input_name=session.get_inputs()[0].name)
    output = session.run(None, reader.get_next())[0]

    entry = {
        "variant": args.variant,
        "description": VARIANTS[args.variant],
        "source_onnx": source_path.relative_to(PROJECT_ROOT).as_posix() if source_path.is_relative_to(PROJECT_ROOT) else str(source_path),
        "source_sha256": source_hash,
        "source_bytes": source_path.stat().st_size,
        "output_onnx": output_path.relative_to(PROJECT_ROOT).as_posix() if output_path.is_relative_to(PROJECT_ROOT) else str(output_path),
        "output_sha256": sha256_file(output_path),
        "output_bytes": output_path.stat().st_size,
        "output_shape": list(output.shape),
        "settings": {
            "pre_process": "onnxruntime.quantization.shape_inference.quant_pre_process (defaults)",
            "quant_format": "QDQ",
            "per_channel": True,
            "activation_type": args.activation_type,
            "weight_type": "qint8",
            "calibrate_method": args.calibrate_method,
            "preprocessing": "edgesight.onnx_detector.preprocess(frame, 640)",
            "imgsz": IMAGE_SIZE,
        },
        "nodes_excluded": excluded,
        "qdq_nodes": qdq_counts,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
        "quantization_seconds": round(seconds, 2),
    }

    calibration = {
        "split": "train",
        "directory": CALIBRATION_DIR.relative_to(PROJECT_ROOT).as_posix(),
        "count": len(calibration_paths),
        "images": {path.name: sha256_file(path) for path in calibration_paths},
    }

    update_manifest(args.manifest, output_path.stem, entry, calibration)

    print(f"Excluded nodes: {len(excluded)}")
    print(f"QDQ nodes: {qdq_counts}")
    print(f"Size: {entry['source_bytes'] / 1e6:.2f} MB -> {entry['output_bytes'] / 1e6:.2f} MB")
    print(f"Saved: {output_path}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
