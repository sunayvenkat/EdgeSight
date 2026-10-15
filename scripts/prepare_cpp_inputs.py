


import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from scripts.benchmark_inference import (IMAGE_DIR, load_images, sha256_file)
from scripts.tune_onnx import (ONNX_PATH, PRESETS, make_options, prepare_input, verify_export)
from scripts.train_model import PROJECT_ROOT


CPP_RUNS = PROJECT_ROOT / "runs" / "cpp"
CONFIG_PATH = PROJECT_ROOT / "models" / "runtime_config.json"


def validate_preset(config):
    preset = config["recommended_preset"]

    if preset not in PRESETS:
        raise ValueError(f"Unknown optimized preset: {preset}")

    expected = PRESETS[preset]

    actual = config["configuration"]

    if (actual["intra_op_threads"] != expected[0] or actual["graph_optimization"] != expected[1] or actual["spinning"] != expected[2]):
        raise ValueError(
            "Saved runtime configuration differs "
            "from the Step 14 preset"
        )

    return preset


def main():
    verify_export()

    manifest_path = CPP_RUNS / "reference.json"

    if manifest_path.exists():
        raise FileExistsError(
            "Reference data already exists. "
            "Review it before regenerating."
        )

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    preset = validate_preset(config)

    images = load_images(IMAGE_DIR)

    # Shape: N x 1 x 3 x 640 x 640.
    tensors = np.stack([prepare_input(frame) for _, frame in images])

    if tensors.shape[1:] != (1, 3, 640, 640):
        raise ValueError("Unexpected input tensor shape")

    # Explicitly serialize contiguous little-endian FP32.
    tensors = np.ascontiguousarray(tensors, dtype="<f4")

    CPP_RUNS.mkdir(parents=True, exist_ok=True)

    input_path = CPP_RUNS / "inputs.bin"
    tensors.tofile(input_path)

    session = ort.InferenceSession(str(ONNX_PATH), sess_options=make_options(preset), providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: tensors[0]})

    output_metadata = []

    for index, array in enumerate(outputs):
        if array.dtype != np.float32:
            raise ValueError(f"Output {index} is not FP32")

        path = CPP_RUNS / f"reference_output_{index}.bin"

        np.ascontiguousarray(array, dtype="<f4").tofile(path)

        output_metadata.append({
            "file": path.name,
            "shape": list(array.shape),
            "elements": int(array.size),
        })

    manifest = {
        "preset": preset,
        "configuration": config["configuration"],
        "onnx_sha256": sha256_file(ONNX_PATH),
        "onnxruntime_version": ort.__version__,
        "images": [
            path.name for path, _ in images
        ],
        "frames": len(images),
        "input_shape": list(tensors.shape),
        "input_file": input_path.name,
        "input_sha256": sha256_file(input_path),
        "outputs": output_metadata,
    }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("C++ benchmark inputs prepared!")
    print(f"Preset: {preset}")
    print(f"Frames: {len(images)}")
    print(f"Input: {input_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
