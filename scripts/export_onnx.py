

import json
from pathlib import Path

import onnx
import onnxruntime as ort
from ultralytics import YOLO

from scripts.benchmark_inference import (MODEL_PATH, verify_model, sha256_file)

ONNX_PATH = MODEL_PATH.with_suffix(".onnx")
MANIFEST_PATH = MODEL_PATH.parent / ("onnx_manifest.json")

def export_onnx():
    # Export and validate the selected model
    source_hash = verify_model()

    if ONNX_PATH.exists():
        raise FileExistsError(f"{ONNX_PATH} already exists.")

    print("Loading selected PyTorch checkpoint... ")

    model = YOLO(str(MODEL_PATH))

    print("Exporting to ONNX... ")

    exported = Path(model.export(format="onnx", imgsz=640, batch=1, dynamic=False, simplify=True, device="cpu")).resolve()

    if exported != ONNX_PATH.resolve():
        raise RuntimeError(f"Unexpected export location: {exported}")

    print("Validating ONNX graph... ")

    onnx.checker.check_model(str(exported))

    session = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])

    input_info = session.get_inputs()[0]

    if input_info.shape != [1, 3, 640, 640]:
        raise ValueError(f"Unexpected input shape: {input_info.shape}")

    manifest = {
        "source_checkpoint": str(MODEL_PATH),
        "source_sha256": source_hash,
        "onnx_checkpoint": str(exported),
        "onnx_sha256": sha256_file(exported),
        "source_bytes": MODEL_PATH.stat().st_size,
        "onnx_bytes": exported.stat().st_size,
        "input_name": input_info.name,
        "input_shape": input_info.shape,
        "providers": session.get_providers(),
        "dynamic": False,
        "batch": 1,
        "imgsz": 640,
        "simplify": True,
    }

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nONNX export successful!")
    print(f"Model: {exported}")
    print(f"Input shape: {input_info.shape}")
    print(f"Providers: {session.get_providers()}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    export_onnx()



