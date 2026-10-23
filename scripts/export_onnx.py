

import argparse
import json
from pathlib import Path

import onnx
import onnxruntime as ort
from ultralytics import YOLO

from scripts.benchmark_inference import (MODEL_PATH, PROJECT_ROOT, verify_model, sha256_file)

ONNX_PATH = MODEL_PATH.with_suffix(".onnx")
MANIFEST_PATH = MODEL_PATH.parent / ("onnx_manifest.json")

def export_checkpoint(weights_path):
    # Export a checkpoint to a fixed-shape ONNX file beside it, then validate the graph
    weights_path = Path(weights_path)
    onnx_path = weights_path.with_suffix(".onnx")

    if onnx_path.exists():
        raise FileExistsError(f"{onnx_path} already exists.")

    print(f"Loading {weights_path}... ")

    model = YOLO(str(weights_path))

    print("Exporting to ONNX... ")

    exported = Path(model.export(format="onnx", imgsz=640, batch=1, dynamic=False, simplify=True, device="cpu")).resolve()

    if exported != onnx_path.resolve():
        raise RuntimeError(f"Unexpected export location: {exported}")

    print("Validating ONNX graph... ")

    onnx.checker.check_model(str(exported))

    session = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])

    input_info = session.get_inputs()[0]

    if input_info.shape != [1, 3, 640, 640]:
        raise ValueError(f"Unexpected input shape: {input_info.shape}")

    return exported, session, input_info

def export_onnx():
    # Export and validate the selected model
    source_hash = verify_model()

    exported, session, input_info = export_checkpoint(MODEL_PATH)

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

def main():
    parser = argparse.ArgumentParser(description="Export the released model (default) or a candidate checkpoint to ONNX.")

    parser.add_argument("--weights", type=Path, default=None, help="Candidate checkpoint, e.g. runs/train/<name>/weights/best.pt; exported beside it, release files untouched")

    args = parser.parse_args()

    if args.weights is None:
        export_onnx()
        return

    weights = args.weights if args.weights.is_absolute() else PROJECT_ROOT / args.weights

    if not weights.is_file():
        parser.error(f"Checkpoint not found: {weights}")

    exported, _, input_info = export_checkpoint(weights)

    print("\nCandidate ONNX export successful!")
    print(f"Model: {exported}")
    print(f"Input shape: {input_info.shape}")
    print(f"Test it with: python -m scripts.analyze_video <clip> --expect <class> --model {exported}")


if __name__ == "__main__":
    main()
