"""Check that this machine can run EdgeSight: packages, release model files, the ONNX session, and optionally a camera (see docs/raspberry_pi.md)."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_ROOT = PROJECT_ROOT / "models"

PACKAGES = ("numpy", "cv2", "onnxruntime", "torch", "ultralytics")


class Skip(Exception):
    """An optional check that does not apply on this machine."""


def check_packages():
    versions = []

    for name in PACKAGES:
        module = importlib.import_module(name)
        versions.append(f"{name} {getattr(module, '__version__', '?')}")

    return f"Python {platform.python_version()} on {platform.machine()}; " + ", ".join(versions)


def check_release_files(models=MODELS_ROOT):
    """The release files exist and match the hashes recorded when they were released."""
    from edgesight.detector import load_class_confidence, sha256_file

    selection = json.loads((models / "selection.json").read_text(encoding="utf-8"))
    runtime = json.loads((models / "runtime_config.json").read_text(encoding="utf-8"))

    for name, expected in (("selected.onnx", runtime["onnx_sha256"]), ("selected.pt", selection["checkpoint_sha256"])):
        path = models / name

        if not path.is_file():
            raise FileNotFoundError(f"models/{name} is missing; copy it from the development machine")

        if sha256_file(path) != expected:
            raise ValueError(f"models/{name} does not match the hash recorded at release; copy it again")

    cutoffs = load_class_confidence(models / "selected.onnx", models / "class_confidence.json")

    return f"{selection['release']}, cutoffs {cutoffs or 'single cutoff'}"


def check_int8(models=MODELS_ROOT):
    """INT8 models from the release, if copied, match their recorded hashes (optional; Phase 8.2)."""
    from edgesight.detector import sha256_file

    path = models / "class_confidence.json"
    derived = json.loads(path.read_text(encoding="utf-8")).get("derived_sha256", {}) if path.is_file() else {}
    present = {name: digest for name, digest in derived.items() if (models / f"{name}.onnx").is_file()}

    if not present:
        raise Skip("no INT8 models copied (only needed for the 8.2 comparison)")

    wrong = sorted(name for name, digest in present.items() if sha256_file(models / f"{name}.onnx") != digest)

    if wrong:
        raise ValueError(f"hash mismatch: {', '.join(wrong)}")

    return f"{len(present)} of {len(derived)} present: {', '.join(sorted(present))}"


def check_inference():
    """The app's ONNX session loads (it checks runtime_config.json) and detects on a blank frame."""
    import numpy as np

    from edgesight import config
    from edgesight.detector import OnnxDetector, create_session, load_class_confidence

    detector = OnnxDetector(session=create_session(), class_confidence=load_class_confidence(config.ONNX_MODEL_PATH))
    frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
    detector.detect(frame)

    start = time.perf_counter()
    detector.detect(frame)

    return f"one frame in {(time.perf_counter() - start) * 1000:.0f} ms with {config.ONNX_INTRA_OP_THREADS} threads"


def check_camera(index):
    import cv2

    capture = cv2.VideoCapture(index)

    try:
        ok, frame = capture.read()
    finally:
        capture.release()

    if not ok or frame is None:
        raise RuntimeError(f"camera {index} did not return a frame")

    return f"camera {index}: {frame.shape[1]}x{frame.shape[0]}"


def run(checks):
    """Run (name, function) checks; return (name, status, detail) rows with status PASS, FAIL, or SKIP."""
    rows = []

    for name, check in checks:
        try:
            rows.append((name, "PASS", check()))
        except Skip as reason:
            rows.append((name, "SKIP", str(reason)))
        except Exception as error:
            rows.append((name, "FAIL", f"{type(error).__name__}: {error}"))

    return rows


def main():
    parser = argparse.ArgumentParser(description="Check that this machine can run EdgeSight.")
    parser.add_argument("--camera", type=int, default=None, help="Also check that this camera index returns a frame")
    args = parser.parse_args()

    checks = [("packages", check_packages), ("release files", check_release_files), ("INT8 models", check_int8), ("ONNX inference", check_inference)]

    if args.camera is not None:
        checks.append(("camera", lambda: check_camera(args.camera)))

    rows = run(checks)

    for name, status, detail in rows:
        print(f"{status:4}  {name}: {detail}")

    failed = [name for name, status, _ in rows if status == "FAIL"]
    print("Ready." if not failed else f"Not ready: {', '.join(failed)}. Then run python -m pytest -q.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
