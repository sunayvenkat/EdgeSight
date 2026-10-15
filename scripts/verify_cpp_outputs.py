

import argparse
import json

import numpy as np
import onnxruntime as ort

from scripts.benchmark_inference import sha256_file
from scripts.prepare_cpp_inputs import (CPP_RUNS, CONFIG_PATH)
from scripts.tune_onnx import (ONNX_PATH, verify_export)


def compare_arrays(reference, candidate):
    """Compare raw FP32 tensors with fixed tolerances."""

    if reference.shape != candidate.shape:
        return False, None

    equivalent = bool(np.allclose(reference, candidate, rtol=1e-3, atol=1e-4))

    maximum_error = (
        float(np.max(np.abs(reference - candidate)))
        if reference.size
        else 0.0
    )

    return equivalent, maximum_error


def verify_run(run_number):
    """Verify native C++ outputs against Python reference."""

    verify_export()

    reference_path = CPP_RUNS / "reference.json"

    reference = json.loads(reference_path.read_text(encoding="utf-8"))

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    output_dir = CPP_RUNS / f"native_run{run_number}"

    native = json.loads((output_dir / "benchmark.json").read_text(encoding="utf-8"))

    # Verify the model hasn't changed.
    if (reference["onnx_sha256"] != sha256_file(ONNX_PATH)):
        raise ValueError("ONNX model changed")

    # Verify the serialized input data hasn't changed.
    input_path = CPP_RUNS / reference["input_file"]

    if (reference["input_sha256"] != sha256_file(input_path)):
        raise ValueError("Input data changed")

    # Verify matching runtime versions.
    if (reference["onnxruntime_version"] != ort.__version__):
        raise ValueError("Python ONNX Runtime version changed")

    if native["ort_version"] != ort.__version__:
        raise ValueError("Native and Python ONNX Runtime versions differ")

    # Verify the number of frames is identical.
    if native["frames"] != reference["frames"]:
        raise ValueError("Different number of input images")

    # Verify runtime configurations.
    settings = config["configuration"]

    if (native["threads"] != settings["intra_op_threads"] or native["graph"] != settings["graph_optimization"] or native["spinning"] != settings["spinning"]):
        raise ValueError("Native runtime settings do not match")

    results = []

    # Compare every output tensor.
    for index, metadata in enumerate(reference["outputs"]):
        expected = np.fromfile(CPP_RUNS / metadata["file"], dtype="<f4")

        actual_path = (output_dir / f"output_{index}.bin")

        actual = np.fromfile(actual_path, dtype="<f4")

        if (expected.size != metadata["elements"] or actual.size != metadata["elements"]):
            raise ValueError(f"Unexpected output size for tensor {index}")

        shape = tuple(metadata["shape"])

        equivalent, max_error = compare_arrays(
            expected.reshape(shape),
            actual.reshape(shape),
        )

        results.append({
            "output": index,
            "shape": list(shape),
            "allclose": equivalent,
            "max_abs_error": max_error,
        })

    report = {
        "run": run_number,
        "onnx_sha256": reference["onnx_sha256"],
        "all_outputs_consistent": all(
            item["allclose"]
            for item in results
        ),
        "outputs": results,
    }

    report_path = (output_dir / "verification.json")

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for item in results:
        print(
            f"Output {item['output']}: "
            f"consistent={item['allclose']}, "
            f"max_error={item['max_abs_error']}"
        )

    if not report["all_outputs_consistent"]:
        raise RuntimeError("Python and C++ raw outputs differ")

    print("C++ inference verified successfully!")
    print(f"Report: {report_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run", type=int, default=1)

    args = parser.parse_args()

    if args.run < 1:
        parser.error("Run must be positive")

    verify_run(args.run)


if __name__ == "__main__":
    main()