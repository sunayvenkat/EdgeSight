


import argparse
import json
import statistics

from scripts.benchmark_inference import (OUTPUT_ROOT, PROJECT_ROOT)
from scripts.tune_onnx import tuning_file_name

PRESETS = ("default", "intra1", "intra2", "intra4", "intra4_nospin", "intra4_basic")


def choose_best(results):
    """Select the fastest numerically consistent preset."""

    eligible = [
        (name, data)
        for name, data in results.items()
        if data["consistent"]
    ]

    if not eligible:
        raise ValueError("No numerically consistent configurations")

    return min(eligible, key=lambda item: (item[1]["median_ms"], item[1]["p95_ms"], item[0] != "default"))[0]


def load_reports(tag=None):
    """Load and verify three runs per preset."""

    reports = {}

    for preset in PRESETS:
        runs = []

        for number in (1, 2, 3):
            path = OUTPUT_ROOT / tuning_file_name(preset, number, tag)

            data = json.loads(path.read_text(encoding="utf-8"))

            if data["preset"] != preset:
                raise ValueError(f"Incorrect preset in {path}")

            if data["run"] != number:
                raise ValueError(f"Incorrect run number in {path}")

            runs.append(data)

        reports[preset] = runs

    # Ensure all configurations use the same
    # model, images, and benchmark workload.
    reference = reports["default"][0]

    for runs in reports.values():
        for report in runs:
            if (report["onnx_sha256"] != reference["onnx_sha256"] or report["image_hashes"] != reference["image_hashes"]):
                raise ValueError("Benchmark inputs are not identical")

            config = report["configuration"]
            reference_config = reference["configuration"]

            for key in ("imgsz", "batch", "warmup", "iterations"):
                if config[key] != reference_config[key]:
                    raise ValueError(f"Benchmark workload differs: {key}")

    return reports


def summarize_reports(reports):
    """Aggregate performance across repetitions."""

    results = {}

    for preset, runs in reports.items():
        metrics = [run["metrics"] for run in runs]

        results[preset] = {
            "median_ms": round(
                statistics.median(
                    m["median_ms"] for m in metrics
                ), 3
            ),
            "p95_ms": round(
                statistics.median(
                    m["p95_ms"] for m in metrics
                ), 3
            ),
            "throughput_fps": round(
                statistics.median(
                    m["throughput_fps"]
                    for m in metrics
                ), 3
            ),
            "cpu_percent": round(
                statistics.median(
                    m["process_cpu_percent"]
                    for m in metrics
                ), 2
            ),
            "rss_mb": round(
                statistics.median(
                    m["peak_sampled_rss_mb"]
                    for m in metrics
                ), 2
            ),
            "consistent": all(
                run["consistency"]["allclose"]
                for run in runs
            ),
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=None, help="Tag used with scripts/tune_onnx.py --tag (default: v1 file names)")
    args = parser.parse_args()

    reports = load_reports(args.tag)
    results = summarize_reports(reports)

    winner = choose_best(results)
    baseline = results["default"]["median_ms"]
    optimized = results[winner]["median_ms"]

    improvement = ((baseline - optimized) / baseline * 100)

    print("\nEdgeSight CPU Runtime Comparison")
    print("-" * 76)

    print(
        f"{'Preset':<20}"
        f"{'Median ms':>11}"
        f"{'P95 ms':>11}"
        f"{'FPS':>10}"
        f"{'CPU %':>10}"
        f"{'RSS MB':>11}"
    )

    for preset, metric in results.items():
        print(
            f"{preset:<20}"
            f"{metric['median_ms']:>11.2f}"
            f"{metric['p95_ms']:>11.2f}"
            f"{metric['throughput_fps']:>10.2f}"
            f"{metric['cpu_percent']:>10.1f}"
            f"{metric['rss_mb']:>11.1f}"
        )

    print(f"\nRecommended preset: {winner}")
    print(f"Median latency reduction: {improvement:.2f}%")

    selected_config = reports[winner][0]["configuration"]

    recommendation = {
        "recommended_preset": winner,
        "onnx_sha256": (
            reports[winner][0]["onnx_sha256"]
        ),
        "configuration": {
            key: selected_config[key]
            for key in (
                "intra_op_threads",
                "inter_op_threads",
                "execution_mode",
                "graph_optimization",
                "spinning",
                "batch",
                "imgsz",
            )
        },
        "baseline_median_ms": baseline,
        "optimized_median_ms": optimized,
        "latency_reduction_percent": round(
            improvement, 2
        ),
        "all_results": results,
    }

    # Save the recommended configuration for
    # future C++ and embedded inference work.
    output = (PROJECT_ROOT / "models" / "runtime_config.json")

    if output.exists():
        raise FileExistsError("A runtime configuration already exists.")

    output.write_text(json.dumps(recommendation, indent=2), encoding="utf-8")

    print(f"Saved configuration: {output}")


if __name__ == "__main__":
    main()
