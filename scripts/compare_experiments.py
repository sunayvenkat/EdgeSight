


import argparse
import json
from pathlib import Path


METRICS = ("precision", "recall", "mAP50", "mAP50_95")


def calculate_changes(baseline, candidate):
    # Calculate metric changes on a common split

    if baseline["split"] != candidate["split"]:
        raise ValueError("Experiments use different dataset splits")

    if baseline["images"] != candidate["images"]:
        raise ValueError("Experiments have different image counts")

    return {
        key: round(candidate[key] - baseline[key], 4)
        for key in METRICS
    }


def per_class_changes(baseline, candidate):
    # Compare per-class mAP50 values

    old = {
        item["class_id"]: item
        for item in baseline["per_class"]
    }

    new = {
        item["class_id"]: item
        for item in candidate["per_class"]
    }

    results = []

    for class_id in sorted(set(old) & set(new)):
        before = old[class_id]
        after = new[class_id]

        results.append({
            "class": before["class_name"],
            "before": before["mAP50"],
            "after": after["mAP50"],
            "delta": round(
                after["mAP50"] - before["mAP50"],
                4,
            ),
        })

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)

    args = parser.parse_args()

    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))

    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))

    changes = calculate_changes(baseline, candidate)

    print("\nEdgeSight Experiment Comparison")
    print("--------------------------------")
    print("Metric             V1       V2      Change")

    for metric in METRICS:
        print(
            f"{metric:<15}"
            f"{baseline[metric]:>8.4f}"
            f"{candidate[metric]:>9.4f}"
            f"{changes[metric]:>+11.4f}"
        )

    print("\nPer-class mAP50 changes")

    for item in per_class_changes(baseline, candidate):
        print(
            f"{item['class']:<10}"
            f"{item['before']:.3f} -> "
            f"{item['after']:.3f} "
            f"({item['delta']:+.3f})"
        )


if __name__ == "__main__":
    main()
