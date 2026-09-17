"""
Run PhysAttest pipeline on real SWaT dataset and produce paper-ready results.

Usage:
    python -m physattest.experiments.run_real_swat
"""

import sys
import os
import io
import json
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline import PhysAttestPipeline


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "swat")
    data_dir = os.path.abspath(data_dir)

    if not os.path.exists(os.path.join(data_dir, "SWaT_Dataset_Normal_v1.csv")):
        print("Real SWaT data not found. Download from iTrust SUTD.")
        return

    pipe = PhysAttestPipeline(threshold_sigma=3.0)
    pipe.calibrate(data_dir, max_rows=86400, warmup=3600)
    results = pipe.run(data_dir)
    pipe.print_results(results)

    output_path = os.path.join(data_dir, "real_swat_results.json")
    serializable = {
        "overall": results["overall"],
        "thresholds": results["thresholds"],
        "attack_metrics": [
            {
                "label": m.attack_label,
                "start": m.attack_start,
                "end": m.attack_end,
                "duration": m.duration,
                "tp": m.tp, "fp": m.fp, "fn": m.fn, "tn": m.tn,
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "detection_delay": m.detection_delay,
            }
            for m in results["attack_metrics"]
        ],
    }
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
