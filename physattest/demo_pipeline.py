"""
Demo: Full PhysAttest pipeline on synthetic SWaT data.

1. Calibrate thresholds on normal data
2. Run detection on all 10 attack scenarios
3. Compute precision, recall, F1, detection delay
4. Generate results table and visualization

Run: python demo_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline import PhysAttestPipeline


def run_demo():
    print("=" * 80)
    print("PhysAttest Full Pipeline — End-to-End Detection Demo")
    print("=" * 80)

    pipe = PhysAttestPipeline(threshold_sigma=3.0)

    # Step 1: Calibrate on normal data
    print("\n[Step 1] Calibrating on normal data...")
    pipe.calibrate("data/SWaT_Normal_Synthetic.csv", max_rows=20000)

    # Step 2: Run on attack data
    print("\n[Step 2] Running detection on attack data (10 scenarios)...")
    results = pipe.run("data/SWaT_Attack_Synthetic.csv")

    # Step 3: Print results
    pipe.print_results(results)

    # Step 4: Visualization
    print("\n[Step 4] Generating visualizations...")
    _plot_detection_timeline(results)
    _plot_per_attack_bars(results)
    _plot_residual_detail(results)

    print(f"\n{'='*80}")
    print("Pipeline complete. Generated files:")
    print("  pipeline_timeline.png     — residual timeline across all attacks")
    print("  pipeline_per_attack.png   — recall/precision/F1 per attack")
    print("  pipeline_residual_detail.png — per-domain residual breakdown")
    print(f"{'='*80}")


def _plot_detection_timeline(results: dict):
    """Full timeline: residuals + attack labels + detections."""
    detections = results["detections"]
    n = len(detections)
    time_h = np.arange(n) / 3600

    r_combined = [d.residual_combined for d in detections]
    true_labels = [d.true_label for d in detections]
    detected = [d.is_detected for d in detections]

    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)

    # Panel 1: Combined residual
    axes[0].plot(time_h, r_combined, linewidth=0.3, color="steelblue")
    axes[0].axhline(y=results["thresholds"]["combined"], color="red",
                     linestyle="--", linewidth=1, label=f"Threshold")
    axes[0].set_ylabel("Combined Residual")
    axes[0].set_title("PhysAttest Detection Timeline")
    axes[0].legend(loc="upper right")

    # Panel 2: Ground truth
    axes[1].fill_between(time_h, 0, true_labels, alpha=0.5, color="red", label="Attack")
    axes[1].set_ylabel("Ground Truth")
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["Normal", "Attack"])
    axes[1].legend(loc="upper right")

    # Panel 3: Detections
    axes[2].fill_between(time_h, 0, detected, alpha=0.5, color="orange", label="Detected")
    axes[2].set_ylabel("Detection")
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["Normal", "Alert"])
    axes[2].set_xlabel("Time (hours)")
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("pipeline_timeline.png", dpi=150, bbox_inches="tight")
    print("  Saved pipeline_timeline.png")


def _plot_per_attack_bars(results: dict):
    """Bar chart: recall, precision, F1 per attack."""
    metrics = results["attack_metrics"]
    labels = [m.attack_label.replace("Attack (", "").replace(")", "") for m in metrics]
    recalls = [m.recall for m in metrics]
    precisions = [m.precision for m in metrics]
    f1s = [m.f1 for m in metrics]
    delays = [m.detection_delay if m.detection_delay >= 0 else -1 for m in metrics]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    x = np.arange(len(labels))
    width = 0.25

    # Top: Recall / Precision / F1
    axes[0].bar(x - width, recalls, width, label="Recall", color="#2ecc71")
    axes[0].bar(x, precisions, width, label="Precision", color="#3498db")
    axes[0].bar(x + width, f1s, width, label="F1", color="#e74c3c")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Per-Attack Detection Performance")
    axes[0].legend()
    axes[0].set_ylim(0, 1.1)

    # Bottom: Detection delay
    colors = ["#2ecc71" if d >= 0 and d <= 5 else "#f39c12" if d >= 0 else "#e74c3c" for d in delays]
    axes[1].bar(x, [max(d, 0) for d in delays], color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Detection Delay (seconds)")
    axes[1].set_title("Detection Delay (green ≤5s, orange >5s, red = missed)")

    plt.tight_layout()
    plt.savefig("pipeline_per_attack.png", dpi=150, bbox_inches="tight")
    print("  Saved pipeline_per_attack.png")


def _plot_residual_detail(results: dict):
    """Per-domain residuals for first two attacks to show which domain catches what."""
    detections = results["detections"]

    # Find first attack period
    attack_periods = results["attack_metrics"]
    if len(attack_periods) < 2:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    for panel, atk_metric in enumerate(attack_periods[:2]):
        ax = axes[panel]
        start = atk_metric.attack_start - 500   # 500s context before
        end = atk_metric.attack_end + 200        # 200s context after
        start = max(0, start - 1)   # adjust for 0-based detections
        end = min(len(detections) - 1, end - 1)

        subset = detections[start:end]
        time = np.arange(len(subset))

        ax.plot(time, [d.residual_physics for d in subset],
                linewidth=0.8, label="Physics", color="#e74c3c")
        ax.plot(time, [d.residual_chemistry for d in subset],
                linewidth=0.8, label="Chemistry", color="#3498db")
        ax.plot(time, [d.residual_math for d in subset],
                linewidth=0.8, label="Math", color="#2ecc71")

        # Mark attack period
        for d in subset:
            if d.true_label:
                atk_start_rel = d.timestamp - (start + 1)
                ax.axvspan(atk_start_rel, atk_start_rel + 1,
                           alpha=0.05, color="red")

        ax.set_ylabel("Residual")
        label_clean = atk_metric.attack_label.replace("Attack (", "").replace(")", "")
        ax.set_title(f"{label_clean}: Per-Domain Residuals (red shading = attack period)")
        ax.legend(loc="upper right")

    axes[1].set_xlabel("Time (seconds relative to context window)")
    plt.tight_layout()
    plt.savefig("pipeline_residual_detail.png", dpi=150, bbox_inches="tight")
    print("  Saved pipeline_residual_detail.png")


if __name__ == "__main__":
    run_demo()
