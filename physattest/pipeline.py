"""
Integrated PhysAttest Pipeline.

Connects all components into a single detection loop:
    Data Loader → Observer → Detector → Healer → Metrics

Usage:
    from pipeline import PhysAttestPipeline
    pipe = PhysAttestPipeline()
    results = pipe.run("data/SWaT_Attack_Synthetic.csv")
    pipe.print_results(results)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from data.loader import SWaTLoader
from observer.multi_domain_observer import (
    MultiDomainObserver, N_STATES, N_INPUTS,
    STATE_H1, STATE_H3, STATE_H4, STATE_PH, STATE_ORP, STATE_COND, STATE_DP3,
    INPUT_MV101, INPUT_P101, INPUT_P201, INPUT_P301, INPUT_P302, INPUT_P501,
    INPUT_P601, INPUT_UV401,
)
from graph.coupling_graph import build_swat_coupling_graph
from observer.healing import SelfHealingFunction, CompromiseType


# -----------------------------------------------------------------------
# Column-to-observer mapping
# -----------------------------------------------------------------------

# Maps SWaT data columns → observer state indices
SENSOR_TO_STATE = {
    "LIT101":  STATE_H1,
    "LIT301":  STATE_H3,
    "LIT401":  STATE_H4,
    "AIT201":  STATE_PH,
    "AIT202":  STATE_ORP,
    "AIT203":  STATE_COND,
    "DPIT301": STATE_DP3,
}

# Maps SWaT data columns → observer input indices
ACTUATOR_TO_INPUT = {
    "MV101":  (INPUT_MV101, 2),   # (index, "on" value — MV101=2 means open)
    "P101":   (INPUT_P101, 2),
    "P201":   (INPUT_P201, 2),
    "P301":   (INPUT_P301, 2),
    "P302":   (INPUT_P302, 2),
    "P501":   (INPUT_P501, 2),
    "P601":   (INPUT_P601, 2),
    "UV401":  (INPUT_UV401, 2),
}

# Normalization: convert data units to observer units
# Observer uses meters for level; SWaT data uses millimeters
UNIT_SCALE = {
    "LIT101": 0.001,    # mm → m
    "LIT301": 0.001,
    "LIT401": 0.001,
    "AIT201": 1.0,      # pH is pH
    "AIT202": 1.0,      # mV is mV
    "AIT203": 1.0,      # µS/cm
    "DPIT301": 0.01,    # kPa → bar (approximate)
}


@dataclass
class DetectionResult:
    """Result for a single timestep."""
    timestamp: int
    residual_physics: float
    residual_chemistry: float
    residual_math: float
    residual_combined: float
    is_detected: bool
    true_label: bool        # ground truth: is this actually an attack?
    healed_sensors: List[str] = field(default_factory=list)


@dataclass
class AttackMetrics:
    """Performance metrics for one attack period."""
    attack_label: str
    attack_start: int
    attack_end: int
    duration: int
    tp: int = 0             # true positives (attack detected during attack)
    fp: int = 0             # false positives (detected but no attack)
    fn: int = 0             # false negatives (attack but not detected)
    tn: int = 0             # true negatives (no attack, not detected)
    detection_delay: int = -1   # seconds from attack start to first detection (-1 = missed)
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


class PhysAttestPipeline:
    """
    End-to-end detection pipeline.

    Steps each second:
      1. Read sensor data from the loader
      2. Map columns to observer state/input vectors
      3. Run observer → get residuals
      4. Apply threshold detection (3-sigma from training)
      5. If detected: flag sensor, run healer
      6. Record result for metrics
    """

    def __init__(self, threshold_sigma: float = 3.0):
        self.threshold_sigma = threshold_sigma
        self.observer = MultiDomainObserver(dt=1.0)
        self.coupling_graph = build_swat_coupling_graph()
        self.healer = SelfHealingFunction(self.coupling_graph)

        # Per-domain thresholds (on norm of residual vector)
        self.thresholds = {
            "physics": None,
            "chemistry": None,
            "math": None,
            "combined": None,
        }

        # Per-state thresholds (on individual residual components)
        # This is the primary detector — handles model mismatch gracefully
        self.state_thresholds = np.zeros(N_STATES)
        self.state_means = np.zeros(N_STATES)
        self.state_stds = np.zeros(N_STATES)

    def calibrate(self, normal_data_path: str, max_rows: int = 86400,
                  warmup: int = 3600):
        """
        Learn detection thresholds from normal (attack-free) data.

        Runs the observer on normal data to measure baseline residual
        distribution. Skips a warmup period so the observer reaches
        steady state before measuring thresholds.

        Threshold = mean + threshold_sigma × std.
        """
        print("Calibrating thresholds from normal data...")
        loader = SWaTLoader(normal_data_path, max_rows=max_rows)
        df = loader.df

        sensor_names = loader.get_sensor_names()
        actuator_names = loader.get_actuator_names()

        # Initialize observer with first reading
        y0 = self._extract_state(df.iloc[0], sensor_names)
        self.observer.initialize(y0)

        # Collect residuals on normal data (skip warmup)
        residuals = {"physics": [], "chemistry": [], "math": [], "combined": []}
        per_state_residuals = []
        n_rows = len(df)
        for i in range(1, n_rows):
            y = self._extract_state(df.iloc[i], sensor_names)
            u = self._extract_input(df.iloc[i], actuator_names)
            result = self.observer.step(y, u)

            if i >= warmup:
                for domain in residuals:
                    residuals[domain].append(np.linalg.norm(result[domain]))
                per_state_residuals.append(np.abs(result["combined"]))

        # Set domain-level thresholds (on norm)
        for domain in self.thresholds:
            vals = np.array(residuals[domain])
            mean = np.mean(vals)
            std = np.std(vals)
            self.thresholds[domain] = mean + self.threshold_sigma * std

        # Set per-state thresholds (on individual components)
        # This is more robust: a consistently large chemistry residual
        # is "normal" — only CHANGES from the baseline trigger detection
        per_state_residuals = np.array(per_state_residuals)  # (n_samples, N_STATES)
        self.state_means = np.mean(per_state_residuals, axis=0)
        self.state_stds = np.std(per_state_residuals, axis=0)
        self.state_stds = np.maximum(self.state_stds, 1e-6)  # avoid division by zero
        self.state_thresholds = self.state_means + self.threshold_sigma * self.state_stds

        print(f"  Calibrated on {n_rows - warmup} samples (after {warmup}s warmup).")
        print(f"  Domain thresholds ({self.threshold_sigma}σ):")
        for domain, thresh in self.thresholds.items():
            print(f"    {domain:12s}: {thresh:.6f}")

        state_names = ["h1", "h3", "h4", "pH", "ORP", "cond", "ΔP"]
        print(f"  Per-state thresholds:")

        # Determine which states have a calibrated model (useful for detection)
        # vs constant offset (model mismatch — not useful for z-score detection).
        # A state where std/mean < 0.05 has a big constant residual with little
        # variation — the observer model is systematically wrong. That state
        # still runs in the observer but its z-score isn't used for detection.
        self.active_detection_states = []
        for i, name in enumerate(state_names):
            ratio = self.state_stds[i] / max(abs(self.state_means[i]), 1e-9)
            is_active = ratio > 0.05
            self.active_detection_states.append(is_active)
            status = "ACTIVE" if is_active else "SKIP (model mismatch)"
            print(f"    {name:6s}: mean={self.state_means[i]:.6f}  "
                  f"std={self.state_stds[i]:.6f}  "
                  f"std/mean={ratio:.3f}  [{status}]")

        n_active = sum(self.active_detection_states)
        print(f"  Active detection states: {n_active}/{N_STATES}")

        return self.thresholds

    def run(self, attack_data_path: str, max_rows: Optional[int] = None) -> Dict:
        """
        Run the full pipeline on attack data.

        Returns a dictionary with:
            'detections': list of DetectionResult per timestep
            'attack_metrics': list of AttackMetrics per attack period
            'overall': overall precision, recall, F1
        """
        if self.thresholds["combined"] is None:
            raise RuntimeError("Must call calibrate() before run()")

        loader = SWaTLoader(attack_data_path, max_rows=max_rows)
        df = loader.df
        sensor_names = loader.get_sensor_names()
        actuator_names = loader.get_actuator_names()

        # Get attack periods for per-attack metrics
        attack_periods = loader.get_attack_periods()

        # Reset observer
        self.observer = MultiDomainObserver(dt=1.0)
        y0 = self._extract_state(df.iloc[0], sensor_names)
        self.observer.initialize(y0)

        # Reset healer
        self.healer = SelfHealingFunction(self.coupling_graph)

        # Warmup: let the observer converge before detecting
        warmup = 3600
        n_rows = len(df)
        for i in range(1, min(warmup, n_rows)):
            y = self._extract_state(df.iloc[i], sensor_names)
            u = self._extract_input(df.iloc[i], actuator_names)
            self.observer.step(y, u)

        # Run detection after warmup
        detections: List[DetectionResult] = []
        ALERT_SIGMA = 4.0      # per-state z-score threshold

        # Detection is purely instantaneous z-score based.
        # No Kalman gain zeroing here — that feedback loop requires
        # the 3-way classifier (Component 6, Member 2) to distinguish
        # genuine attacks from noise before committing to a flag.
        # Without the classifier, flagging cascades through the
        # coupled Kalman gain and creates runaway false positives.

        for i in range(max(1, warmup), n_rows):
            row = df.iloc[i]
            y = self._extract_state(row, sensor_names)
            u = self._extract_input(row, actuator_names)

            result = self.observer.step(y, u)

            r_phys = np.linalg.norm(result["physics"])
            r_chem = np.linalg.norm(result["chemistry"])
            r_math = np.linalg.norm(result["math"])
            r_comb = np.linalg.norm(result["combined"])

            # Per-state z-score: flag if any calibrated state deviates
            abs_residual = np.abs(result["combined"])
            z_scores = (abs_residual - self.state_means) / self.state_stds

            is_detected = any(
                z_scores[s] > ALERT_SIGMA
                for s in range(N_STATES)
                if self.active_detection_states[s]
            )

            true_label = bool(row.get("is_attack", False))

            healed = []
            if is_detected:
                idx_to_sensor = {v: k for k, v in SENSOR_TO_STATE.items()}
                healed = [idx_to_sensor[s] for s in flagged_states if s in idx_to_sensor]

            detections.append(DetectionResult(
                timestamp=i,
                residual_physics=r_phys,
                residual_chemistry=r_chem,
                residual_math=r_math,
                residual_combined=r_comb,
                is_detected=is_detected,
                true_label=true_label,
                healed_sensors=healed,
            ))

        # Compute metrics
        attack_metrics = self._compute_attack_metrics(detections, attack_periods)
        overall = self._compute_overall_metrics(detections)

        return {
            "detections": detections,
            "attack_metrics": attack_metrics,
            "overall": overall,
            "thresholds": dict(self.thresholds),
        }

    def _extract_state(self, row, sensor_names: List[str]) -> np.ndarray:
        """Map a data row to the observer's 7-state vector."""
        y = np.zeros(N_STATES)
        for col, state_idx in SENSOR_TO_STATE.items():
            if col in sensor_names:
                val = float(row.get(col, 0))
                scale = UNIT_SCALE.get(col, 1.0)
                y[state_idx] = val * scale
        return y

    def _extract_input(self, row, actuator_names: List[str]) -> np.ndarray:
        """Map a data row to the observer's 8-input vector."""
        u = np.zeros(N_INPUTS)
        for col, (input_idx, on_value) in ACTUATOR_TO_INPUT.items():
            if col in actuator_names:
                val = float(row.get(col, 0))
                u[input_idx] = 1.0 if val == on_value else 0.0
        return u

    def _identify_flagged_sensor(self, result: Dict) -> Optional[str]:
        """
        Find which sensor contributed most to the residual spike.
        Returns the sensor name with the largest absolute residual.
        """
        combined = result["combined"]
        max_idx = np.argmax(np.abs(combined))

        idx_to_sensor = {v: k for k, v in SENSOR_TO_STATE.items()}
        return idx_to_sensor.get(max_idx)

    def _compute_attack_metrics(
        self,
        detections: List[DetectionResult],
        attack_periods: List[Dict],
    ) -> List[AttackMetrics]:
        """Compute precision, recall, F1 and detection delay per attack."""
        metrics_list = []

        for period in attack_periods:
            start = period["start"] - 1   # offset by 1 (we skip row 0)
            end = period["end"] - 1
            label = period["label"]
            duration = period["duration"]

            m = AttackMetrics(
                attack_label=label,
                attack_start=period["start"],
                attack_end=period["end"],
                duration=duration,
            )

            first_detection = -1

            for det in detections:
                t = det.timestamp
                in_attack = start <= t <= end

                if in_attack and det.is_detected:
                    m.tp += 1
                    if first_detection < 0:
                        first_detection = t - start
                elif in_attack and not det.is_detected:
                    m.fn += 1
                elif not in_attack and det.is_detected:
                    # Only count FP near this attack (within 60s buffer)
                    if abs(t - start) < 60 or abs(t - end) < 60:
                        m.fp += 1
                # TN counted in overall metrics

            m.detection_delay = first_detection

            if m.tp + m.fp > 0:
                m.precision = m.tp / (m.tp + m.fp)
            if m.tp + m.fn > 0:
                m.recall = m.tp / (m.tp + m.fn)
            if m.precision + m.recall > 0:
                m.f1 = 2 * m.precision * m.recall / (m.precision + m.recall)

            metrics_list.append(m)

        return metrics_list

    def _compute_overall_metrics(self, detections: List[DetectionResult]) -> Dict:
        """Compute overall detection performance across all data."""
        tp = sum(1 for d in detections if d.is_detected and d.true_label)
        fp = sum(1 for d in detections if d.is_detected and not d.true_label)
        fn = sum(1 for d in detections if not d.is_detected and d.true_label)
        tn = sum(1 for d in detections if not d.is_detected and not d.true_label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(detections) if detections else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        return {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "fpr": fpr,
        }

    @staticmethod
    def print_results(results: Dict):
        """Pretty-print pipeline results."""
        print("\n" + "=" * 80)
        print("PhysAttest Detection Results")
        print("=" * 80)

        # Overall metrics
        o = results["overall"]
        print(f"\n--- Overall Performance ---")
        print(f"  Precision:  {o['precision']:.4f}  (of detections, how many were real attacks)")
        print(f"  Recall:     {o['recall']:.4f}  (of real attacks, how many were detected)")
        print(f"  F1 Score:   {o['f1']:.4f}")
        print(f"  Accuracy:   {o['accuracy']:.4f}")
        print(f"  FPR:        {o['fpr']:.4f}  (false positive rate)")
        print(f"  TP={o['tp']}  FP={o['fp']}  FN={o['fn']}  TN={o['tn']}")

        # Per-attack table
        print(f"\n--- Per-Attack Metrics ---")
        print(f"  {'Attack':<20s} {'Recall':>8s} {'Precision':>10s} "
              f"{'F1':>8s} {'Delay':>8s} {'TP':>6s} {'FN':>6s}")
        print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")

        for m in results["attack_metrics"]:
            delay_str = f"{m.detection_delay}s" if m.detection_delay >= 0 else "MISSED"
            print(f"  {m.attack_label:<20s} {m.recall:>8.3f} {m.precision:>10.3f} "
                  f"{m.f1:>8.3f} {delay_str:>8s} {m.tp:>6d} {m.fn:>6d}")

        # Thresholds
        print(f"\n--- Detection Thresholds ---")
        for domain, thresh in results["thresholds"].items():
            print(f"  {domain:12s}: {thresh:.6f}")
