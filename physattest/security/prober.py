"""
Active Probing Module for PhysAttest (Component 7).

When coupling between sensors is weak, the observer can't cross-check
against neighbours. Active probing fills the gap by injecting small
random perturbations and checking if sensors respond per physics.

Key insight — fault vs attack:
  - Faulty sensor: still responds to perturbation (imperfectly)
  - Attacked sensor: firmware replaying fake data can't predict the
    random perturbation, so response correlation ≈ 0

Detection bound:
  P(evasion after n probes) = (1 - p_detect)^n → 0 exponentially

All perturbations are CBF-bounded — probing cannot endanger the plant.
"""

import numpy as np
import secrets
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ProbeVerdict(Enum):
    HONEST = "honest"           # ρ > threshold_high — tracks perturbation
    FAULTY = "faulty"           # threshold_low < ρ < threshold_high — partial response
    ATTACKED = "attacked"       # ρ < threshold_low — no response
    INCONCLUSIVE = "inconclusive"


@dataclass
class ProbeResult:
    sensor_id: int
    verdict: ProbeVerdict
    correlation: float          # ρ ∈ [-1, 1]
    expected_response: float    # δy_expected
    actual_response: float      # δy_actual
    n_probes_total: int         # cumulative probes on this sensor
    evasion_probability: float  # (1-p)^n


@dataclass
class ProbeConfig:
    # Perturbation magnitude (fraction of operating range)
    magnitude: float = 0.02

    # Correlation thresholds for classification
    honest_threshold: float = 0.6    # ρ > this → honest
    attack_threshold: float = 0.2    # ρ < this → attacked

    # Minimum probes before declaring a verdict
    min_probes: int = 3

    # Maximum perturbation (CBF safety bound)
    max_perturbation: float = 0.05


class PerturbationDesigner:
    """
    Designs CBF-bounded random perturbations using cryptographic RNG.

    The perturbation δu is:
    1. Random direction — attacker can't predict
    2. Bounded magnitude — CBF-safe
    3. Orthogonal to recent perturbations — maximise information
    """

    def __init__(self, n_actuators: int, config: ProbeConfig):
        self.n_actuators = n_actuators
        self.config = config
        self.recent_directions: list[np.ndarray] = []

    def design(self, target_sensors: list[int], B_matrix: np.ndarray) -> np.ndarray:
        """
        Design a perturbation that maximises response at target sensors.

        Args:
            target_sensors: which sensors to probe
            B_matrix: control input matrix (n_sensors × n_actuators)
                      maps actuator commands to sensor responses

        Returns:
            δu: perturbation vector (n_actuators,)
        """
        n = self.n_actuators

        # Cryptographic random direction — unpredictable
        raw_bytes = secrets.token_bytes(n * 8)
        raw = np.frombuffer(raw_bytes, dtype=np.float64)
        # Map to [-1, 1] range
        direction = (raw % 2.0) - 1.0
        direction = direction / (np.linalg.norm(direction) + 1e-10)

        # Project onto directions that maximise response at target sensors
        # The rows of B corresponding to target sensors tell us which
        # actuator directions produce the strongest sensor response
        if len(target_sensors) > 0 and B_matrix is not None:
            target_rows = B_matrix[target_sensors, :]
            # SVD to find the direction of maximum influence
            if target_rows.shape[0] > 0:
                U, S, Vt = np.linalg.svd(target_rows, full_matrices=False)
                if len(S) > 0 and S[0] > 1e-8:
                    # Mix random direction with optimal direction
                    # (pure optimal would be predictable)
                    optimal = Vt[0]
                    direction = 0.5 * direction + 0.5 * optimal
                    direction = direction / (np.linalg.norm(direction) + 1e-10)

        # Gram-Schmidt against recent perturbations for information diversity
        for prev in self.recent_directions[-3:]:
            proj = np.dot(direction, prev) * prev
            direction = direction - 0.5 * proj  # partial orthogonalisation
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction = direction / norm

        # Scale to safe magnitude
        delta_u = direction * self.config.magnitude

        # Hard clamp to CBF safety bound
        max_mag = self.config.max_perturbation
        if np.max(np.abs(delta_u)) > max_mag:
            delta_u = delta_u * (max_mag / np.max(np.abs(delta_u)))

        self.recent_directions.append(direction.copy())
        if len(self.recent_directions) > 10:
            self.recent_directions.pop(0)

        return delta_u


class ResponseAnalyser:
    """
    Analyses sensor responses to perturbations.

    Computes correlation between expected and actual response,
    accumulates evidence across multiple probes, and classifies
    sensors as honest/faulty/attacked.
    """

    def __init__(self, n_sensors: int, config: ProbeConfig):
        self.n_sensors = n_sensors
        self.config = config

        # Per-sensor probe history
        self.probe_count = np.zeros(n_sensors, dtype=int)
        self.correlation_sum = np.zeros(n_sensors)
        self.correlation_sq_sum = np.zeros(n_sensors)
        self.detection_count = np.zeros(n_sensors, dtype=int)

    def analyse(
        self,
        sensor_id: int,
        delta_u: np.ndarray,
        y_before: np.ndarray,
        y_after: np.ndarray,
        B_matrix: np.ndarray,
        C_matrix: np.ndarray,
    ) -> ProbeResult:
        """
        Analyse one sensor's response to a perturbation.

        Expected response:  δy_expected = C[i] @ B @ δu
        Actual response:    δy_actual = y_after[i] - y_before[i]
        Correlation:        ρ = δy_actual · δy_expected / (|actual| · |expected|)

        Args:
            sensor_id: which sensor to check
            delta_u: the perturbation that was applied
            y_before: sensor readings before perturbation
            y_after: sensor readings after perturbation
            B_matrix: control input matrix
            C_matrix: observation matrix
        """
        # Expected response from physics
        delta_x_expected = B_matrix @ delta_u
        delta_y_expected = C_matrix[sensor_id] @ delta_x_expected
        expected = float(delta_y_expected)

        # Actual response
        actual = float(y_after[sensor_id] - y_before[sensor_id])

        # Correlation (scalar version for single sensor)
        if abs(expected) < 1e-10:
            # Perturbation has no expected effect on this sensor
            rho = 0.0 if abs(actual) > 0.01 else 1.0
        else:
            # Response ratio: actual / expected
            # Honest: ratio ≈ 1.0, Faulty: 0 < ratio < 1, Attacked: ratio ≈ 0
            ratio = actual / expected
            # Convert to correlation-like score in [0, 1]
            # Perfect match → 1.0, opposite → 0.0, no response → 0.0
            rho = max(0.0, 1.0 - abs(1.0 - ratio))

        # Update running statistics
        self.probe_count[sensor_id] += 1
        self.correlation_sum[sensor_id] += rho
        n = self.probe_count[sensor_id]

        # Running average correlation
        avg_rho = self.correlation_sum[sensor_id] / n

        if rho < self.config.attack_threshold:
            self.detection_count[sensor_id] += 1

        # Classification based on average correlation
        if n < self.config.min_probes:
            verdict = ProbeVerdict.INCONCLUSIVE
        elif avg_rho > self.config.honest_threshold:
            verdict = ProbeVerdict.HONEST
        elif avg_rho < self.config.attack_threshold:
            verdict = ProbeVerdict.ATTACKED
        else:
            verdict = ProbeVerdict.FAULTY

        # Evasion probability: (1 - p_detect)^n
        # p_detect estimated from detection rate
        p_detect = self.detection_count[sensor_id] / n if n > 0 else 0
        evasion_prob = (1 - p_detect) ** n if p_detect < 1.0 else 0.0

        return ProbeResult(
            sensor_id=sensor_id,
            verdict=verdict,
            correlation=float(avg_rho),
            expected_response=expected,
            actual_response=actual,
            n_probes_total=int(n),
            evasion_probability=evasion_prob,
        )

    def reset(self, sensor_id: int):
        self.probe_count[sensor_id] = 0
        self.correlation_sum[sensor_id] = 0
        self.correlation_sq_sum[sensor_id] = 0
        self.detection_count[sensor_id] = 0


class ActiveProber:
    """
    Complete active probing system.

    Orchestrates perturbation design, injection, and response analysis.
    Maintains probe history and computes cumulative evasion bounds.

    Usage:
        prober = ActiveProber(n_sensors=6, n_actuators=6)
        delta_u = prober.design_probe([suspect_sensor], B)
        # ... apply delta_u to plant, wait one cycle ...
        results = prober.analyse_response(y_before, y_after, delta_u, B, C)
    """

    def __init__(
        self,
        n_sensors: int,
        n_actuators: int,
        config: Optional[ProbeConfig] = None,
    ):
        self.config = config or ProbeConfig()
        self.designer = PerturbationDesigner(n_actuators, self.config)
        self.analyser = ResponseAnalyser(n_sensors, self.config)
        self.n_sensors = n_sensors
        self.n_actuators = n_actuators
        self.probe_log: list[dict] = []

    def design_probe(
        self,
        target_sensors: list[int],
        B_matrix: np.ndarray,
    ) -> np.ndarray:
        """Design a CBF-bounded random perturbation targeting suspect sensors."""
        return self.designer.design(target_sensors, B_matrix)

    def analyse_response(
        self,
        y_before: np.ndarray,
        y_after: np.ndarray,
        delta_u: np.ndarray,
        B_matrix: np.ndarray,
        C_matrix: np.ndarray,
        target_sensors: Optional[list[int]] = None,
    ) -> list[ProbeResult]:
        """
        Analyse all target sensors' responses to the perturbation.

        Returns a ProbeResult for each target sensor.
        """
        if target_sensors is None:
            target_sensors = list(range(self.n_sensors))

        results = []
        for sid in target_sensors:
            result = self.analyser.analyse(
                sid, delta_u, y_before, y_after, B_matrix, C_matrix
            )
            results.append(result)

        self.probe_log.append({
            "delta_u": delta_u.tolist(),
            "targets": target_sensors,
            "verdicts": {r.sensor_id: r.verdict.value for r in results},
        })

        return results

    def get_cumulative_bounds(self) -> dict[int, float]:
        """
        Get the cumulative evasion probability bound for each sensor.

        After n probes: P(evasion) = (1 - p_detect)^n

        Lower is better (harder for attacker to evade).
        """
        bounds = {}
        for sid in range(self.n_sensors):
            n = int(self.analyser.probe_count[sid])
            if n == 0:
                bounds[sid] = 1.0
                continue
            detections = int(self.analyser.detection_count[sid])
            p = detections / n
            bounds[sid] = (1 - p) ** n if p < 1.0 else 0.0
        return bounds


def demo_active_probing():
    """
    Demo: probe a mix of honest, faulty, and attacked sensors.

    Uses batch probing: each round sends K random perturbations,
    collects responses, computes vector correlation ρ over the batch.
    This gives a reliable statistic per round.

    Shows fault/attack distinction and exponentially decreasing
    evasion probability.
    """
    np.random.seed(42)

    n_sensors = 6
    n_actuators = 6

    B = 0.02 * np.eye(n_sensors, n_actuators)
    C = np.eye(n_sensors)

    SENSOR_TYPES = {0: "honest", 1: "honest", 2: "faulty", 3: "honest", 4: "attacked", 5: "attacked"}
    K = 20  # sub-probes per batch round

    print("=" * 65)
    print("  PhysAttest Active Probing — Demo")
    print("  Sensors: 0,1,3=honest  2=faulty  4,5=attacked")
    print(f"  Each round: {K} random perturbations, vector correlation")
    print("=" * 65)

    # Per-sensor accumulators across rounds
    probe_counts = np.zeros(n_sensors, dtype=int)
    rho_history: dict[int, list[float]] = {i: [] for i in range(n_sensors)}

    designer = PerturbationDesigner(n_actuators, ProbeConfig(magnitude=0.03))

    for round_num in range(1, 11):
        # Collect K perturbation-response pairs
        expected_matrix = np.zeros((K, n_sensors))
        actual_matrix = np.zeros((K, n_sensors))

        for k in range(K):
            delta_u = designer.design(list(range(n_sensors)), B)
            delta_y_true = C @ (B @ delta_u)

            for sid in range(n_sensors):
                expected_matrix[k, sid] = delta_y_true[sid]

                if SENSOR_TYPES[sid] == "honest":
                    actual_matrix[k, sid] = delta_y_true[sid] + np.random.randn() * 0.00005
                elif SENSOR_TYPES[sid] == "faulty":
                    actual_matrix[k, sid] = 0.5 * delta_y_true[sid] + np.random.randn() * 0.0001
                else:  # attacked
                    actual_matrix[k, sid] = np.random.randn() * 0.00005

        # Compute per-sensor vector correlation over the K pairs
        for sid in range(n_sensors):
            exp_vec = expected_matrix[:, sid]
            act_vec = actual_matrix[:, sid]

            exp_norm = np.linalg.norm(exp_vec)
            act_norm = np.linalg.norm(act_vec)
            if exp_norm < 1e-12 or act_norm < 1e-12:
                rho = 0.0
            else:
                rho = float(np.dot(exp_vec, act_vec) / (exp_norm * act_norm))
                rho = max(0.0, rho)

            rho_history[sid].append(rho)
            probe_counts[sid] += K

        # Print selected rounds
        if round_num in [1, 3, 5, 10]:
            print(f"\n── Round {round_num} ({probe_counts[0]} total sub-probes) ──")
            for sid in range(n_sensors):
                avg_rho = np.mean(rho_history[sid])

                if avg_rho > 0.8:
                    verdict = "honest"
                elif avg_rho > 0.3:
                    verdict = "faulty"
                else:
                    verdict = "attacked"

                # Evasion bound: for attacked sensors with ρ≈0,
                # each round is an independent test with p_detect ≈ 1
                n_rounds = len(rho_history[sid])
                low_rho_count = sum(1 for r in rho_history[sid] if r < 0.3)
                p_det = low_rho_count / n_rounds if n_rounds > 0 else 0
                evasion = (1 - p_det) ** n_rounds if p_det < 1 else 0.0

                marker = ""
                if verdict == "attacked":
                    marker = " ◀ ATTACK"
                elif verdict == "faulty":
                    marker = " ◀ FAULT"

                print(
                    f"  Sensor {sid} ({SENSOR_TYPES[sid]:>8s}): "
                    f"ρ={avg_rho:.3f}  verdict={verdict:<8s}  "
                    f"P(evasion)={evasion:.2e}{marker}"
                )

    print(f"\n── Summary ──")
    print(f"  Honest sensors:   ρ ≈ 1.0 (perfect response tracking)")
    print(f"  Faulty sensors:   0 < ρ < 1 (partial response — gain error)")
    print(f"  Attacked sensors: ρ ≈ 0   (no response — firmware replay)")
    print(f"  Evasion drops EXPONENTIALLY: (1-p)^n with n = rounds")
    print("=" * 65)


if __name__ == "__main__":
    demo_active_probing()
