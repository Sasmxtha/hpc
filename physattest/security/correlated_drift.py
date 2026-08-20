"""
Correlated Drift Analysis for PhysAttest (Server Security).

Detects server code tampering by analysing the correlation structure
of observer residuals.

Key insight:
  - Sensor attack → isolated spike → residuals UNCORRELATED
  - Code tampering → systematic drift → residuals CORRELATED

One number (τ = mean off-diagonal |correlation|) distinguishes them.

Also monitors eigenvalue concentration of the residual covariance.
When the largest eigenvalue dominates (λ₁/Σλ ≈ 1), all residuals
are moving in one direction — the hallmark of systematic tampering.
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from collections import deque


class IntegrityStatus(Enum):
    CLEAN = "clean"
    SENSOR_ATTACK = "sensor_attack"
    CODE_TAMPERING = "code_tampering"
    UNCERTAIN = "uncertain"


@dataclass
class DriftAnalysisResult:
    status: IntegrityStatus
    tau: float                    # mean off-diagonal |correlation|
    eigenvalue_ratio: float       # λ₁ / Σλ — dominance of first principal component
    correlation_matrix: np.ndarray
    eigenvalues: np.ndarray
    details: str

    # Per-sensor isolation scores (high = that sensor is the outlier)
    isolation_scores: np.ndarray


@dataclass
class DriftConfig:
    window_size: int = 60         # seconds of residual history
    tau_threshold: float = 0.4    # above this → code tampering
    eigen_threshold: float = 0.7  # λ₁/Σλ above this → single systematic cause
    min_samples: int = 20         # need this many samples before analysis


class CorrelatedDriftAnalyser:
    """
    Analyses correlation structure of observer residuals to detect
    server code tampering vs sensor attacks.

    The physical world is the integrity checker: if the server's
    physics equations are modified, the predictions diverge from
    reality for ALL sensors simultaneously, producing correlated
    residual drift.
    """

    def __init__(self, n_sensors: int, config: Optional[DriftConfig] = None):
        self.n_sensors = n_sensors
        self.config = config or DriftConfig()
        self.residual_history: deque[np.ndarray] = deque(
            maxlen=self.config.window_size
        )
        self.analysis_log: list[DriftAnalysisResult] = []

    def add_residual(self, r: np.ndarray):
        """Add one residual vector (from observer) to the sliding window."""
        self.residual_history.append(r.copy())

    def analyse(self) -> DriftAnalysisResult:
        """
        Analyse the correlation structure of recent residuals.

        Steps:
        1. Build residual matrix R (window × n_sensors)
        2. Compute correlation matrix
        3. Compute τ = mean off-diagonal |correlation|
        4. Compute eigenvalue concentration
        5. Compute per-sensor isolation scores
        6. Classify: clean / sensor_attack / code_tampering
        """
        n = len(self.residual_history)

        if n < self.config.min_samples:
            result = DriftAnalysisResult(
                status=IntegrityStatus.UNCERTAIN,
                tau=0.0,
                eigenvalue_ratio=0.0,
                correlation_matrix=np.eye(self.n_sensors),
                eigenvalues=np.ones(self.n_sensors),
                details=f"Only {n}/{self.config.min_samples} samples — waiting",
                isolation_scores=np.zeros(self.n_sensors),
            )
            self.analysis_log.append(result)
            return result

        # --- Step 1: Build residual matrix ---
        R_mat = np.array(list(self.residual_history))  # (window, n_sensors)

        # --- Step 2: Correlation matrix ---
        # Handle constant columns (zero variance)
        stds = np.std(R_mat, axis=0)
        active = stds > 1e-10
        n_active = np.sum(active)

        if n_active < 2:
            result = DriftAnalysisResult(
                status=IntegrityStatus.CLEAN,
                tau=0.0,
                eigenvalue_ratio=0.0,
                correlation_matrix=np.eye(self.n_sensors),
                eigenvalues=np.ones(self.n_sensors),
                details="Insufficient variance in residuals",
                isolation_scores=np.zeros(self.n_sensors),
            )
            self.analysis_log.append(result)
            return result

        # Compute correlation only on active sensors
        R_active = R_mat[:, active]
        corr_active = np.corrcoef(R_active.T)  # (n_active, n_active)

        # Expand back to full size
        corr_full = np.eye(self.n_sensors)
        active_idx = np.where(active)[0]
        for i_local, i_global in enumerate(active_idx):
            for j_local, j_global in enumerate(active_idx):
                corr_full[i_global, j_global] = corr_active[i_local, j_local]

        # --- Step 3: τ = mean off-diagonal |correlation| ---
        mask = ~np.eye(self.n_sensors, dtype=bool)
        tau = float(np.mean(np.abs(corr_full[mask])))

        # --- Step 4: Eigenvalue concentration ---
        cov = np.cov(R_mat.T)
        eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
        eigenvalues = np.maximum(eigenvalues, 0)  # numerical stability
        total_var = np.sum(eigenvalues)
        if total_var > 1e-12:
            eigen_ratio = float(eigenvalues[0] / total_var)
        else:
            eigen_ratio = 0.0

        # --- Step 5: Per-sensor isolation scores ---
        # A sensor under isolated attack has HIGH individual variance
        # but LOW correlation with others
        isolation_scores = np.zeros(self.n_sensors)
        for i in range(self.n_sensors):
            if not active[i]:
                continue
            # Mean absolute correlation with other sensors
            others = [abs(corr_full[i, j]) for j in range(self.n_sensors) if j != i and active[j]]
            mean_corr_with_others = np.mean(others) if others else 0
            # Isolation = high variance + low correlation
            normalised_var = stds[i] / (np.mean(stds[active]) + 1e-10)
            isolation_scores[i] = normalised_var * (1 - mean_corr_with_others)

        # --- Step 6: Classification ---
        if tau > self.config.tau_threshold and eigen_ratio > self.config.eigen_threshold:
            status = IntegrityStatus.CODE_TAMPERING
            details = (
                f"CORRELATED DRIFT DETECTED. τ={tau:.3f} (>{self.config.tau_threshold}), "
                f"λ₁/Σλ={eigen_ratio:.3f} (>{self.config.eigen_threshold}). "
                f"All residuals moving together → server code likely modified."
            )
        elif tau > self.config.tau_threshold:
            status = IntegrityStatus.CODE_TAMPERING
            details = (
                f"HIGH CORRELATION. τ={tau:.3f} (>{self.config.tau_threshold}). "
                f"Possible code tampering. λ₁/Σλ={eigen_ratio:.3f}."
            )
        elif np.max(isolation_scores) > 2.0:
            isolated_sensor = int(np.argmax(isolation_scores))
            status = IntegrityStatus.SENSOR_ATTACK
            details = (
                f"ISOLATED ANOMALY on sensor {isolated_sensor}. "
                f"τ={tau:.3f} (low → not correlated). "
                f"Isolation score={isolation_scores[isolated_sensor]:.2f}. "
                f"Consistent with sensor-level attack."
            )
        else:
            status = IntegrityStatus.CLEAN
            details = f"Normal. τ={tau:.3f}, λ₁/Σλ={eigen_ratio:.3f}."

        result = DriftAnalysisResult(
            status=status,
            tau=tau,
            eigenvalue_ratio=eigen_ratio,
            correlation_matrix=corr_full,
            eigenvalues=eigenvalues,
            details=details,
            isolation_scores=isolation_scores,
        )
        self.analysis_log.append(result)
        return result


def demo_correlated_drift():
    """
    Demo: distinguish sensor attack from server code tampering.

    Scenario 1: Normal operation → τ ≈ 0
    Scenario 2: Sensor 2 attacked → τ low, sensor 2 isolated
    Scenario 3: Server code tampered → τ high, eigenvalue concentrated
    """
    np.random.seed(42)
    n_sensors = 6

    print("=" * 65)
    print("  PhysAttest Correlated Drift Analysis — Demo")
    print("  Sensor attack: τ ≈ 0  |  Code tampering: τ >> 0")
    print("=" * 65)

    # --- Scenario 1: Normal operation ---
    print("\n── Scenario 1: Normal Operation ──")
    analyser = CorrelatedDriftAnalyser(n_sensors)
    for t in range(60):
        # Independent noise on each sensor (no correlation)
        r = np.random.randn(n_sensors) * 0.01
        analyser.add_residual(r)
    result = analyser.analyse()
    print(f"  Status:     {result.status.value}")
    print(f"  τ (corr):   {result.tau:.4f}")
    print(f"  λ₁/Σλ:     {result.eigenvalue_ratio:.4f}")
    print(f"  Detail:     {result.details}")

    # --- Scenario 2: Sensor attack (sensor 2 spoofed) ---
    print("\n── Scenario 2: Sensor 2 Under Attack ──")
    analyser2 = CorrelatedDriftAnalyser(n_sensors)
    for t in range(60):
        r = np.random.randn(n_sensors) * 0.01
        # Sensor 2 has a large residual (spoofed data)
        r[2] = np.random.randn() * 0.5 + 2.0
        analyser2.add_residual(r)
    result = analyser2.analyse()
    print(f"  Status:     {result.status.value}")
    print(f"  τ (corr):   {result.tau:.4f}")
    print(f"  λ₁/Σλ:     {result.eigenvalue_ratio:.4f}")
    print(f"  Isolation:  {np.round(result.isolation_scores, 2)}")
    print(f"  Detail:     {result.details}")

    # --- Scenario 3: Server code tampering ---
    print("\n── Scenario 3: Server Code Tampered ──")
    print("  (Physics equations modified → ALL residuals drift together)")
    analyser3 = CorrelatedDriftAnalyser(n_sensors)
    for t in range(60):
        # Systematic bias from wrong physics + independent noise
        # All sensors drift in the same direction with a shared component
        shared_drift = 0.3 * np.sin(0.1 * t) + 0.1 * t / 60
        r = np.random.randn(n_sensors) * 0.01
        r += shared_drift  # same drift added to every sensor
        # Each sensor also gets a sensor-specific systematic error
        r += np.array([0.05, 0.03, 0.04, 0.06, 0.02, 0.05]) * t / 60
        analyser3.add_residual(r)
    result = analyser3.analyse()
    print(f"  Status:     {result.status.value}")
    print(f"  τ (corr):   {result.tau:.4f}")
    print(f"  λ₁/Σλ:     {result.eigenvalue_ratio:.4f}")
    print(f"  Isolation:  {np.round(result.isolation_scores, 2)}")
    print(f"  Detail:     {result.details}")

    # --- Scenario 4: Subtle code tampering (small bias) ---
    print("\n── Scenario 4: Subtle Code Tampering (small systematic bias) ──")
    analyser4 = CorrelatedDriftAnalyser(n_sensors)
    for t in range(60):
        r = np.random.randn(n_sensors) * 0.01
        # Small shared drift — harder to detect
        shared_drift = 0.02 * t / 60
        r += shared_drift
        analyser4.add_residual(r)
    result = analyser4.analyse()
    print(f"  Status:     {result.status.value}")
    print(f"  τ (corr):   {result.tau:.4f}")
    print(f"  λ₁/Σλ:     {result.eigenvalue_ratio:.4f}")
    print(f"  Detail:     {result.details}")

    # --- Summary ---
    print("\n" + "=" * 65)
    print("  Detection principle:")
    print("  • Sensor attack: one sensor's residual spikes independently")
    print("    → low τ, high isolation score on one sensor")
    print("  • Code tampering: ALL residuals drift in the same direction")
    print("    → high τ, high λ₁/Σλ, no isolation")
    print("  • The physical world is the integrity checker")
    print("=" * 65)


if __name__ == "__main__":
    demo_correlated_drift()
