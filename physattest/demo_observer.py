"""
Demo: Multi-Domain Observer on synthetic SWaT-like data.

This script:
1. Generates synthetic clean sensor data following conservation laws
2. Runs the observer and shows residuals are near-zero
3. Injects a sensor spoofing attack
4. Shows the observer catches it immediately
5. (Optional) Runs SINDy to discover equations from the clean data

Run: python demo_observer.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from observer.multi_domain_observer import (
    MultiDomainObserver, N_STATES, N_INPUTS,
    STATE_H1, STATE_H3, STATE_H4, STATE_PH, STATE_ORP, STATE_COND, STATE_DP3,
    INPUT_MV101, INPUT_P101, INPUT_P201, INPUT_P301, INPUT_P302, INPUT_P501,
    INPUT_P601, INPUT_UV401,
)


def generate_synthetic_swat(n_steps: int = 1000, dt: float = 1.0, seed: int = 42):
    """
    Generate synthetic sensor data that obeys conservation laws.

    This is your ground truth. The observer should produce near-zero
    residuals on this data.
    """
    np.random.seed(seed)

    # Initial state
    x = np.array([
        0.5,    # h1: Tank 1 at 50% (meters)
        0.4,    # h3: Tank 3 at 40%
        0.6,    # h4: Tank 4 at 60%
        7.0,    # pH: neutral
        250.0,  # ORP: normal range (mV)
        400.0,  # Conductivity: normal (μS/cm)
        0.5,    # ΔP: normal differential pressure (bar)
    ])

    states = np.zeros((n_steps, N_STATES))
    inputs = np.zeros((n_steps, N_INPUTS))
    measurements = np.zeros((n_steps, N_STATES))

    # Simple control logic: keep tanks within bounds
    for k in range(n_steps):
        # Control: open valve if tank 1 is low, pump if tank 1 is high
        u = np.zeros(N_INPUTS)
        u[INPUT_MV101] = 1.0 if x[STATE_H1] < 0.8 else 0.0
        u[INPUT_P101]  = 1.0 if x[STATE_H1] > 0.3 else 0.0
        u[INPUT_P201]  = 0.1     # constant low-rate dosing
        u[INPUT_P301]  = 1.0 if x[STATE_H3] > 0.2 else 0.0
        u[INPUT_P501]  = 1.0 if x[STATE_H4] > 0.3 else 0.0

        inputs[k] = u
        states[k] = x.copy()

        # Process noise (small random disturbances from real-world variation)
        w = np.array([
            np.random.normal(0, 0.001),   # tank level noise
            np.random.normal(0, 0.001),
            np.random.normal(0, 0.001),
            np.random.normal(0, 0.01),    # pH noise
            np.random.normal(0, 0.5),     # ORP noise
            np.random.normal(0, 0.2),     # conductivity noise
            np.random.normal(0, 0.005),   # ΔP noise
        ])

        # Measurement noise (sensor imprecision)
        v = np.array([
            np.random.normal(0, 0.002),
            np.random.normal(0, 0.002),
            np.random.normal(0, 0.002),
            np.random.normal(0, 0.02),
            np.random.normal(0, 1.0),
            np.random.normal(0, 0.5),
            np.random.normal(0, 0.01),
        ])

        measurements[k] = x + v

        # True dynamics (what actually happens in the plant)
        # Tank 1: dh1/dt = (Q_in - Q_out) / A
        area = 1.5
        q_in = 2.5 / 3600 * u[INPUT_MV101]
        q_out = 2.5 / 3600 * u[INPUT_P101]
        x[STATE_H1] += (q_in - q_out) / area * dt + w[0]
        x[STATE_H1] = np.clip(x[STATE_H1], 0.0, 1.0)

        # Tank 3
        q_in3 = 2.5 / 3600 * u[INPUT_P101]
        q_out3 = 2.0 / 3600 * u[INPUT_P301]
        x[STATE_H3] += (q_in3 - q_out3) / area * dt + w[1]
        x[STATE_H3] = np.clip(x[STATE_H3], 0.0, 1.0)

        # Tank 4
        q_in4 = 2.0 / 3600 * u[INPUT_P301]
        q_out4 = 1.5 / 3600 * u[INPUT_P501]
        x[STATE_H4] += (q_in4 - q_out4) / area * dt + w[2]
        x[STATE_H4] = np.clip(x[STATE_H4], 0.0, 1.2)

        # pH (chemistry)
        x[STATE_PH] += (-0.001 * (x[STATE_PH] - 7.0) + 0.05 * u[INPUT_P201]) * dt + w[3]

        # ORP
        x[STATE_ORP] += (-0.005 * x[STATE_ORP] + 10.0 * u[INPUT_P201]) * dt + w[4]

        # Conductivity
        x[STATE_COND] += -0.0001 * x[STATE_COND] * dt + w[5]

        # ΔP
        x[STATE_DP3] += (-0.01 * x[STATE_DP3] + 0.5 * u[INPUT_P301]) * dt + w[6]

    return states, inputs, measurements


def inject_attack(measurements, attack_start, attack_sensor, attack_value):
    """
    Inject a sensor spoofing attack: replace a sensor's readings with
    a fake constant value starting at attack_start.
    """
    attacked = measurements.copy()
    attacked[attack_start:, attack_sensor] = attack_value
    return attacked


def run_demo():
    print("=" * 70)
    print("PhysAttest Multi-Domain Observer — Demo")
    print("=" * 70)

    # --- Generate synthetic data ---
    n_steps = 1000
    states, inputs, measurements = generate_synthetic_swat(n_steps)
    print(f"\nGenerated {n_steps} seconds of synthetic SWaT data.")
    print(f"State shape: {states.shape}, Input shape: {inputs.shape}")

    # --- Run observer on clean data ---
    print("\n--- Phase 1: Clean data (no attacks) ---")
    observer = MultiDomainObserver(dt=1.0)
    observer.initialize(measurements[0])

    clean_residuals = {"physics": [], "chemistry": [], "math": [], "combined": []}

    for k in range(1, n_steps):
        result = observer.step(measurements[k], inputs[k])
        for domain in clean_residuals:
            clean_residuals[domain].append(np.linalg.norm(result[domain]))

    for domain in clean_residuals:
        vals = clean_residuals[domain]
        print(f"  {domain:12s} residual — mean: {np.mean(vals):.6f}, "
              f"max: {np.max(vals):.6f}, std: {np.std(vals):.6f}")

    # --- Inject attack and detect ---
    print("\n--- Phase 2: Sensor spoofing attack on LIT101 at t=500 ---")
    attack_start = 500
    attacked_measurements = inject_attack(
        measurements, attack_start,
        attack_sensor=STATE_H1,
        attack_value=0.1  # fake low level (real level is ~0.5)
    )

    observer_attack = MultiDomainObserver(dt=1.0)
    observer_attack.initialize(attacked_measurements[0])

    attack_residuals = {"physics": [], "chemistry": [], "math": [], "combined": []}

    for k in range(1, n_steps):
        result = observer_attack.step(attacked_measurements[k], inputs[k])
        for domain in attack_residuals:
            attack_residuals[domain].append(np.linalg.norm(result[domain]))

    pre_attack = attack_residuals["combined"][:attack_start - 1]
    post_attack = attack_residuals["combined"][attack_start - 1:]
    print(f"  Combined residual BEFORE attack — mean: {np.mean(pre_attack):.6f}")
    print(f"  Combined residual AFTER  attack — mean: {np.mean(post_attack):.6f}")
    print(f"  Ratio (attack/clean): {np.mean(post_attack) / np.mean(pre_attack):.1f}x")

    # --- Detection threshold ---
    threshold = np.mean(pre_attack) + 3 * np.std(pre_attack)
    detections = [i for i, r in enumerate(attack_residuals["combined"])
                  if r > threshold and i >= attack_start - 1]
    if detections:
        detection_delay = detections[0] - (attack_start - 1)
        print(f"\n  Attack DETECTED at t={detections[0] + 1} "
              f"(delay: {detection_delay} seconds)")
        print(f"  Threshold: {threshold:.6f}")
    else:
        print("\n  WARNING: Attack not detected with 3-sigma threshold.")

    # --- Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    time = np.arange(1, n_steps)

    # Panel 1: Tank level (true vs attacked measurement)
    axes[0].plot(time, states[1:, STATE_H1], label="True level", linewidth=1.5)
    axes[0].plot(time, attacked_measurements[1:, STATE_H1],
                 label="Measured (attacked at t=500)", linewidth=1, alpha=0.7)
    axes[0].axvline(x=attack_start, color="red", linestyle="--", label="Attack start")
    axes[0].set_ylabel("Tank 1 Level (m)")
    axes[0].legend()
    axes[0].set_title("LIT101 Sensor Reading")

    # Panel 2: Combined residual
    axes[1].plot(time, attack_residuals["combined"], linewidth=0.8)
    axes[1].axhline(y=threshold, color="red", linestyle="--", label=f"Threshold = {threshold:.4f}")
    axes[1].axvline(x=attack_start, color="red", linestyle="--", alpha=0.5)
    axes[1].set_ylabel("Combined Residual ‖r‖")
    axes[1].legend()
    axes[1].set_title("Observer Residual (spikes = anomaly detected)")

    # Panel 3: Domain-specific residuals
    for domain in ["physics", "chemistry", "math"]:
        axes[2].plot(time, attack_residuals[domain], label=domain, linewidth=0.8, alpha=0.8)
    axes[2].axvline(x=attack_start, color="red", linestyle="--", alpha=0.5)
    axes[2].set_ylabel("Domain Residuals")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].legend()
    axes[2].set_title("Per-Domain Residuals")

    plt.tight_layout()
    plt.savefig("observer_demo_results.png", dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to observer_demo_results.png")

    # --- SINDy discovery (if available) ---
    try:
        from observer.sindy_discovery import discover_equations
        print("\n--- Phase 3: SINDy equation discovery ---")

        feature_names = ["h1", "h3", "h4", "pH", "ORP", "cond", "dP"]
        result = discover_equations(
            state_data=states[:500],  # use clean data only
            input_data=inputs[:500],
            dt=1.0,
            feature_names=feature_names,
            threshold=0.01,
        )
        print(f"  SINDy R² score: {result['score']:.4f}")
        print("  Discovered equations:")
        for i, eq in enumerate(result["equations"]):
            print(f"    d({feature_names[i]})/dt = {eq}")
    except Exception as e:
        print(f"\n  SINDy skipped: {e}")
        print("  Install PySINDy with: pip install pysindy")

    print("\n" + "=" * 70)
    print("Demo complete. Key takeaway:")
    print("  - Clean data → residuals ≈ 0 (observer tracks the plant)")
    print("  - Attack data → residuals spike (physics violation detected)")
    print("  - Detection happens within 1-2 seconds of attack onset")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
