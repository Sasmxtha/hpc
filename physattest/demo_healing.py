"""
Demo: Self-Healing Function — attack reconstruction, fault calibration,
and quarantine decision with Theorem 3 accuracy bounds.

Run: python demo_healing.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from graph.coupling_graph import build_swat_coupling_graph
from observer.healing import SelfHealingFunction, CompromiseType, HealingAction


def run_demo():
    print("=" * 70)
    print("PhysAttest Self-Healing Function — Demo")
    print("=" * 70)

    # Build coupling graph
    G = build_swat_coupling_graph()
    healer = SelfHealingFunction(G)

    # Simulate current sensor readings (normal operation)
    true_readings = {
        "LIT101": 0.500, "FIT101": 2.50, "MV101": 1.0, "P101": 1.0, "P102": 0.0,
        "AIT201": 7.05,  "AIT202": 260.0, "AIT203": 410.0, "FIT201": 2.45,
        "P201": 0.1, "P203": 0.0,
        "LIT301": 0.420, "FIT301": 2.10, "DPIT301": 0.55, "P301": 1.0, "P302": 0.0,
        "LIT401": 0.580, "FIT401": 2.05, "AIT401": 120.0, "UV401": 1.0,
        "AIT501": 6.95,  "AIT502": 255.0, "AIT503": 45.0,
        "FIT501": 2.00,  "FIT502": 1.40, "FIT503": 0.40, "FIT504": 0.20,
        "PIT501": 150.0, "PIT502": 10.0, "PIT503": 145.0,
        "P501": 1.0, "P502": 0.0,
        "FIT601": 1.80, "P601": 1.0, "P602": 0.0,
    }

    # ===================================================================
    # SCENARIO 1: Attack on LIT101 (tank level spoofed to 0.1)
    # ===================================================================
    print("\n--- Scenario 1: Attack on LIT101 ---")
    print(f"True level: {true_readings['LIT101']:.3f} m")
    print(f"Attacker reports: 0.100 m (spoofed low to prevent pump activation)")

    attacked_readings = true_readings.copy()
    attacked_readings["LIT101"] = 0.100  # attacker's fake value

    result = healer.heal(
        sensor="LIT101",
        compromise_type=CompromiseType.ATTACK,
        current_readings=attacked_readings,
        observer_estimate=0.495,  # observer's prediction from Component 1
    )

    print(f"\n  Action:         {result.action.value}")
    print(f"  Original:       {result.original_value:.3f} m (spoofed)")
    print(f"  Reconstructed:  {result.healed_value:.3f} m")
    print(f"  True value:     {true_readings['LIT101']:.3f} m")
    print(f"  Recon error:    {abs(result.healed_value - true_readings['LIT101']):.4f} m")
    print(f"  Accuracy bound: ε = {result.accuracy_bound:.4f} m (Theorem 3)")
    print(f"  Confidence:     {result.confidence:.1%}")
    print(f"  Contributors:   {result.contributors}")
    print(f"  Explanation:    {result.explanation}")

    # Verify Theorem 3: actual error should be ≤ ε
    actual_error = abs(result.healed_value - true_readings["LIT101"])
    print(f"\n  Theorem 3 check: actual error ({actual_error:.4f}) "
          f"{'≤' if actual_error <= result.accuracy_bound else '>'} "
          f"bound ({result.accuracy_bound:.4f}) → "
          f"{'HOLDS' if actual_error <= result.accuracy_bound else 'VIOLATED'}")

    # ===================================================================
    # SCENARIO 2: Fault on AIT201 (pH sensor drifting +0.3)
    # ===================================================================
    print("\n--- Scenario 2: Fault on AIT201 (pH sensor drift) ---")
    healer.mark_honest("LIT101")  # restore LIT101 first

    print(f"True pH: {true_readings['AIT201']:.2f}")
    print(f"Drifted reading: 7.35 (sensor aging, +0.3 bias)")

    faulty_readings = true_readings.copy()
    faulty_readings["AIT201"] = 7.35  # drifted high

    result_fault = healer.heal(
        sensor="AIT201",
        compromise_type=CompromiseType.FAULT,
        current_readings=faulty_readings,
    )

    print(f"\n  Action:         {result_fault.action.value}")
    print(f"  Original:       {result_fault.original_value:.3f} (drifted)")
    print(f"  Corrected:      {result_fault.healed_value:.3f}")
    print(f"  True value:     {true_readings['AIT201']:.3f}")
    print(f"  Correction err: {abs(result_fault.healed_value - true_readings['AIT201']):.4f}")
    print(f"  Accuracy bound: ε = {result_fault.accuracy_bound:.4f}")
    print(f"  Confidence:     {result_fault.confidence:.1%}")
    print(f"  Explanation:    {result_fault.explanation}")

    # ===================================================================
    # SCENARIO 3: Quarantine — too many compromised neighbours
    # ===================================================================
    print("\n--- Scenario 3: Quarantine (insufficient honest neighbours) ---")

    # Compromise all of UV401's neighbours
    healer.compromised.clear()
    healer.compromised["AIT401"] = CompromiseType.ATTACK

    result_q = healer.heal(
        sensor="UV401",
        compromise_type=CompromiseType.ATTACK,
        current_readings=true_readings,
    )

    print(f"\n  Action:         {result_q.action.value}")
    print(f"  Accuracy bound: ε = {result_q.accuracy_bound}")
    print(f"  Confidence:     {result_q.confidence:.1%}")
    print(f"  Explanation:    {result_q.explanation}")

    # ===================================================================
    # SCENARIO 4: Coordinated attack — 3 sensors compromised
    # ===================================================================
    print("\n--- Scenario 4: Coordinated attack on Stage 5 RO ---")
    healer.compromised.clear()

    # Attacker compromises FIT501, FIT502, FIT503
    coordinated_readings = true_readings.copy()
    coordinated_readings["FIT501"] = 3.50  # inflated inlet
    coordinated_readings["FIT502"] = 2.80  # inflated permeate
    coordinated_readings["FIT503"] = 0.50  # slightly inflated concentrate

    # Heal each one
    for sensor in ["FIT501", "FIT502", "FIT503"]:
        result_c = healer.heal(
            sensor=sensor,
            compromise_type=CompromiseType.ATTACK,
            current_readings=coordinated_readings,
        )
        print(f"\n  {sensor}:")
        print(f"    Action:       {result_c.action.value}")
        print(f"    Spoofed:      {coordinated_readings[sensor]:.2f}")
        print(f"    Reconstructed:{result_c.healed_value:.2f}")
        print(f"    True:         {true_readings[sensor]:.2f}")
        print(f"    ε:            {result_c.accuracy_bound:.4f}")
        print(f"    Confidence:   {result_c.confidence:.1%}")

    # Check: FIT504 (not attacked) should help reconstruct the others
    # via flow balance FIT501 = FIT502 + FIT503 + FIT504
    q_balance = true_readings["FIT502"] + true_readings["FIT503"] + true_readings["FIT504"]
    print(f"\n  Flow balance check:")
    print(f"    FIT502 + FIT503 + FIT504 = {q_balance:.2f} should ≈ FIT501 = {true_readings['FIT501']:.2f}")

    # ===================================================================
    # System health summary
    # ===================================================================
    print(f"\n--- System Health ---")
    health = healer.get_system_health()
    print(f"  Total sensors:  {health['total_sensors']}")
    print(f"  Healthy:        {health['healthy']}")
    print(f"  Compromised:    {health['compromised']} "
          f"({health['attacks']} attacks, {health['faults']} faults)")
    print(f"  Health:         {health['health_percentage']:.1f}%")

    # ===================================================================
    # Visualization: healing over time (simulated drift correction)
    # ===================================================================
    print("\n--- Generating drift correction plot ---")

    healer2 = SelfHealingFunction(G)
    n_steps = 200
    true_ph = 7.05
    drift_rate = 0.003  # pH units per second

    true_vals = []
    measured_vals = []
    corrected_vals = []
    bounds = []

    for t in range(n_steps):
        # True pH is constant
        true_vals.append(true_ph)

        # Measured pH drifts linearly
        drift = drift_rate * t
        measured = true_ph + drift + np.random.normal(0, 0.02)
        measured_vals.append(measured)

        # Build readings
        readings = true_readings.copy()
        readings["AIT201"] = measured
        # AIT501 tracks true pH (honest neighbour)
        readings["AIT501"] = true_ph + np.random.normal(0, 0.02)

        result_t = healer2.heal(
            sensor="AIT201",
            compromise_type=CompromiseType.FAULT,
            current_readings=readings,
        )
        corrected_vals.append(result_t.healed_value)
        bounds.append(result_t.accuracy_bound)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    time = np.arange(n_steps)

    # Panel 1: pH values
    axes[0].plot(time, true_vals, "g-", label="True pH", linewidth=2)
    axes[0].plot(time, measured_vals, "r-", label="Measured (drifting)", linewidth=1, alpha=0.7)
    axes[0].plot(time, corrected_vals, "b-", label="Healed (calibration corrected)", linewidth=1.5)
    axes[0].fill_between(
        time,
        [c - b for c, b in zip(corrected_vals, bounds)],
        [c + b for c, b in zip(corrected_vals, bounds)],
        alpha=0.2, color="blue", label="Accuracy bound ε (Theorem 3)",
    )
    axes[0].set_ylabel("pH")
    axes[0].legend(loc="upper left")
    axes[0].set_title("Self-Healing: Fault Calibration Correction on AIT201")

    # Panel 2: reconstruction error vs bound
    errors = [abs(c - t) for c, t in zip(corrected_vals, true_vals)]
    axes[1].plot(time, errors, "b-", label="Actual error |corrected - true|", linewidth=1)
    axes[1].plot(time, bounds, "r--", label="Theorem 3 bound ε", linewidth=1.5)
    axes[1].set_ylabel("Error")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].legend()
    axes[1].set_title("Theorem 3 Verification: Actual Error vs Bound")

    plt.tight_layout()
    plt.savefig("healing_demo_results.png", dpi=150, bbox_inches="tight")
    print(f"Plot saved to healing_demo_results.png")

    # Check Theorem 3 holds
    violations = sum(1 for e, b in zip(errors, bounds) if e > b)
    print(f"\nTheorem 3 violations: {violations}/{n_steps} timesteps "
          f"({violations/n_steps:.1%})")

    print(f"\n{'='*70}")
    print("Summary:")
    print("  - Attack: full reconstruction from honest neighbours via coupling graph")
    print("  - Fault: calibration correction preserves sensor responsiveness")
    print("  - Quarantine: triggered when accuracy bound exceeds safety margin")
    print("  - Theorem 3: actual error stays within the provable bound")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_demo()
