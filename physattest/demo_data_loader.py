"""
Demo: Data loader on synthetic SWaT data.
Tests that loading, splitting, windowing, and statistics all work.
Then runs the observer on real (synthetic) data format.

Run: python demo_data_loader.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.loader import SWaTLoader


def run_demo():
    print("=" * 70)
    print("PhysAttest Data Loader — Demo")
    print("=" * 70)

    # --- Load normal data ---
    print("\n--- Loading normal data ---")
    loader_normal = SWaTLoader("data/SWaT_Normal_Synthetic.csv")
    print(loader_normal.info())

    # --- Load attack data ---
    print("\n--- Loading attack data ---")
    loader_attack = SWaTLoader("data/SWaT_Attack_Synthetic.csv")
    print(loader_attack.info())

    # --- Attack periods ---
    periods = loader_attack.get_attack_periods()
    print(f"\n--- Attack Periods ({len(periods)} found) ---")
    for p in periods:
        print(f"  [{p['start']:>7d} - {p['end']:>7d}]  "
              f"duration={p['duration']:>5d}s ({p['duration']/60:.0f} min)  "
              f"label={p['label']}")

    # --- Statistics from normal data ---
    print("\n--- Sensor Statistics (normal data) ---")
    stats = loader_normal.compute_statistics(normal_only=True)
    print(stats.to_string())

    # --- Get arrays ---
    sensors, actuators, labels = loader_attack.get_arrays()
    print(f"\n--- Array shapes ---")
    print(f"  Sensors:   {sensors.shape}")
    print(f"  Actuators: {actuators.shape}")
    print(f"  Labels:    {labels.shape} ({labels.sum()} attack rows)")

    # --- Sliding windows ---
    print(f"\n--- Sliding windows (60-second, step=60) ---")
    n_windows = 0
    n_attack_windows = 0
    for window, window_labels, start_idx in loader_attack.sliding_windows(window_size=60, step=60):
        n_windows += 1
        if window_labels.any():
            n_attack_windows += 1
    print(f"  Total windows: {n_windows}")
    print(f"  Attack windows: {n_attack_windows}")
    print(f"  Normal windows: {n_windows - n_attack_windows}")

    # --- Verify conservation laws hold in synthetic data ---
    print("\n--- Conservation Law Verification (first 1000 rows) ---")
    df = loader_normal.df.head(1000)

    # Mass conservation Tank 1: A × dLIT101/dt ≈ FIT101 - Q_P101
    lit101 = df["LIT101"].values
    fit101 = df["FIT101"].values
    p101 = df["P101"].values
    dh_dt = np.diff(lit101)  # mm/s
    q_in = fit101[:-1] / 3600 * 1000   # m³/h → mm/s (via m→mm and h→s)
    q_out_expected = 2.5 / 3600 * 1000 * (p101[:-1] == 2).astype(float)
    area = 1.5  # m²
    residual = dh_dt - (q_in - q_out_expected) / area
    print(f"  Tank 1 mass conservation residual:")
    print(f"    mean = {np.mean(np.abs(residual)):.4f} mm/s")
    print(f"    max  = {np.max(np.abs(residual)):.4f} mm/s")
    print(f"    (should be near zero — limited by noise and discretization)")

    # Flow balance at RO: FIT501 ≈ FIT502 + FIT503 + FIT504
    fit501 = df["FIT501"].values
    fit502 = df["FIT502"].values
    fit503 = df["FIT503"].values
    fit504 = df["FIT504"].values
    flow_balance = fit501 - (fit502 + fit503 + fit504)
    print(f"\n  RO flow balance (FIT501 - FIT502 - FIT503 - FIT504):")
    print(f"    mean = {np.mean(np.abs(flow_balance)):.4f} m³/h")
    print(f"    max  = {np.max(np.abs(flow_balance)):.4f} m³/h")

    # Conductivity rejection: AIT503 << AIT203
    ait203 = df["AIT203"].values
    ait503 = df["AIT503"].values
    rejection = 1 - (ait503.mean() / ait203.mean())
    print(f"\n  RO conductivity rejection: {rejection:.1%}")
    print(f"    AIT203 (feed) mean = {ait203.mean():.1f} µS/cm")
    print(f"    AIT503 (permeate) mean = {ait503.mean():.1f} µS/cm")

    # --- Plot attack scenario A1 ---
    print("\n--- Plotting attack scenario A1 ---")
    df_atk = loader_attack.df
    # A1 is the first attack scenario (rows 0 to 14400)
    subset = df_atk.iloc[:14400]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    time_h = np.arange(len(subset)) / 3600

    # LIT101 — attacked sensor
    axes[0].plot(time_h, subset["LIT101"], linewidth=0.8)
    axes[0].set_ylabel("LIT101 (mm)")
    axes[0].set_title("Attack A1: LIT101 spoofed to 700mm at t=2h")

    # Highlight attack period
    is_atk = subset["Normal/Attack"].str.contains("Attack", na=False)
    for ax in axes:
        for i in range(len(is_atk)):
            if is_atk.iloc[i] and (i == 0 or not is_atk.iloc[i-1]):
                ax.axvline(x=time_h[i], color="red", linestyle="--", alpha=0.5)

    # MV101 — affected actuator
    axes[1].plot(time_h, subset["MV101"], linewidth=0.8)
    axes[1].set_ylabel("MV101 state")
    axes[1].set_title("Actuator Response (valve closes due to fake high level)")

    # FIT101 — flow consequence
    axes[2].plot(time_h, subset["FIT101"], linewidth=0.8)
    axes[2].set_ylabel("FIT101 (m³/h)")
    axes[2].set_xlabel("Time (hours)")
    axes[2].set_title("Flow Consequence")

    plt.tight_layout()
    plt.savefig("data/attack_A1_demo.png", dpi=150)
    print("Plot saved to data/attack_A1_demo.png")

    print(f"\n{'='*70}")
    print("Data pipeline ready. When you get the real SWaT dataset:")
    print("  1. Place the CSV/XLSX in the data/ folder")
    print("  2. Change the filepath: SWaTLoader('data/SWaT_Dataset_Attack_v0.csv')")
    print("  3. Everything else works identically")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_demo()
