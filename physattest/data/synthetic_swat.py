"""
Synthetic SWaT dataset generator.

Generates realistic sensor data that follows the same conservation laws
as the real SWaT testbed. Uses exact column names so the data loader
and observer work identically on synthetic and real data.

Includes:
  - Normal operation with realistic control logic
  - 10 attack scenarios mimicking real SWaT attack types
  - Proper sensor noise and actuator dynamics

Usage:
    python -m data.synthetic_swat            # generates CSV files
    python -m data.synthetic_swat --plot     # generates + plots
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from .swat_columns import ALL_COLUMNS, SENSOR_COLUMNS, ACTUATOR_COLUMNS, ALL_STAGES


# -----------------------------------------------------------------------
# Plant parameters (approximate values from SWaT documentation)
# -----------------------------------------------------------------------
TANK_AREA = {1: 1.5, 3: 1.5, 4: 1.5}    # m² cross-section
TANK_MAX  = {1: 800, 3: 1000, 4: 800}    # mm max level

PUMP_FLOW = {  # m³/h when running (state=2)
    "P101": 2.5, "P102": 2.5,
    "P301": 2.0, "P302": 2.0,
    "P401": 2.0, "P402": 2.0,
    "P501": 1.8, "P502": 1.8,
    "P601": 1.5, "P602": 1.5,
}

# Setpoints for tank level control (mm)
SETPOINTS = {
    "LIT101": {"low": 250, "high": 800},
    "LIT301": {"low": 250, "high": 800},
    "LIT401": {"low": 250, "high": 800},
}

# Sensor noise standard deviations (in sensor units)
NOISE = {
    "FIT101": 0.03, "LIT101": 1.5,
    "AIT201": 0.015, "AIT202": 0.8, "AIT203": 0.4, "FIT201": 0.03,
    "DPIT301": 0.3, "FIT301": 0.03, "LIT301": 1.5,
    "AIT401": 2.0, "AIT402": 0.8, "FIT401": 0.03, "LIT401": 1.5,
    "AIT501": 0.015, "AIT502": 0.8, "AIT503": 0.1, "AIT504": 0.8,
    "FIT501": 0.03, "FIT502": 0.02, "FIT503": 0.02, "FIT504": 0.02,
    "PIT501": 1.0, "PIT502": 0.3, "PIT503": 1.0,
    "FIT601": 0.03,
}


class SWaTSimulator:
    """
    Simulates the SWaT water treatment process.

    The plant has 6 stages connected in series. Water flows:
        Raw water → Tank 1 → Chemical dosing → Tank 3 (UF) →
        Tank 4 (RO feed) → RO membrane → Permeate storage

    Each stage has control logic that maintains tank levels within
    setpoints by switching pumps on/off.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.dt = 1.0  # 1 second timestep
        self._reset_state()

    def _reset_state(self):
        """Initialize all process variables to normal operating values."""
        self.state = {
            # Stage 1
            "FIT101": 2.5, "LIT101": 500.0,
            "MV101": 2, "P101": 2, "P102": 0,
            # Stage 2
            "AIT201": 7.05, "AIT202": 260.0, "AIT203": 410.0, "FIT201": 2.45,
            "MV201": 2, "P201": 2, "P202": 0, "P203": 0, "P204": 0, "P205": 0, "P206": 0,
            # Stage 3
            "DPIT301": 40.0, "FIT301": 2.0, "LIT301": 500.0,
            "MV301": 2, "MV302": 2, "MV303": 0, "MV304": 0, "P301": 2, "P302": 0,
            # Stage 4
            "AIT401": 120.0, "AIT402": 250.0, "FIT401": 1.95, "LIT401": 500.0,
            "P401": 2, "P402": 0, "P403": 2, "P404": 0, "UV401": 2,
            # Stage 5
            "AIT501": 6.95, "AIT502": 255.0, "AIT503": 25.0, "AIT504": 260.0,
            "FIT501": 1.80, "FIT502": 1.20, "FIT503": 0.40, "FIT504": 0.20,
            "PIT501": 180.0, "PIT502": 8.0, "PIT503": 175.0,
            "P501": 2, "P502": 0,
            # Stage 6
            "FIT601": 0.0, "P601": 0, "P602": 0,
        }
        self.backwash_timer = 0
        self.backwash_interval = 3600  # backwash every hour
        self.backwash_duration = 60    # lasts 60 seconds

    def step(self) -> Dict[str, float]:
        """Advance simulation by 1 second. Returns sensor readings with noise."""
        self._run_control_logic()
        self._update_physics()
        return self._measure()

    def _run_control_logic(self):
        """
        PLC control logic — keeps tanks within setpoints.

        This is the real SWaT's control strategy (simplified):
        each tank has low/high setpoints. When level hits high,
        the outflow pump turns on. When it hits low, the inflow
        valve opens (or outflow pump turns off).
        """
        s = self.state

        # Stage 1: MV101 opens when LIT101 < low, P101 runs when LIT101 > low
        if s["LIT101"] < SETPOINTS["LIT101"]["low"]:
            s["MV101"] = 2   # open valve
            s["P101"] = 0    # stop outflow
        elif s["LIT101"] > SETPOINTS["LIT101"]["high"]:
            s["MV101"] = 0   # close valve
            s["P101"] = 2    # start pumping out

        # Stage 3: P301 runs when LIT301 has water
        if s["LIT301"] < SETPOINTS["LIT301"]["low"]:
            s["P301"] = 0
        elif s["LIT301"] > SETPOINTS["LIT301"]["high"]:
            s["P301"] = 2

        # Stage 4: P501 runs when LIT401 has water
        if s["LIT401"] < SETPOINTS["LIT401"]["low"]:
            s["P501"] = 0
        elif s["LIT401"] > SETPOINTS["LIT401"]["high"]:
            s["P501"] = 2

        # Backwash timer
        self.backwash_timer += 1
        if self.backwash_timer >= self.backwash_interval:
            s["P601"] = 2
            s["MV304"] = 2
            if self.backwash_timer >= self.backwash_interval + self.backwash_duration:
                s["P601"] = 0
                s["MV304"] = 0
                self.backwash_timer = 0

    def _update_physics(self):
        """
        Advance physical state by dt=1 second using conservation laws.
        This is the ground truth — what actually happens in the plant.
        """
        s = self.state
        dt = self.dt

        # --- Stage 1: Tank 1 mass balance ---
        # dLIT101/dt = (Q_in - Q_out) / A × 1000  (mm/s, since level in mm)
        q_in = PUMP_FLOW.get("P101", 0) / 3600 * (1 if s["MV101"] == 2 else 0)
        q_out = PUMP_FLOW.get("P101", 0) / 3600 * (1 if s["P101"] == 2 else 0)
        s["FIT101"] = q_in * 3600 if s["MV101"] == 2 else 0.0
        dh1 = (q_in - q_out) / TANK_AREA[1] * 1000  # m→mm
        s["LIT101"] = np.clip(s["LIT101"] + dh1 * dt, 0, TANK_MAX[1])

        # --- Stage 2: Chemical dosing dynamics ---
        flow_through = q_out * 3600  # what P101 pushes is what flows through stage 2
        s["FIT201"] = flow_through
        V_mix = 0.5  # mixing volume m³

        # pH dynamics
        dosing_NaOCl = 0.001 * (1 if s["P201"] == 2 else 0)
        dosing_HCl = 0.002 * (1 if s["P203"] == 2 else 0)
        dpH = (dosing_NaOCl * 5.0 - dosing_HCl * 8.0) / V_mix - 0.001 * (s["AIT201"] - 7.0)
        s["AIT201"] = np.clip(s["AIT201"] + dpH * dt, 5.0, 10.0)

        # ORP dynamics
        dORP = (dosing_NaOCl * 500 - 0.005 * (s["AIT202"] - 200)) / V_mix
        s["AIT202"] = np.clip(s["AIT202"] + dORP * dt, 50, 600)

        # Conductivity (conservative tracer — changes only from dilution)
        sigma_in = 420.0  # raw water conductivity
        if flow_through > 0.01:
            dsigma = flow_through / 3600 * (sigma_in - s["AIT203"]) / V_mix
            s["AIT203"] = np.clip(s["AIT203"] + dsigma * dt, 100, 800)

        # --- Stage 3: UF feed tank ---
        q_in3 = q_out  # P101 output → stage 2 → stage 3
        q_out3 = PUMP_FLOW.get("P301", 0) / 3600 * (1 if s["P301"] == 2 else 0)
        s["FIT301"] = q_out3 * 3600
        dh3 = (q_in3 - q_out3) / TANK_AREA[3] * 1000
        s["LIT301"] = np.clip(s["LIT301"] + dh3 * dt, 0, TANK_MAX[3])

        # UF membrane differential pressure (fouling model)
        if s["P301"] == 2:
            s["DPIT301"] += 0.001 * dt  # slow fouling
        if s["P601"] == 2:  # backwash cleans membrane
            s["DPIT301"] = max(s["DPIT301"] - 0.5 * dt, 20.0)
        s["DPIT301"] = np.clip(s["DPIT301"], 0, 100)

        # --- Stage 4: RO feed tank ---
        q_in4 = q_out3
        q_out4 = PUMP_FLOW.get("P501", 0) / 3600 * (1 if s["P501"] == 2 else 0)
        s["FIT401"] = q_in4 * 3600
        dh4 = (q_in4 - q_out4) / TANK_AREA[4] * 1000
        s["LIT401"] = np.clip(s["LIT401"] + dh4 * dt, 0, TANK_MAX[4])

        # Hardness and ORP are approximately stable
        s["AIT401"] += self.rng.normal(0, 0.1)
        s["AIT401"] = np.clip(s["AIT401"], 50, 300)
        s["AIT402"] = s["AIT202"] * 0.3 + self.rng.normal(0, 0.5)  # UV reduces ORP

        # --- Stage 5: Reverse osmosis ---
        q_ro_in = q_out4
        s["FIT501"] = q_ro_in * 3600

        if s["P501"] == 2 and q_ro_in > 0:
            recovery = 0.65  # 65% recovery rate (typical RO)
            s["FIT502"] = s["FIT501"] * recovery
            s["FIT503"] = s["FIT501"] * (1 - recovery) * 0.75
            s["FIT504"] = s["FIT501"] - s["FIT502"] - s["FIT503"]

            s["PIT501"] = 180.0 + self.rng.normal(0, 0.5)
            s["PIT502"] = 8.0 + self.rng.normal(0, 0.2)
            s["PIT503"] = s["PIT501"] - 5.0 + self.rng.normal(0, 0.3)
        else:
            s["FIT502"] = 0.0
            s["FIT503"] = 0.0
            s["FIT504"] = 0.0
            s["PIT501"] = 0.0
            s["PIT502"] = 0.0
            s["PIT503"] = 0.0

        # Chemistry at RO
        s["AIT501"] = s["AIT201"] - 0.1 + self.rng.normal(0, 0.01)  # pH tracks with delay
        s["AIT502"] = s["AIT202"] * 0.95 + self.rng.normal(0, 0.5)
        s["AIT503"] = s["AIT203"] * 0.05 + self.rng.normal(0, 0.05)  # 95% rejection
        s["AIT504"] = s["AIT502"] * 0.9 + self.rng.normal(0, 0.5)

        # --- Stage 6: Backwash ---
        s["FIT601"] = PUMP_FLOW.get("P601", 0) * (1 if s["P601"] == 2 else 0)

    def _measure(self) -> Dict[str, float]:
        """Add measurement noise to get realistic sensor readings."""
        measured = {}
        for col in ALL_COLUMNS:
            true_val = self.state.get(col, 0.0)
            info = ALL_STAGES.get(col, {})
            if info.get("type") == "actuator":
                measured[col] = true_val  # actuators have no measurement noise
            else:
                noise_std = NOISE.get(col, 0.01)
                measured[col] = true_val + self.rng.normal(0, noise_std)
        return measured


# -----------------------------------------------------------------------
# Attack scenarios (mimicking real SWaT attacks)
# -----------------------------------------------------------------------

ATTACK_SCENARIOS = [
    {
        "name": "A1",
        "target": "LIT101",
        "type": "constant",
        "value": 700.0,
        "desc": "Spoof Tank 1 level high → MV101 closes, tank overflows from real inflow",
    },
    {
        "name": "A2",
        "target": "FIT101",
        "type": "constant",
        "value": 0.0,
        "desc": "Spoof flow to zero → controller thinks no water entering",
    },
    {
        "name": "A3",
        "target": "LIT301",
        "type": "constant",
        "value": 900.0,
        "desc": "Spoof UF tank level high → P301 stays on, tank drains",
    },
    {
        "name": "A4",
        "target": "AIT201",
        "type": "constant",
        "value": 8.5,
        "desc": "Spoof pH high → controller adds acid (P203), actual pH drops dangerously",
    },
    {
        "name": "A5",
        "target": "LIT101",
        "type": "slow_drift",
        "rate": 0.5,
        "desc": "Slowly increase reported level → controller slowly shuts down inflow",
    },
    {
        "name": "A6",
        "target": "AIT202",
        "type": "constant",
        "value": 150.0,
        "desc": "Spoof ORP low → controller increases chlorine dosing, over-chlorination",
    },
    {
        "name": "A7",
        "target": "FIT501",
        "type": "constant",
        "value": 0.0,
        "desc": "Spoof RO inlet flow to zero → controller thinks RO is offline",
    },
    {
        "name": "A8",
        "target": "DPIT301",
        "type": "constant",
        "value": 20.0,
        "desc": "Spoof low diff pressure → controller thinks membrane is clean, skips backwash",
    },
    {
        "name": "A9",
        "target": "LIT401",
        "type": "slow_drift",
        "rate": -0.3,
        "desc": "Slowly decrease RO feed tank level → controller keeps pumping in, overflow",
    },
    {
        "name": "A10",
        "target": "PIT501",
        "type": "constant",
        "value": 50.0,
        "desc": "Spoof low RO pressure → pump runs harder, real pressure dangerously high",
    },
]


def generate_normal(n_hours: float = 24, seed: int = 42) -> pd.DataFrame:
    """Generate normal operation data."""
    sim = SWaTSimulator(seed=seed)
    n_steps = int(n_hours * 3600)
    rows = []
    start_time = datetime(2024, 12, 28, 10, 0, 0)

    for i in range(n_steps):
        reading = sim.step()
        reading["Timestamp"] = (start_time + timedelta(seconds=i)).strftime("%d/%m/%Y %I:%M:%S %p")
        reading["Normal/Attack"] = "Normal"
        rows.append(reading)

    df = pd.DataFrame(rows)
    cols = ["Timestamp"] + ALL_COLUMNS + ["Normal/Attack"]
    return df[cols]


def generate_attack(
    attack_scenario: Dict,
    normal_hours: float = 2,
    attack_hours: float = 1,
    post_hours: float = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate data with a specific attack injected."""
    sim = SWaTSimulator(seed=seed)
    rows = []
    start_time = datetime(2024, 12, 28, 10, 0, 0)

    n_normal = int(normal_hours * 3600)
    n_attack = int(attack_hours * 3600)
    n_post = int(post_hours * 3600)
    total = n_normal + n_attack + n_post

    attack_start_val = None

    for i in range(total):
        reading = sim.step()
        reading["Timestamp"] = (start_time + timedelta(seconds=i)).strftime("%d/%m/%Y %I:%M:%S %p")

        if i < n_normal:
            reading["Normal/Attack"] = "Normal"
        elif i < n_normal + n_attack:
            # Inject attack
            target = attack_scenario["target"]
            if attack_scenario["type"] == "constant":
                reading[target] = attack_scenario["value"]
            elif attack_scenario["type"] == "slow_drift":
                if attack_start_val is None:
                    attack_start_val = reading[target]
                elapsed = i - n_normal
                reading[target] = attack_start_val + attack_scenario["rate"] * elapsed
            reading["Normal/Attack"] = f"Attack ({attack_scenario['name']})"
        else:
            reading["Normal/Attack"] = "Normal"

        rows.append(reading)

    df = pd.DataFrame(rows)
    cols = ["Timestamp"] + ALL_COLUMNS + ["Normal/Attack"]
    return df[cols]


def generate_full_dataset(output_dir: str = "data"):
    """Generate the complete synthetic SWaT dataset."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("Generating normal operation data (24 hours)...")
    df_normal = generate_normal(n_hours=24, seed=42)
    path_normal = os.path.join(output_dir, "SWaT_Normal_Synthetic.csv")
    df_normal.to_csv(path_normal, index=False)
    print(f"  Saved: {path_normal} ({len(df_normal)} rows, {df_normal.shape[1]} columns)")

    print("\nGenerating attack data (10 scenarios)...")
    attack_frames = []
    for scenario in ATTACK_SCENARIOS:
        print(f"  {scenario['name']}: {scenario['desc']}")
        df_atk = generate_attack(scenario, normal_hours=2, attack_hours=1, post_hours=1, seed=42)
        attack_frames.append(df_atk)

    df_attack = pd.concat(attack_frames, ignore_index=True)
    path_attack = os.path.join(output_dir, "SWaT_Attack_Synthetic.csv")
    df_attack.to_csv(path_attack, index=False)
    print(f"  Saved: {path_attack} ({len(df_attack)} rows)")

    # Summary statistics
    print(f"\n--- Dataset Summary ---")
    print(f"Normal: {len(df_normal)} rows ({len(df_normal)/3600:.1f} hours)")
    print(f"Attack: {len(df_attack)} rows ({len(ATTACK_SCENARIOS)} scenarios)")
    print(f"Columns: {df_normal.shape[1]} (Timestamp + {len(ALL_COLUMNS)} process + Label)")

    # Print sensor ranges from generated data
    print(f"\nSensor ranges (normal data):")
    for col in SENSOR_COLUMNS:
        lo, hi = df_normal[col].min(), df_normal[col].max()
        mean = df_normal[col].mean()
        std = df_normal[col].std()
        print(f"  {col:10s}  [{lo:>8.2f} - {hi:>8.2f}]  "
              f"mean={mean:>8.2f}  std={std:>6.3f}")

    return df_normal, df_attack


if __name__ == "__main__":
    import sys
    generate_full_dataset(output_dir="data")

    if "--plot" in sys.argv:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = pd.read_csv("data/SWaT_Normal_Synthetic.csv")
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

        time_h = np.arange(len(df)) / 3600

        axes[0].plot(time_h, df["LIT101"], linewidth=0.5)
        axes[0].set_ylabel("LIT101 (mm)")
        axes[0].set_title("Tank Levels")
        axes[0].plot(time_h, df["LIT301"], linewidth=0.5)
        axes[0].plot(time_h, df["LIT401"], linewidth=0.5)
        axes[0].legend(["Tank 1", "Tank 3", "Tank 4"])

        axes[1].plot(time_h, df["AIT201"], linewidth=0.5)
        axes[1].set_ylabel("pH")
        axes[1].set_title("Chemical Parameters")

        axes[2].plot(time_h, df["FIT101"], linewidth=0.5)
        axes[2].plot(time_h, df["FIT301"], linewidth=0.5)
        axes[2].plot(time_h, df["FIT501"], linewidth=0.5)
        axes[2].set_ylabel("Flow (m³/h)")
        axes[2].set_title("Flow Rates")
        axes[2].legend(["FIT101", "FIT301", "FIT501"])

        axes[3].plot(time_h, df["DPIT301"], linewidth=0.5)
        axes[3].set_ylabel("kPa")
        axes[3].set_xlabel("Time (hours)")
        axes[3].set_title("UF Membrane Differential Pressure")

        plt.tight_layout()
        plt.savefig("data/synthetic_normal_overview.png", dpi=150)
        print("\nPlot saved to data/synthetic_normal_overview.png")
