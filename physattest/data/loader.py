"""
SWaT dataset loader.

Handles two modes:
1. Real SWaT data (CSV from iTrust) — when you have the files
2. Synthetic SWaT data — realistic simulation for development

The real SWaT dataset comes as:
  - SWaT_Dataset_Normal_v1.csv  (7 days normal operation, ~500K rows)
  - SWaT_Dataset_Attack_v0.csv  (4 days with 36 attacks, ~450K rows)

Columns: Timestamp, sensor/actuator readings..., Normal/Attack label
Sampling rate: 1 second
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from .swat_config import (
    ALL_SENSORS, CONTINUOUS_SENSORS, ACTUATORS,
    LEVEL_SENSORS, FLOW_SENSORS, PRESSURE_SENSORS, CHEMICAL_SENSORS,
    SWaTPhysics,
)


class SWaTLoader:
    """
    Load and preprocess SWaT dataset for PhysAttest experiments.

    Usage:
        # With real data:
        loader = SWaTLoader(data_dir="path/to/swat/csvs")
        normal, attacks = loader.load()

        # Without real data (synthetic):
        loader = SWaTLoader()
        normal, attacks = loader.load_synthetic()
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else None
        self.physics = SWaTPhysics()

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load real SWaT CSV files.

        Returns:
            (normal_df, attack_df) — both with standardised column names
        """
        if self.data_dir is None:
            raise FileNotFoundError(
                "No data directory specified. Use load_synthetic() for development, "
                "or provide data_dir pointing to the SWaT CSV files from iTrust."
            )

        normal_path = self._find_file("Normal")
        attack_path = self._find_file("Attack")

        normal_df = self._load_csv(normal_path)
        attack_df = self._load_csv(attack_path)

        return normal_df, attack_df

    def _find_file(self, pattern: str) -> Path:
        """Find the CSV file matching a pattern."""
        candidates = list(self.data_dir.glob(f"*{pattern}*.csv"))
        if not candidates:
            candidates = list(self.data_dir.glob(f"*{pattern.lower()}*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"No CSV file matching '*{pattern}*' in {self.data_dir}"
            )
        return candidates[0]

    def _load_csv(self, path: Path) -> pd.DataFrame:
        """Load and standardise a SWaT CSV file."""
        df = pd.read_csv(path)

        # Standardise column names (strip whitespace, handle variations)
        df.columns = [c.strip().replace(" ", "") for c in df.columns]

        # Parse timestamp
        for col in ["Timestamp", "timestamp", "DateTime"]:
            if col in df.columns:
                df["timestamp"] = pd.to_datetime(df[col])
                break

        # Parse attack label
        for col in ["Normal/Attack", "Attack", "Label", "label"]:
            if col in df.columns:
                df["is_attack"] = df[col].str.strip().str.lower().isin(
                    ["attack", "a", "1", "true"]
                )
                break

        if "is_attack" not in df.columns:
            df["is_attack"] = False

        # Convert numeric columns
        for col in df.columns:
            if col not in ["timestamp", "is_attack"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def load_synthetic(
        self,
        n_normal: int = 10000,
        n_attack: int = 5000,
        seed: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate realistic synthetic SWaT data for development.

        Models the 6 sub-processes with:
        - Realistic steady-state values and noise levels
        - Mass/flow conservation between processes
        - 15 attack scenarios matching the real SWaT catalogue
        """
        rng = np.random.default_rng(seed)

        # --- Normal data ---
        normal = self._generate_normal(n_normal, rng)

        # --- Attack data ---
        attack = self._generate_attacks(n_attack, rng)

        return normal, attack

    def _generate_normal(self, n: int, rng) -> pd.DataFrame:
        """Generate normal operation data with realistic dynamics."""
        t = np.arange(n, dtype=float)
        data = {}

        # P1: Raw water — tank fills and drains cyclically
        lit101_base = 500 + 100 * np.sin(2 * np.pi * t / 3600)  # 1hr cycle
        data["LIT101"] = lit101_base + rng.normal(0, 2, n)
        data["FIT101"] = 1.5 + 0.3 * np.sin(2 * np.pi * t / 3600 + 0.5) + rng.normal(0, 0.05, n)
        data["MV101"] = (np.sin(2 * np.pi * t / 3600) > 0).astype(float)
        data["P101"] = (data["LIT101"] > 400).astype(float)
        data["P102"] = (data["LIT101"] > 600).astype(float)

        # P2: Pre-treatment — chemical dosing
        data["FIT201"] = data["FIT101"] * 0.95 + rng.normal(0, 0.03, n)
        data["AIT201"] = 200 + rng.normal(0, 5, n)     # conductivity
        data["AIT202"] = 7.2 + rng.normal(0, 0.1, n)   # pH
        data["AIT203"] = 250 + rng.normal(0, 10, n)     # ORP
        for s in ["MV201", "P201", "P202", "P203", "P204", "P205", "P206"]:
            data[s] = rng.choice([0.0, 1.0, 2.0], n)

        # P3: UF — pressure-flow dynamics
        data["LIT301"] = 600 + 80 * np.sin(2 * np.pi * t / 2400) + rng.normal(0, 3, n)
        data["FIT301"] = data["FIT201"] * 0.9 + rng.normal(0, 0.04, n)
        data["DPIT301"] = 20 + 5 * (data["FIT301"] / 2.0) + rng.normal(0, 0.5, n)
        for s in ["MV301", "MV302", "MV303", "MV304", "P301", "P302"]:
            data[s] = rng.choice([0.0, 1.0, 2.0], n)

        # P4: De-chlorination
        data["LIT401"] = 500 + 60 * np.sin(2 * np.pi * t / 1800) + rng.normal(0, 2, n)
        data["FIT401"] = data["FIT301"] * 0.85 + rng.normal(0, 0.03, n)
        data["AIT401"] = 180 + rng.normal(0, 5, n)
        data["AIT402"] = 7.0 + rng.normal(0, 0.08, n)
        for s in ["P401", "P402", "P403", "P404", "UV401"]:
            data[s] = rng.choice([0.0, 1.0, 2.0], n)

        # P5: RO — membrane dynamics
        data["FIT501"] = data["FIT401"] * 0.8 + rng.normal(0, 0.02, n)
        data["FIT502"] = data["FIT501"] * 0.75 + rng.normal(0, 0.02, n)  # permeate
        data["FIT503"] = data["FIT501"] * 0.25 + rng.normal(0, 0.01, n)  # concentrate
        data["FIT504"] = data["FIT502"] + rng.normal(0, 0.01, n)
        data["AIT501"] = 250 + rng.normal(0, 8, n)
        data["AIT502"] = 7.1 + rng.normal(0, 0.1, n)
        data["AIT503"] = 280 + rng.normal(0, 10, n)
        data["AIT504"] = 150 + rng.normal(0, 5, n)  # permeate: lower conductivity
        for s in ["MV501", "MV502", "MV503", "MV504", "P501", "P502"]:
            data[s] = rng.choice([0.0, 1.0, 2.0], n)

        # P6: Backwash
        data["FIT601"] = data["FIT503"] * 0.3 + rng.normal(0, 0.01, n)
        for s in ["P601", "P602", "P603"]:
            data[s] = rng.choice([0.0, 1.0], n)

        df = pd.DataFrame(data)
        df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="1s")
        df["is_attack"] = False
        df["attack_id"] = 0

        return df

    def _generate_attacks(self, n: int, rng) -> pd.DataFrame:
        """Generate attack data with labelled attack windows."""
        # Start with normal data as baseline
        normal = self._generate_normal(n, rng)

        attacks = normal.copy()
        attacks["is_attack"] = False
        attacks["attack_id"] = 0

        attack_window = n // 15  # ~15 attacks spread across the data

        attack_defs = [
            # (attack_id, target_sensor, injection_fn)
            (1,  "LIT101", lambda v, t: v + 200),                    # spike high
            (3,  "LIT101", lambda v, t: v - 300),                    # spike low
            (4,  "LIT301", lambda v, t: np.full_like(v, 200.0)),     # constant low
            (5,  "AIT202", lambda v, t: v + 3.0),                    # pH spike
            (6,  "FIT101", lambda v, t: np.full_like(v, 0.0)),       # zero flow
            (7,  "LIT101", lambda v, t: v - t * 0.05),               # slow drift down
            (10, "FIT401", lambda v, t: v * 1.5),                    # scale up
            (16, "LIT301", lambda v, t: v + t * 0.03),               # slow drift up
            (19, "AIT504", lambda v, t: v + 100),                    # conductivity spike
            (25, "LIT401", lambda v, t: np.full_like(v, 900.0)),     # overflow attempt
        ]

        # Multi-sensor coordinated attacks
        multi_attacks = [
            (30, ["LIT101", "FIT101"], [lambda v, t: v + 200, lambda v, t: v * 0.5]),
            (31, ["LIT401", "FIT401"], [lambda v, t: v - 200, lambda v, t: v * 1.5]),
            (36, ["LIT301", "FIT301", "DPIT301"], [
                lambda v, t: v + 150,
                lambda v, t: np.full_like(v, 0.5),
                lambda v, t: v + 20,
            ]),
        ]

        # Apply single-sensor attacks
        for i, (atk_id, sensor, inject_fn) in enumerate(attack_defs):
            start = i * attack_window + attack_window // 4
            end = min(start + attack_window // 2, n)
            if end <= start or start >= n:
                continue

            window = slice(start, end)
            t_local = np.arange(end - start, dtype=float)
            original = attacks.loc[attacks.index[window], sensor].values

            attacks.loc[attacks.index[window], sensor] = inject_fn(original, t_local)
            attacks.loc[attacks.index[window], "is_attack"] = True
            attacks.loc[attacks.index[window], "attack_id"] = atk_id

        # Apply multi-sensor attacks
        for i, (atk_id, sensors, fns) in enumerate(multi_attacks):
            start = (len(attack_defs) + i) * attack_window + attack_window // 4
            end = min(start + attack_window // 2, n)
            if end <= start or start >= n:
                continue

            window = slice(start, end)
            t_local = np.arange(end - start, dtype=float)

            for sensor, fn in zip(sensors, fns):
                original = attacks.loc[attacks.index[window], sensor].values
                attacks.loc[attacks.index[window], sensor] = fn(original, t_local)

            attacks.loc[attacks.index[window], "is_attack"] = True
            attacks.loc[attacks.index[window], "attack_id"] = atk_id

        return attacks

    def get_sensor_readings(self, df: pd.DataFrame) -> np.ndarray:
        """Extract continuous sensor readings as numpy array."""
        cols = [c for c in CONTINUOUS_SENSORS if c in df.columns]
        return df[cols].values

    def get_actuator_states(self, df: pd.DataFrame) -> np.ndarray:
        """Extract actuator states as numpy array."""
        cols = [c for c in ACTUATORS if c in df.columns]
        return df[cols].values

    def get_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Extract binary attack labels."""
        return df["is_attack"].values.astype(bool)

    def get_attack_ids(self, df: pd.DataFrame) -> np.ndarray:
        """Extract per-sample attack IDs (0 = normal)."""
        return df["attack_id"].values.astype(int)
