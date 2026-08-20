"""
Data loader for SWaT dataset (real or synthetic).

Handles:
  - Real SWaT CSV/XLSX files from iTrust
  - Synthetic CSV files from our generator
  - Column name normalization (real dataset has spaces in column names)
  - Train/test splitting by normal/attack periods
  - Windowing for the observer (sliding windows of sensor readings)

Usage:
    from data.loader import SWaTLoader

    loader = SWaTLoader("data/SWaT_Normal_Synthetic.csv")
    print(loader.info())

    sensors, actuators, labels = loader.get_arrays()
    for window in loader.sliding_windows(window_size=60):
        # window is a (60, n_sensors) array — 1 minute of data
        pass
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Iterator, Optional, List, Dict
from .swat_columns import SENSOR_COLUMNS, ACTUATOR_COLUMNS, ALL_COLUMNS, ALL_STAGES


class SWaTLoader:
    """
    Loads and preprocesses SWaT data for the observer pipeline.
    """

    def __init__(self, filepath: str, max_rows: Optional[int] = None):
        """
        Parameters:
            filepath: path to CSV or XLSX file
            max_rows: limit rows loaded (useful for quick testing)
        """
        self.filepath = Path(filepath)
        self.df = self._load(max_rows)
        self._normalize_columns()
        self._parse_labels()

    def _load(self, max_rows: Optional[int]) -> pd.DataFrame:
        """Load CSV or XLSX file."""
        suffix = self.filepath.suffix.lower()
        if suffix == ".xlsx":
            df = pd.read_excel(self.filepath, nrows=max_rows)
        elif suffix == ".csv":
            df = pd.read_csv(self.filepath, nrows=max_rows)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        return df

    def _normalize_columns(self):
        """
        The real SWaT dataset has quirky column names:
          - Leading/trailing spaces: ' Timestamp ', ' LIT101 '
          - Some versions use different capitalization

        This normalizes everything to clean names matching our registry.
        """
        self.df.columns = [col.strip() for col in self.df.columns]

        rename_map = {}
        for col in self.df.columns:
            clean = col.strip().upper()
            if clean in ALL_STAGES:
                rename_map[col] = clean
            elif clean == "TIMESTAMP":
                rename_map[col] = "Timestamp"
            elif clean in ("NORMAL/ATTACK", "LABEL", "ATTACK", "NORMAL/ ATTACK"):
                rename_map[col] = "Normal/Attack"

        self.df.rename(columns=rename_map, inplace=True)

        # Ensure numeric columns are actually numeric
        for col in ALL_COLUMNS:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

    def _parse_labels(self):
        """Parse the Normal/Attack label column into a boolean attack flag."""
        if "Normal/Attack" in self.df.columns:
            self.df["is_attack"] = self.df["Normal/Attack"].str.strip().str.lower() != "normal"
        else:
            self.df["is_attack"] = False

    def info(self) -> str:
        """Print a summary of the loaded dataset."""
        n_rows = len(self.df)
        hours = n_rows / 3600

        available_sensors = [c for c in SENSOR_COLUMNS if c in self.df.columns]
        available_actuators = [c for c in ACTUATOR_COLUMNS if c in self.df.columns]
        missing = [c for c in ALL_COLUMNS if c not in self.df.columns]

        n_attack = self.df["is_attack"].sum() if "is_attack" in self.df.columns else 0

        lines = [
            f"Dataset: {self.filepath.name}",
            f"Rows: {n_rows:,} ({hours:.1f} hours at 1 Hz)",
            f"Sensors: {len(available_sensors)}/{len(SENSOR_COLUMNS)}",
            f"Actuators: {len(available_actuators)}/{len(ACTUATOR_COLUMNS)}",
            f"Attack rows: {n_attack:,} ({n_attack/n_rows*100:.1f}%)" if n_rows > 0 else "",
        ]
        if missing:
            lines.append(f"Missing columns: {', '.join(missing[:10])}" +
                         (f" (+{len(missing)-10} more)" if len(missing) > 10 else ""))
        return "\n".join(lines)

    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get sensor readings, actuator states, and labels as numpy arrays.

        Returns:
            sensors:   (n_timesteps, n_sensors) float array
            actuators: (n_timesteps, n_actuators) float array
            labels:    (n_timesteps,) boolean array (True = attack)
        """
        available_sensors = [c for c in SENSOR_COLUMNS if c in self.df.columns]
        available_actuators = [c for c in ACTUATOR_COLUMNS if c in self.df.columns]

        sensors = self.df[available_sensors].values.astype(np.float64)
        actuators = self.df[available_actuators].values.astype(np.float64)
        labels = self.df["is_attack"].values if "is_attack" in self.df.columns else np.zeros(len(self.df), dtype=bool)

        # Replace NaN with forward fill then zero
        sensors = pd.DataFrame(sensors).ffill().fillna(0).values
        actuators = pd.DataFrame(actuators).ffill().fillna(0).values

        return sensors, actuators, labels

    def get_sensor_names(self) -> List[str]:
        """Return list of available sensor column names."""
        return [c for c in SENSOR_COLUMNS if c in self.df.columns]

    def get_actuator_names(self) -> List[str]:
        """Return list of available actuator column names."""
        return [c for c in ACTUATOR_COLUMNS if c in self.df.columns]

    def get_normal_data(self) -> pd.DataFrame:
        """Return only normal (non-attack) rows."""
        return self.df[~self.df["is_attack"]].copy()

    def get_attack_data(self) -> pd.DataFrame:
        """Return only attack rows."""
        return self.df[self.df["is_attack"]].copy()

    def get_attack_periods(self) -> List[Dict]:
        """
        Find contiguous attack periods with start/end indices.

        Returns list of dicts with:
            start: first attack row index
            end: last attack row index
            duration: length in seconds
            label: attack label string
        """
        if "is_attack" not in self.df.columns:
            return []

        attacks = self.df["is_attack"].values
        periods = []
        in_attack = False
        start = 0

        for i in range(len(attacks)):
            if attacks[i] and not in_attack:
                start = i
                in_attack = True
            elif not attacks[i] and in_attack:
                label_col = self.df["Normal/Attack"].iloc[start] if "Normal/Attack" in self.df.columns else "Attack"
                periods.append({
                    "start": start,
                    "end": i - 1,
                    "duration": i - start,
                    "label": str(label_col).strip(),
                })
                in_attack = False

        if in_attack:
            label_col = self.df["Normal/Attack"].iloc[start] if "Normal/Attack" in self.df.columns else "Attack"
            periods.append({
                "start": start,
                "end": len(attacks) - 1,
                "duration": len(attacks) - start,
                "label": str(label_col).strip(),
            })

        return periods

    def sliding_windows(
        self,
        window_size: int = 60,
        step: int = 1,
        sensors_only: bool = True,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, int]]:
        """
        Yield sliding windows of data for the observer.

        Parameters:
            window_size: number of timesteps per window
            step: stride between windows
            sensors_only: if True, yield only sensor data; else include actuators

        Yields:
            (window_data, window_labels, start_index)
        """
        sensors, actuators, labels = self.get_arrays()
        data = sensors if sensors_only else np.hstack([sensors, actuators])

        for i in range(0, len(data) - window_size + 1, step):
            yield data[i:i + window_size], labels[i:i + window_size], i

    def compute_statistics(self, normal_only: bool = True) -> pd.DataFrame:
        """
        Compute per-sensor statistics (used for threshold calibration).

        Parameters:
            normal_only: if True, compute stats only from normal data
        """
        df = self.get_normal_data() if normal_only else self.df
        available = [c for c in SENSOR_COLUMNS if c in df.columns]

        stats = []
        for col in available:
            vals = df[col].dropna()
            stats.append({
                "sensor": col,
                "mean": vals.mean(),
                "std": vals.std(),
                "min": vals.min(),
                "max": vals.max(),
                "median": vals.median(),
                "q1": vals.quantile(0.25),
                "q3": vals.quantile(0.75),
            })

        return pd.DataFrame(stats).set_index("sensor")
