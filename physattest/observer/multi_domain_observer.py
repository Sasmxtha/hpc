"""
Multi-Domain Observer for SWaT — the heart of PhysAttest (Component 1).

Implements a Luenberger observer with Kalman-optimal gain, computing
three independent residuals per sensor (physics, chemistry, math).

Usage:
    observer = MultiDomainObserver()
    observer.initialize(y_initial)
    for each timestep:
        residuals = observer.step(y_measured, u_actuators)
        if any |residual| > threshold: flag sensor
"""

import numpy as np
from scipy.linalg import solve_discrete_are, expm
from typing import Dict, Tuple, Optional

from .physics_equations import (
    mass_conservation_tank, flow_conservation_junction,
    hydrostatic_pressure, TANK_PARAMS, PUMP_FLOW_RATES
)
from .chemistry_equations import (
    ph_dosing_response, orp_chlorination_response, conductivity_mixing
)
from .math_invariants import (
    volume_level_consistency, flow_balance_across_stages,
    total_water_mass_conservation
)


# -----------------------------------------------------------------------
# State indices (positions in the state vector x)
# -----------------------------------------------------------------------
STATE_H1   = 0   # Tank 1 level (m)
STATE_H3   = 1   # Tank 3 level (m)
STATE_H4   = 2   # Tank 4 level (m)
STATE_PH   = 3   # pH after dosing
STATE_ORP  = 4   # ORP after chlorination
STATE_COND = 5   # Conductivity
STATE_DP3  = 6   # Differential pressure across UF membrane

N_STATES = 7

# -----------------------------------------------------------------------
# Input indices (positions in the input vector u)
# -----------------------------------------------------------------------
INPUT_MV101 = 0   # Intake valve (0/1)
INPUT_P101  = 1   # Raw water pump (0/1)
INPUT_P201  = 2   # Dosing pump (0-1 continuous)
INPUT_P301  = 3   # UF pump (0/1)
INPUT_P302  = 4   # UF pump 2 (0/1)
INPUT_P501  = 5   # RO pump (0/1)
INPUT_P601  = 6   # Backwash pump (0/1)
INPUT_UV401 = 7   # UV lamp (0/1)

N_INPUTS = 8

# -----------------------------------------------------------------------
# Sensor-to-state mapping
# Maps SWaT sensor names to state indices and measurement scaling.
# -----------------------------------------------------------------------
SENSOR_MAP = {
    "LIT101": (STATE_H1, 1.0),
    "LIT301": (STATE_H3, 1.0),
    "LIT401": (STATE_H4, 1.0),
    "AIT201": (STATE_PH, 1.0),
    "AIT202": (STATE_ORP, 1.0),
    "AIT203": (STATE_COND, 1.0),
    "DPIT301": (STATE_DP3, 1.0),
}


class MultiDomainObserver:
    """
    State-space observer with three verification domains.

    The observer runs at 1 Hz (matching SWaT's 1-second sampling).
    At each step it:
      1. Predicts the next state from the current estimate + actuator inputs
      2. Computes residuals = measured - predicted for each domain
      3. Corrects the estimate using the Kalman gain (only for trusted sensors)
    """

    def __init__(self, dt: float = 1.0):
        self.dt = dt
        self.n_states = N_STATES
        self.n_inputs = N_INPUTS

        # State estimate
        self.x_hat = np.zeros(N_STATES)

        # Build continuous-time system matrices
        self.A_c, self.B_c = self._build_continuous_system()

        # Discretize: A_d = e^(A_c * dt), B_d = A_c⁻¹(A_d - I)B_c
        self.A_d, self.B_d = self._discretize(self.A_c, self.B_c, dt)

        # Output matrix: direct measurement of all states
        self.C = np.eye(N_STATES)

        # Noise covariances (tune these from clean data)
        self.Q = np.diag([1e-4, 1e-4, 1e-4, 1e-3, 1.0, 0.1, 0.01])  # process
        self.R = np.diag([1e-3, 1e-3, 1e-3, 0.01, 1.0, 0.5, 0.05])  # measurement

        # Normalization scales for each state (so residuals are comparable)
        self.residual_scales = np.array([
            1.0,    # h1: meters
            1.0,    # h3: meters
            1.0,    # h4: meters
            1.0,    # pH: pH units
            100.0,  # ORP: millivolts (divide by 100 to normalize)
            100.0,  # conductivity: μS/cm
            1.0,    # ΔP: bar
        ])

        # Compute Kalman gain
        self.L = self._compute_kalman_gain()

        # Track which sensors are trusted (1 = trusted, 0 = flagged)
        self.trust_mask = np.ones(N_STATES)

        # Running integrals for math domain checks
        self.total_inflow_integral = 0.0
        self.total_outflow_integral = 0.0
        self.initial_total_volume = None

        # History for SINDy (stores recent residuals)
        self.residual_history = []

    def _build_continuous_system(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the A and B matrices encoding plant dynamics.

        A encodes how states evolve autonomously (drainage, decay, mixing).
        B encodes how actuator inputs drive state changes.

        These are linearized versions of the conservation equations.
        SINDy and PINN will later refine these with learned coefficients.
        """
        A = np.zeros((N_STATES, N_STATES))
        B = np.zeros((N_STATES, N_INPUTS))

        # --- Tank 1 level dynamics ---
        # dh1/dt = (Q_MV101 - Q_P101) / A1
        # Q depends on actuator state, so it goes in B
        area_1 = TANK_PARAMS["T1"]["area"]
        q_p101 = PUMP_FLOW_RATES["P101"] / 3600  # convert m³/h to m³/s
        B[STATE_H1, INPUT_MV101] = q_p101 / area_1    # inflow when valve open
        B[STATE_H1, INPUT_P101]  = -q_p101 / area_1   # outflow when pump on

        # --- Tank 3 level dynamics ---
        area_3 = TANK_PARAMS["T3"]["area"]
        q_p301 = PUMP_FLOW_RATES["P301"] / 3600
        B[STATE_H3, INPUT_P101]  = q_p101 / area_3    # inflow from stage 1
        B[STATE_H3, INPUT_P301]  = -q_p301 / area_3   # outflow through UF

        # --- Tank 4 level dynamics ---
        area_4 = TANK_PARAMS["T4"]["area"]
        q_p501 = PUMP_FLOW_RATES["P501"] / 3600
        B[STATE_H4, INPUT_P301]  = q_p301 / area_4    # inflow from stage 3
        B[STATE_H4, INPUT_P501]  = -q_p501 / area_4   # outflow to RO

        # --- pH dynamics (chemistry) ---
        # dpH/dt = dosing_effect - decay*(pH - 7)
        A[STATE_PH, STATE_PH] = -0.001     # natural equilibration toward neutral
        B[STATE_PH, INPUT_P201] = 0.05      # dosing pump effect on pH

        # --- ORP dynamics (chemistry) ---
        A[STATE_ORP, STATE_ORP] = -0.005    # chlorine decay
        B[STATE_ORP, INPUT_P201] = 10.0     # chlorine dosing raises ORP

        # --- Conductivity dynamics ---
        # dσ/dt = -dilution_rate * σ (mixing)
        A[STATE_COND, STATE_COND] = -0.0001  # slow drift from mixing

        # --- Differential pressure dynamics ---
        # dΔP/dt = fouling_rate * flow - decay
        A[STATE_DP3, STATE_DP3] = -0.01      # pressure decay
        B[STATE_DP3, INPUT_P301] = 0.5       # flow increases ΔP

        return A, B

    def _discretize(self, A_c: np.ndarray, B_c: np.ndarray,
                     dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Exact discretization using matrix exponential.
        A_d = e^(A_c * dt)
        B_d = A_c⁻¹ (A_d - I) B_c   (or numerical integration if A singular)
        """
        A_d = expm(A_c * dt)

        # For numerical stability, use the integral form
        n = A_c.shape[0]
        # Build augmented matrix [A B; 0 0] and exponentiate
        aug = np.zeros((n + B_c.shape[1], n + B_c.shape[1]))
        aug[:n, :n] = A_c * dt
        aug[:n, n:] = B_c * dt
        aug_exp = expm(aug)
        B_d = aug_exp[:n, n:]

        return A_d, B_d

    def _compute_kalman_gain(self) -> np.ndarray:
        """
        Solve the Discrete Algebraic Riccati Equation (DARE) to get the
        steady-state Kalman gain.

        P = A P Aᵀ - A P Cᵀ (C P Cᵀ + R)⁻¹ C P Aᵀ + Q
        L = P Cᵀ (C P Cᵀ + R)⁻¹
        """
        try:
            P = solve_discrete_are(self.A_d.T, self.C.T, self.Q, self.R)
            S = self.C @ P @ self.C.T + self.R
            L = P @ self.C.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            L = 0.1 * np.eye(N_STATES)
        return L

    def initialize(self, y_initial: np.ndarray):
        """Set the initial state estimate from the first measurement."""
        self.x_hat = y_initial.copy()
        areas = [TANK_PARAMS[t]["area"] for t in ["T1", "T3", "T4"]]
        levels = [self.x_hat[STATE_H1], self.x_hat[STATE_H3], self.x_hat[STATE_H4]]
        self.initial_total_volume = sum(a * h for a, h in zip(areas, levels))
        self.total_inflow_integral = 0.0
        self.total_outflow_integral = 0.0

    def step(self, y: np.ndarray, u: np.ndarray) -> Dict[str, np.ndarray]:
        """
        One observer cycle (called every second).

        Parameters:
            y: measured sensor values (N_STATES,)
            u: actuator states (N_INPUTS,)

        Returns:
            Dictionary with:
                'physics':   physics residuals (N_STATES,)
                'chemistry': chemistry residuals (N_STATES,)
                'math':      math residuals (N_STATES,)
                'combined':  combined residual vector
                'x_hat':     current state estimate
        """
        # --- Step 1: Predict ---
        x_pred = self.A_d @ self.x_hat + self.B_d @ u

        # --- Step 2: Compute domain-specific residuals ---
        r_physics = self._compute_physics_residuals(y, x_pred, u)
        r_chemistry = self._compute_chemistry_residuals(y, x_pred, u)
        r_math = self._compute_math_residuals(y, x_pred, u)

        # Normalize domain residuals so each state contributes equally
        r_physics /= self.residual_scales
        r_chemistry /= self.residual_scales
        r_math /= self.residual_scales

        # Combined residual (used for Kalman update) — UNnormalized for the filter
        r_combined = y - self.C @ x_pred
        r_combined_normalized = r_combined / self.residual_scales

        # --- Step 3: Update (only trust uncompromised sensors) ---
        L_masked = self.L * self.trust_mask[np.newaxis, :]
        self.x_hat = x_pred + L_masked @ r_combined

        # --- Step 4: Update running integrals for math domain ---
        q_in = PUMP_FLOW_RATES.get("P101", 0) / 3600 * u[INPUT_MV101]
        q_out = PUMP_FLOW_RATES.get("P601", 0) / 3600 * u[INPUT_P601]
        self.total_inflow_integral += q_in * self.dt
        self.total_outflow_integral += q_out * self.dt

        # Store for SINDy
        result = {
            "physics": r_physics,
            "chemistry": r_chemistry,
            "math": r_math,
            "combined": r_combined_normalized,
            "x_hat": self.x_hat.copy(),
        }
        self.residual_history.append(result)

        return result

    def _compute_physics_residuals(self, y: np.ndarray, x_pred: np.ndarray,
                                    u: np.ndarray) -> np.ndarray:
        """
        Physics domain: conservation of mass and energy.
        """
        r = np.zeros(N_STATES)

        # Tank 1: mass conservation
        area_1 = TANK_PARAMS["T1"]["area"]
        q_in = PUMP_FLOW_RATES["P101"] / 3600 * u[INPUT_MV101]
        q_out = PUMP_FLOW_RATES["P101"] / 3600 * u[INPUT_P101]
        predicted_h1 = x_pred[STATE_H1]
        r[STATE_H1] = y[STATE_H1] - predicted_h1

        # Tank 3: mass conservation
        area_3 = TANK_PARAMS["T3"]["area"]
        predicted_h3 = x_pred[STATE_H3]
        r[STATE_H3] = y[STATE_H3] - predicted_h3

        # Tank 4: mass conservation
        predicted_h4 = x_pred[STATE_H4]
        r[STATE_H4] = y[STATE_H4] - predicted_h4

        # Differential pressure (energy domain)
        r[STATE_DP3] = y[STATE_DP3] - x_pred[STATE_DP3]

        return r

    def _compute_chemistry_residuals(self, y: np.ndarray, x_pred: np.ndarray,
                                      u: np.ndarray) -> np.ndarray:
        """
        Chemistry domain: stoichiometry and reaction kinetics.
        Only applies to pH, ORP, conductivity sensors.
        Level sensors get zero chemistry residual (vacuously satisfied).
        """
        r = np.zeros(N_STATES)

        # pH: compare measured pH change to predicted from dosing
        r[STATE_PH] = y[STATE_PH] - x_pred[STATE_PH]

        # ORP: compare to predicted from chlorination model
        r[STATE_ORP] = y[STATE_ORP] - x_pred[STATE_ORP]

        # Conductivity: compare to mixing/dilution model
        r[STATE_COND] = y[STATE_COND] - x_pred[STATE_COND]

        return r

    def _compute_math_residuals(self, y: np.ndarray, x_pred: np.ndarray,
                                 u: np.ndarray) -> np.ndarray:
        """
        Math domain: algebraic/geometric invariants.
        These are instantaneous consistency checks, not predictions.
        """
        r = np.zeros(N_STATES)

        # Volume-level consistency: V_integrated vs A*h_measured
        areas = [TANK_PARAMS[t]["area"] for t in ["T1", "T3", "T4"]]
        levels = [y[STATE_H1], y[STATE_H3], y[STATE_H4]]

        if self.initial_total_volume is not None:
            r_global = total_water_mass_conservation(
                levels, areas,
                self.total_inflow_integral,
                self.total_outflow_integral,
                self.initial_total_volume,
            )
            # Distribute global mass error across tank sensors
            for i, state_idx in enumerate([STATE_H1, STATE_H3, STATE_H4]):
                r[state_idx] = r_global / len(levels)

        return r

    def flag_sensor(self, state_idx: int):
        """Mark a sensor as untrusted. Its Kalman correction is zeroed out."""
        self.trust_mask[state_idx] = 0.0

    def trust_sensor(self, state_idx: int):
        """Restore trust in a sensor."""
        self.trust_mask[state_idx] = 1.0

    def get_residual_magnitudes(self) -> Optional[Dict[str, float]]:
        """Return the magnitude of the latest residuals for each domain."""
        if not self.residual_history:
            return None
        latest = self.residual_history[-1]
        return {
            "physics": float(np.linalg.norm(latest["physics"])),
            "chemistry": float(np.linalg.norm(latest["chemistry"])),
            "math": float(np.linalg.norm(latest["math"])),
            "combined": float(np.linalg.norm(latest["combined"])),
        }
