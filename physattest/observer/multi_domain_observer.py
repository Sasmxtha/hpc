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
from typing import Dict, List, Optional, Tuple

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
from .sindy_discovery import discover_equations, refine_observer_matrices


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

STATE_NAMES = ["h1", "h3", "h4", "pH", "ORP", "cond", "dP3"]  # positional, matches STATE_* order

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

INPUT_NAMES = [  # positional, matches INPUT_* order -- used as SINDy feature_names
    "MV101", "P101", "P201", "P301", "P302", "P501", "P601", "UV401"
]

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

        # Raw (y, u) trajectory history, bounded, for SINDy Layer 2 refinement --
        # residual_history alone isn't enough to run SINDy on, since SINDy needs the actual
        # state/control trajectory (what SINDy fits equations TO), not the gap between
        # measurement and the current model's prediction.
        self.y_history: List[np.ndarray] = []
        self.u_history: List[np.ndarray] = []
        self._history_cap = 3000

        # Layer 3 PINN fallbacks, registered per state index via set_pinn_fallback(). Only
        # states where Layer 1 (handwritten) + Layer 2 (SINDy) are known to be a poor fit
        # for genuinely nonlinear dynamics should get one -- see set_pinn_fallback's
        # docstring. Empty until explicitly registered; step() falls back to the Layer 1/2
        # linear prediction for any state with no PINN registered.
        self._pinn_fallbacks: Dict[int, dict] = {}

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
        # --- Step 1: Predict (Layer 1/2: handwritten + SINDy-refined linear model) ---
        x_pred = self.A_d @ self.x_hat + self.B_d @ u

        # --- Step 1b: Layer 3 PINN fallback, for any state where one is registered ---
        # Overrides the linear prediction for that state only; every other state keeps its
        # Layer 1/2 prediction untouched. See set_pinn_fallback() for when this is warranted.
        for state_idx, fallback in self._pinn_fallbacks.items():
            x_pred[state_idx] = fallback["predict"](self.x_hat, u, self.dt)

        # --- Track raw (y, u) trajectory for SINDy Layer 2 refinement ---
        self.y_history.append(y.copy())
        self.u_history.append(u.copy())
        if len(self.y_history) > self._history_cap:
            self.y_history = self.y_history[-self._history_cap:]
            self.u_history = self.u_history[-self._history_cap:]

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

    def refine_with_sindy(
        self, min_samples: int = 300, threshold: float = 1e-5, blend_factor: float = 0.3
    ) -> Optional[Dict]:
        """
        Layer 2: refine the Layer 1 handwritten A/B matrices using SINDy-discovered
        coefficients from the actual (y, u) trajectory this observer has seen.

        Returns None (and does nothing) if fewer than min_samples of history have
        accumulated yet -- SINDy needs a reasonably long, control-input-excited trajectory
        to identify coefficients reliably, matching the persistent-excitation requirement
        documented in sindy_discovery.py's own synthetic-data generator. Call this
        periodically (e.g. every few hundred steps) once the plant has been running long
        enough, not on every step -- refitting SINDy every cycle would be wasteful and would
        make A_d/B_d (and therefore the Kalman gain) jitter step to step.

        threshold defaults small (1e-5), not PySINDy's usual 0.01-0.05: this observer's
        states are in physical units (metres, pH, mV, uS/cm) driven by pump flow rates in
        m^3/s divided by tank areas in m^2, so genuine coefficients here are routinely
        ~1e-4-1e-3 in magnitude. A default sparsity threshold sized for the spec's own
        thermal example (coefficients ~0.02-0.04) silently zeroes out every real coefficient
        for a state like h1 -- confirmed directly: threshold=0.01 eliminated ALL of h1's
        terms and left the "refined" model bitwise identical to Layer 1 alone.

        After a successful refinement, A_d/B_d are re-discretized and the Kalman gain is
        recomputed, since both are functions of A_c/A_d which just changed.
        """
        if len(self.y_history) < min_samples:
            return None

        state_data = np.array(self.y_history)
        input_data = np.array(self.u_history)
        feature_names = STATE_NAMES + INPUT_NAMES

        result = discover_equations(
            state_data=state_data,
            input_data=input_data,
            dt=self.dt,
            feature_names=feature_names,
            threshold=threshold,
            max_poly_degree=1,  # match Layer 1's linear structure -- SINDy refines
            # coefficients of the SAME linear terms Layer 1 already has, not new nonlinear
            # ones (that is Layer 3 / PINN's job, per the spec's own layering).
        )

        self.A_c, self.B_c = refine_observer_matrices(result, self.A_c, self.B_c, blend_factor)
        self.A_d, self.B_d = self._discretize(self.A_c, self.B_c, self.dt)
        self.L = self._compute_kalman_gain()

        return result

    def set_pinn_fallback(self, state_idx: int, pinn_model, u_index: Optional[int], predict_fn) -> None:
        """
        Layer 3: register a PINN as the prediction source for one state, replacing the
        Layer 1/2 linear prediction for that state only.

        Only warranted for a state whose true dynamics are genuinely nonlinear in a way
        neither Layer 1 (hand-derived linear coefficients) nor Layer 2 (SINDy, which this
        observer also restricts to degree-1/linear terms, matching Layer 1's structure) can
        represent -- per the spec, PINN is for "relationships too complex for SINDy," not a
        blanket replacement. In this plant, that's STATE_DP3 (membrane fouling): fouling
        accumulates roughly with flow-RATE-SQUARED, not flow-linear, which a linear A/B
        matrix structurally cannot express regardless of how its coefficients are fitted.
        See physattest/observer/pinn.py's membrane_fouling_residual/PINN_DP3 for the trained
        model and the validation showing it actually helps (physattest/observer/pinn.py's
        __main__, dp3_fouling section).

        predict_fn(x_hat, u, dt) -> float is the caller-supplied glue between this
        observer's state representation and the PINN's own (t, control) input convention --
        kept as an explicit callback rather than hardcoding the PINN's calling convention
        here, since different states may need different PINNs with different control inputs.
        """
        self._pinn_fallbacks[state_idx] = {
            "model": pinn_model,
            "u_index": u_index,
            "predict": predict_fn,
        }

    def clear_pinn_fallback(self, state_idx: int) -> None:
        self._pinn_fallbacks.pop(state_idx, None)

    def enable_dp3_pinn_fallback(self, pinn_model=None) -> None:
        """Registers the membrane-fouling PINN as STATE_DP3's Layer 3 fallback (see
        set_pinn_fallback's docstring for why DP3 specifically). Trains a fresh model
        (physattest.observer.pinn.train_dp3_pinn, a few thousand epochs) if pinn_model isn't
        supplied -- pass an already-trained model (e.g. loaded from a persisted checkpoint)
        to avoid re-training on every call.
        """
        from .pinn import make_dp3_predict_fn, train_dp3_pinn

        if pinn_model is None:
            pinn_model = train_dp3_pinn()
        predict_fn = make_dp3_predict_fn(pinn_model, u_uf_idx=INPUT_P301, u_bw_idx=INPUT_P601)
        self.set_pinn_fallback(STATE_DP3, pinn_model, u_index=None, predict_fn=predict_fn)

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
