"""
PhysAttest — Component 1: Multi-Domain Observer
State-space model + Kalman residual generator for the SWaT testbed.

Physical layout modeled (6-stage SWaT process, iTrust testbed):
  P1  Raw water intake      -> tank LIT101, inflow FIT101, pumps P101/P102
  P2  Chemical dosing       -> pH/ORP/conductivity AIT201-203, dosing pumps P201-206
  P3  Ultrafiltration       -> tank LIT301, flows FIT201/FIT301, diff. pressure DPIT301
  P4  Dechlorination        -> tank LIT401, AIT401/402, UV dechlorinator
  P5  Reverse osmosis       -> pressures PIT501-503, flows FIT501-504, AIT501-504
  P6  RO permeate/backwash  -> tank LIT601, flow FIT601

This file implements the pattern for the full inter-tank network (Layer 1,
handwritten conservation laws). Extend TANKS / PIPES / CHEM_SPECIES to cover
all six stages -- the math and code do not change, only the config grows.
"""

import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 1. PLANT CONFIGURATION
# ---------------------------------------------------------------------------

@dataclass
class Tank:
    name: str                  # e.g. "LIT101"
    area_m2: float             # cross-sectional area A_i
    max_level_m: float
    min_level_m: float = 0.0


@dataclass
class Pipe:
    """A directed flow path between two tanks (or tank <-> environment),
    driven by a pump or gravity valve."""
    name: str                  # e.g. "P101" or "MV101"
    src: str                   # tank name, or "ENV" for external source
    dst: str                   # tank name, or "ENV" for sink
    nominal_flow_m3s: float    # rated flow when fully ON / open
    flow_sensor: str = None    # e.g. "FIT101" if this pipe is instrumented


@dataclass
class ChemSpecies:
    """A dosed/decaying chemical quantity tracked in a tank (chlorine,
    pH-active reagent, etc.) -- feeds the chemistry residual."""
    name: str                  # e.g. "FreeChlorine_P2"
    tank: str                  # which tank this concentration lives in
    decay_rate_per_s: float    # k in dC/dt = -kC + dosing/V
    sensor: str = None         # e.g. "AIT202"


# --- SWaT configuration (extend to all 6 stages as you get plant docs) -----

TANKS = [
    Tank("LIT101", area_m2=1.5, max_level_m=1.2),   # raw water tank
    Tank("LIT301", area_m2=1.2, max_level_m=1.2),   # UF feed tank
    Tank("LIT401", area_m2=1.2, max_level_m=1.2),   # RO feed tank
    Tank("LIT601", area_m2=1.0, max_level_m=1.0),   # permeate tank
]

PIPES = [
    Pipe("P101", src="ENV",    dst="LIT101", nominal_flow_m3s=0.0025, flow_sensor="FIT101"),
    Pipe("P301", src="LIT101", dst="LIT301", nominal_flow_m3s=0.0022, flow_sensor="FIT201"),
    Pipe("P302", src="LIT301", dst="LIT401", nominal_flow_m3s=0.0020, flow_sensor="FIT301"),
    Pipe("P401", src="LIT401", dst="LIT601", nominal_flow_m3s=0.0018, flow_sensor="FIT401"),
    Pipe("P601", src="LIT601", dst="ENV",    nominal_flow_m3s=0.0018, flow_sensor="FIT601"),
]

CHEM_SPECIES = [
    ChemSpecies("FreeChlorine", tank="LIT301", decay_rate_per_s=2.0e-4, sensor="AIT202"),
]


# ---------------------------------------------------------------------------
# 2. MATH — see MATH_FORMULATION.md for full derivation. Summary:
#
# State vector (physics domain):
#   x_phys = [L_1, ..., L_n]^T           tank levels (n = len(TANKS))
#   u      = [u_1, ..., u_m]^T           pump/valve command in {0,1} (or duty cycle)
#
# Continuity equation per tank i (Layer-1 conservation of mass, constant
# density -> volume conservation):
#     dL_i/dt = (1/A_i) * ( sum_{pipes into i} Q_p  -  sum_{pipes out of i} Q_p )
#     Q_p     = nominal_flow_p * u_p        (linear actuator model)
#
# So in matrix form:  xdot = A x + B u + w        (A = 0 here: no self-decay
# of a static tank level; all dynamics enter through B). This is the
# "physics" state-space block.
#
# Chemistry domain adds, per species:
#     dC/dt = -k_decay * C + (dosing_rate / V) * u_dose - k_flow * C * (Q_out/V)
# linearized around operating concentration C0, V0 -> another (A_c, B_c) block.
#
# Measurement model:
#     y = C_obs x + v,   v ~ N(0, R)
# where C_obs simply selects the measured sensors (level/flow/chem sensors
# that exist on the real plant), since SWaT sensors read state variables
# (or, for flow, a component of the input model) directly.
#
# Residual:
#     r(t) = y(t) - C_obs * xhat(t|t-1)
# r ~ 0 under H0 (no attack, correct physics); r grows under spoofing because
# a spoofed sensor breaks the algebraic relationship the *other*, honest
# sensors still obey -- that's the whole defense.
# ---------------------------------------------------------------------------


class PhysicsBlock:
    """Layer-1 conservation-of-mass state-space block for the tank network."""

    def __init__(self, tanks: list[Tank], pipes: list[Pipe]):
        self.tanks = tanks
        self.pipes = pipes
        self.tank_idx = {t.name: i for i, t in enumerate(tanks)}
        self.n = len(tanks)
        self.m = len(pipes)
        self.A, self.B = self._build_matrices()

    def _build_matrices(self):
        n, m = self.n, self.m
        A = np.zeros((n, n))          # static tanks: no self-dynamics
        B = np.zeros((n, m))          # inflow/outflow coupling
        for j, p in enumerate(self.pipes):
            q = p.nominal_flow_m3s
            if p.dst in self.tank_idx:
                i = self.tank_idx[p.dst]
                B[i, j] += q / self.tanks[i].area_m2
            if p.src in self.tank_idx:
                i = self.tank_idx[p.src]
                B[i, j] -= q / self.tanks[i].area_m2
        return A, B

    def step(self, x, u, dt):
        """Euler-integrate one step: x_{k+1} = x_k + dt * (A x + B u)."""
        xdot = self.A @ x + self.B @ u
        return x + dt * xdot

    def flow_consistency_residual(self, u_cmd, q_measured):
        """
        MATH-domain residual (Kirchhoff-style): the commanded actuator state
        times its nominal flow must match the *measured* flow sensor, for
        every instrumented pipe. This is a pure algebraic constraint --
        no dynamics, no chemistry -- hence it lives in the "math" residual
        channel rather than physics or chemistry.
        """
        r = []
        for j, p in enumerate(self.pipes):
            if p.flow_sensor is not None:
                q_expected = p.nominal_flow_m3s * u_cmd[j]
                r.append(q_measured[p.flow_sensor] - q_expected)
        return np.array(r)


class ChemistryBlock:
    """Layer-1 mass-balance block for dosed/decaying chemical species."""

    def __init__(self, species: list[ChemSpecies], tank_volume_m3: dict):
        self.species = species
        self.tank_volume_m3 = tank_volume_m3

    def step(self, c, dosing_rate, dt):
        c_next = np.array(c, dtype=float)
        for i, s in enumerate(self.species):
            V = self.tank_volume_m3[s.tank]
            cdot = -s.decay_rate_per_s * c[i] + dosing_rate[i] / V
            c_next[i] = c[i] + dt * cdot
        return c_next


class KalmanResidualGenerator:
    """
    Standard linear KF used purely as a *residual generator* for the physics
    block:  xhat_{k|k-1} = A xhat + B u ;  r_k = y_k - C xhat_{k|k-1}
    Q, R come from calibration on clean plant data (Layer-1 assumption:
    process/measurement noise is stationary Gaussian on the clean regime).
    """

    def __init__(self, A, B, C, Q, R, x0, P0):
        self.A, self.B, self.C = A, B, C
        self.Q, self.R = Q, R
        self.x = x0.copy()
        self.P = P0.copy()

    def predict(self, u, dt):
        self.x = self.x + dt * (self.A @ self.x + self.B @ u)
        F = np.eye(len(self.x)) + dt * self.A
        self.P = F @ self.P @ F.T + self.Q

    def update(self, y):
        r = y - self.C @ self.x                       # <-- the residual r(t)
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)
        self.x = self.x + K @ r
        self.P = (np.eye(len(self.x)) - K @ self.C) @ self.P
        return r, S


class MultiDomainObserver:
    """
    Component 1, top-level object. Computes r_physics, r_chemistry, r_math
    each timestep and exposes them to Theorem 2's information bound and to
    Component 5 (self-healing).
    """

    def __init__(self, physics_block: PhysicsBlock, kf: KalmanResidualGenerator,
                 chem_block: ChemistryBlock = None, c0=None):
        self.physics_block = physics_block
        self.kf = kf
        self.chem_block = chem_block
        # c_hat is the chemistry domain's own one-step-ahead prediction state,
        # exactly analogous to xhat inside the physics KF.
        self.c_hat = np.array(c0, dtype=float) if c0 is not None else None

    def step(self, u_cmd, y_levels, y_flows, y_chem=None, dosing_cmd=None, dt=1.0):
        # --- physics residual: KF innovation on tank levels ---
        self.kf.predict(u_cmd, dt)
        r_phys, _ = self.kf.update(y_levels)

        # --- math residual: actuator/flow-sensor algebraic consistency ---
        r_math = self.physics_block.flow_consistency_residual(u_cmd, y_flows)

        # --- chemistry residual: one-step-ahead mass-balance prediction ---
        # dosing_cmd is the KNOWN commanded dosing rate (a controllable input,
        # exactly analogous to u_cmd on the flow side) -- set by the PLC, not
        # something we have to guess.
        r_chem = np.array([])
        if self.chem_block is not None and y_chem is not None:
            y_c = np.array([y_chem[s.sensor] for s in self.chem_block.species])
            if self.c_hat is None:
                self.c_hat = y_c.copy()
            r_chem = y_c - self.c_hat                       # innovation, like r_phys
            dosing = (dosing_cmd if dosing_cmd is not None
                      else np.zeros(len(self.chem_block.species)))
            # simple persistence-corrected predictor: nudge toward the
            # measurement (like a KF with fixed small gain), then propagate
            self.c_hat = self.chem_block.step(c=y_c, dosing_rate=dosing, dt=dt)

        return {"r_physics": r_phys, "r_chemistry": r_chem, "r_math": r_math}
