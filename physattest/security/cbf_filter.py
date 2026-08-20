"""
CBF Safety Filter for PhysAttest.

Implements Control Barrier Functions as a QP (CVXPY) to enforce
plant safety regardless of agent intent. Every agent command passes
through this filter before reaching actuators.
"""

import numpy as np
import cvxpy as cp
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PlantConfig:
    """Physical parameters for a water treatment plant (SWaT-like)."""

    n_tanks: int = 3
    n_actuators: int = 6  # pumps + valves

    # Tank cross-section areas (m²)
    tank_areas: np.ndarray = field(default_factory=lambda: np.array([1.5, 1.5, 2.0]))

    # Water level bounds (m)
    level_min: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.2, 0.3]))
    level_max: np.ndarray = field(default_factory=lambda: np.array([1.2, 1.2, 1.5]))

    # Pressure bounds (kPa)
    pressure_max: np.ndarray = field(default_factory=lambda: np.array([500.0, 500.0]))

    # Actuator bounds (normalised 0-1)
    u_min: np.ndarray = field(default_factory=lambda: np.zeros(6))
    u_max: np.ndarray = field(default_factory=lambda: np.ones(6))

    # CBF tuning — how aggressively to repel from boundaries
    alpha: float = 0.5


@dataclass
class PlantState:
    """Current verified state from the observer (not raw sensors)."""

    levels: np.ndarray       # tank water levels (m)
    pressures: np.ndarray    # pipe pressures (kPa)
    flows_in: np.ndarray     # current inflows per tank (m³/s)
    flows_out: np.ndarray    # current outflows per tank (m³/s)


class CBFSafetyFilter:
    """
    Quadratic-program safety filter.

    Given an agent's proposed command u_agent, solve:

        u* = argmin ||u - u_agent||²
              u
             s.t.  ḣ_i(x, u) + α·h_i(x) ≥ 0   ∀ i
                   u_min ≤ u ≤ u_max

    If u_agent is already safe, u* = u_agent (zero modification).
    """

    def __init__(self, config: PlantConfig):
        self.cfg = config
        self._build_qp()

    def _build_qp(self):
        n = self.cfg.n_actuators

        # Decision variable: the safe command
        self.u = cp.Variable(n, name="u_safe")

        # Parameter: the agent's proposed command (changes every call)
        self.u_agent = cp.Parameter(n, name="u_agent")

        # Parameters for barrier values and derivatives (set per call)
        # We build constraints parametrically so CVXPY can warm-start.
        n_barriers = 2 * self.cfg.n_tanks + len(self.cfg.pressure_max)
        self.h_vals = cp.Parameter(n_barriers, name="h_vals")

        # G_cbf @ u >= rhs_cbf  encodes  ḣ(x,u) + α·h(x) ≥ 0
        self.G_cbf = cp.Parameter((n_barriers, n), name="G_cbf")
        self.rhs_cbf = cp.Parameter(n_barriers, name="rhs_cbf")

        # Objective: minimise ||u - u_agent||²
        objective = cp.Minimize(cp.sum_squares(self.u - self.u_agent))

        # Constraints
        constraints = [
            self.G_cbf @ self.u >= self.rhs_cbf,  # CBF safety
            self.u >= self.cfg.u_min,               # actuator lower
            self.u <= self.cfg.u_max,               # actuator upper
        ]

        self.problem = cp.Problem(objective, constraints)

    def _compute_barriers(self, state: PlantState):
        """
        Compute barrier values h(x) and the linear map from u to ḣ.

        Returns (h_vals, G_cbf, rhs_cbf) where
            G_cbf @ u >= rhs_cbf
        encodes ḣ(x,u) + α·h(x) ≥ 0 for every barrier.
        """
        cfg = self.cfg
        n = cfg.n_actuators
        barriers_h = []
        G_rows = []
        rhs_rows = []

        # --- Tank level barriers ---
        # Tank dynamics: dL/dt = (1/A)(Q_in - Q_out)
        # Actuator mapping (SWaT-like):
        #   u[0], u[1]: pump speeds controlling inflow to tanks 0, 1
        #   u[2]: pump speed controlling inflow to tank 2
        #   u[3], u[4], u[5]: outlet valve positions for tanks 0, 1, 2
        #
        # Q_in_i  = u[i]     * Q_max_pump   (simplified linear model)
        # Q_out_i = u[i+3]   * Q_max_valve
        Q_max_pump = 0.005   # m³/s at full speed
        Q_max_valve = 0.005  # m³/s at full open

        for i in range(cfg.n_tanks):
            A = cfg.tank_areas[i]

            # --- Upper barrier: h = L_max - L ---
            h_up = cfg.level_max[i] - state.levels[i]
            barriers_h.append(h_up)

            # ḣ_up = -dL/dt = -(1/A)(Q_in - Q_out)
            #       = -(1/A)(u[i]*Q_max_pump - u[i+3]*Q_max_valve)
            # CBF: ḣ_up + α*h_up ≥ 0
            # → -(1/A)*Q_max_pump * u[i] + (1/A)*Q_max_valve * u[i+3] + α*h_up ≥ 0
            # → G_row @ u ≥ -α*h_up
            g_up = np.zeros(n)
            g_up[i] = -(1 / A) * Q_max_pump       # inflow pump
            g_up[i + 3] = (1 / A) * Q_max_valve    # outlet valve
            G_rows.append(g_up)
            rhs_rows.append(-cfg.alpha * h_up)

            # --- Lower barrier: h = L - L_min ---
            h_lo = state.levels[i] - cfg.level_min[i]
            barriers_h.append(h_lo)

            # ḣ_lo = dL/dt = (1/A)(Q_in - Q_out)
            g_lo = np.zeros(n)
            g_lo[i] = (1 / A) * Q_max_pump
            g_lo[i + 3] = -(1 / A) * Q_max_valve
            G_rows.append(g_lo)
            rhs_rows.append(-cfg.alpha * h_lo)

        # --- Pressure barriers ---
        # Simplified: pressure ~ proportional to pump speed
        # P_j = k * u[j] where k maps pump to downstream pressure
        k_pressure = 600.0  # kPa at full pump speed
        for j, p_max in enumerate(cfg.pressure_max):
            h_p = p_max - state.pressures[j]
            barriers_h.append(h_p)

            # ḣ_p ≈ -k * du/dt  — for static model, constrain u directly:
            # h_p = P_max - k*u[j], so ḣ = -k*du/dt
            # Simpler: treat as static barrier  P_max - k*u[j] ≥ margin
            # CBF becomes: -k*u[j] + P_max ≥ (1-α)*something
            # For a static constraint we just need: k*u[j] ≤ P_max
            g_p = np.zeros(n)
            g_p[j] = -k_pressure / p_max  # normalised
            G_rows.append(g_p)
            rhs_rows.append(-cfg.alpha * h_p / p_max)

        return (
            np.array(barriers_h),
            np.array(G_rows),
            np.array(rhs_rows),
        )

    def filter(
        self, u_agent: np.ndarray, state: PlantState
    ) -> dict:
        """
        Filter an agent command through the CBF.

        Args:
            u_agent: proposed command vector (n_actuators,)
            state: current observer-verified plant state

        Returns:
            dict with:
                u_safe:       the filtered command
                modified:     True if the command was changed
                intervention: L2 norm of modification
                h_values:     barrier values (all should be ≥ 0)
                feasible:     True if QP was feasible
        """
        h_vals, G, rhs = self._compute_barriers(state)

        self.u_agent.value = u_agent
        self.h_vals.value = h_vals
        self.G_cbf.value = G
        self.rhs_cbf.value = rhs

        self.problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)

        if self.problem.status in ("infeasible", "infeasible_inaccurate"):
            # Emergency: no feasible safe command exists.
            # Fall back to most conservative command (all off).
            u_safe = np.zeros_like(u_agent)
            feasible = False
        else:
            u_safe = self.u.value
            feasible = True

        modification = np.linalg.norm(u_safe - u_agent)

        return {
            "u_safe": u_safe,
            "u_agent": u_agent,
            "modified": modification > 1e-4,
            "intervention": float(modification),
            "h_values": h_vals,
            "feasible": feasible,
        }


def demo_hijacked_agent():
    """
    Demonstrate the CBF blocking a dangerous command from a hijacked agent.

    Scenario: Tank 0 is at 1.1m (near the 1.2m overflow limit).
    A hijacked agent commands: "all pumps to maximum, all valves closed."
    This would overflow the tank. The CBF intervenes.
    """
    config = PlantConfig()
    cbf = CBFSafetyFilter(config)

    # Current plant state (from observer, not raw sensors)
    state = PlantState(
        levels=np.array([1.1, 0.6, 0.8]),       # tank 0 near overflow!
        pressures=np.array([300.0, 250.0]),
        flows_in=np.array([0.003, 0.002, 0.002]),
        flows_out=np.array([0.002, 0.002, 0.002]),
    )

    print("=" * 60)
    print("PhysAttest CBF Safety Filter — Demo")
    print("=" * 60)

    # --- Scenario 1: Normal safe command ---
    u_normal = np.array([0.4, 0.3, 0.3, 0.5, 0.5, 0.5])
    result = cbf.filter(u_normal, state)
    print("\n[Scenario 1] Normal command from honest agent")
    print(f"  Agent wants:  {u_normal}")
    print(f"  CBF outputs:  {np.round(result['u_safe'], 4)}")
    print(f"  Modified:     {result['modified']}")
    print(f"  Intervention: {result['intervention']:.4f}")

    # --- Scenario 2: Hijacked agent — overflow attack ---
    u_attack = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    result = cbf.filter(u_attack, state)
    print("\n[Scenario 2] HIJACKED agent: all pumps max, all valves closed")
    print(f"  Agent wants:  {u_attack}")
    print(f"  CBF outputs:  {np.round(result['u_safe'], 4)}")
    print(f"  Modified:     {result['modified']}")
    print(f"  Intervention: {result['intervention']:.4f}")
    print(f"  Barrier vals: {np.round(result['h_values'], 4)}")

    # --- Scenario 3: Subtle attack — slightly too much inflow ---
    u_subtle = np.array([0.8, 0.3, 0.3, 0.1, 0.5, 0.5])
    result = cbf.filter(u_subtle, state)
    print("\n[Scenario 3] Subtle attack: high pump 0 + low valve 0")
    print(f"  Agent wants:  {u_subtle}")
    print(f"  CBF outputs:  {np.round(result['u_safe'], 4)}")
    print(f"  Modified:     {result['modified']}")
    print(f"  Intervention: {result['intervention']:.4f}")

    # --- Scenario 4: Safe state, aggressive command is fine ---
    safe_state = PlantState(
        levels=np.array([0.5, 0.5, 0.5]),
        pressures=np.array([200.0, 200.0]),
        flows_in=np.array([0.003, 0.003, 0.003]),
        flows_out=np.array([0.003, 0.003, 0.003]),
    )
    result = cbf.filter(u_attack, safe_state)
    print("\n[Scenario 4] Same attack command, but tank is half-full")
    print(f"  Agent wants:  {u_attack}")
    print(f"  CBF outputs:  {np.round(result['u_safe'], 4)}")
    print(f"  Modified:     {result['modified']}")
    print(f"  Intervention: {result['intervention']:.4f}")
    print(f"  (More room → less intervention needed)")

    # --- Scenario 5: Truly safe command passes through unchanged ---
    u_gentle = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])
    result = cbf.filter(u_gentle, state)
    print("\n[Scenario 5] Gentle command: low pumps, open valves (near-full tank)")
    print(f"  Agent wants:  {u_gentle}")
    print(f"  CBF outputs:  {np.round(result['u_safe'], 4)}")
    print(f"  Modified:     {result['modified']}")
    print(f"  Intervention: {result['intervention']:.6f}")
    print(f"  (Safe command → zero or near-zero modification)")

    print("\n" + "=" * 60)
    print("Key: CBF is a QP — no text input, no prompt injection possible.")
    print("Even a fully hijacked LLM agent cannot damage the plant.")
    print("=" * 60)


if __name__ == "__main__":
    demo_hijacked_agent()
