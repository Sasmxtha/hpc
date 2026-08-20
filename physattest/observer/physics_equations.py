"""
Conservation laws for water treatment plants (SWaT/WADI).
Each function returns a residual: measured - predicted.
Near-zero residual = sensor is honest. Large residual = anomaly.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
RHO_WATER = 998.0       # kg/m³ at ~20°C
G = 9.81                # m/s²
P_ATM = 101325.0        # Pa


# ---------------------------------------------------------------------------
# SWaT plant geometry (from iTrust documentation)
# Approximate values — refine after you inspect the actual dataset ranges.
# ---------------------------------------------------------------------------
TANK_PARAMS = {
    "T1": {"area": 1.5, "max_level": 1.0},   # Stage 1 raw water tank
    "T3": {"area": 1.5, "max_level": 1.0},   # Stage 3 UF feed tank
    "T4": {"area": 1.5, "max_level": 1.2},   # Stage 4 RO feed tank
}

PUMP_FLOW_RATES = {
    "P101": 2.5,   # m³/h nominal (convert to m³/s in code)
    "P301": 2.0,
    "P302": 2.0,
    "P501": 1.5,
    "P601": 1.5,
}


def mass_conservation_tank(dh_dt, area, q_in, q_out):
    """
    Conservation of mass for an incompressible fluid in a tank.

    Physics: d(ρV)/dt = ρQ_in - ρQ_out
    Since ρ is constant and V = A*h:
        A * dh/dt = Q_in - Q_out

    Returns residual: should be ≈ 0 if sensors are honest.
    """
    predicted_dh_dt = (q_in - q_out) / area
    return dh_dt - predicted_dh_dt


def hydrostatic_pressure(h_measured, p_measured):
    """
    P_bottom = ρgh + P_atm

    If we measure both level and bottom pressure, they must agree.
    Returns residual in Pascals.
    """
    predicted_p = RHO_WATER * G * h_measured + P_ATM
    return p_measured - predicted_p


def flow_conservation_junction(flows_in, flows_out):
    """
    Kirchhoff's current law for fluids: ΣQ_in = ΣQ_out at every junction.
    No water is created or destroyed in a pipe junction.

    Returns residual (should be ≈ 0).
    """
    return np.sum(flows_in) - np.sum(flows_out)


def energy_conservation_pump(q_in, p_in, q_out, p_out, pump_power, efficiency=0.7):
    """
    Conservation of energy across a pump:
    P_pump * η = Q * (P_out - P_in) + ½ρQ(v_out² - v_in²)

    Simplified (neglecting velocity head for low-speed flow):
    P_pump * η ≈ Q * ΔP

    Returns residual in Watts.
    """
    delta_p = p_out - p_in
    q_avg = (q_in + q_out) / 2.0
    predicted_power = q_avg * delta_p / efficiency
    return pump_power - predicted_power


def bernoulli_pipe(p1, v1, p2, v2, h1=0, h2=0, friction_loss=0):
    """
    Bernoulli's equation along a streamline (steady, incompressible):
    P₁ + ½ρv₁² + ρgh₁ = P₂ + ½ρv₂² + ρgh₂ + ΔP_friction

    Returns residual in Pascals.
    """
    lhs = p1 + 0.5 * RHO_WATER * v1**2 + RHO_WATER * G * h1
    rhs = p2 + 0.5 * RHO_WATER * v2**2 + RHO_WATER * G * h2 + friction_loss
    return lhs - rhs
