"""
Mathematical (algebraic/geometric) invariants for water treatment.
These are relationships that must hold at every instant — not ODEs.
"""

import numpy as np


def volume_level_consistency(level, tank_area, volume_from_flow_integration):
    """
    Geometric invariant: V = A × h for a cylindrical/prismatic tank.

    If you independently track volume by integrating flow and also measure
    level, they must agree.

    Returns residual in m³.
    """
    volume_from_level = tank_area * level
    return volume_from_level - volume_from_flow_integration


def flow_balance_across_stages(flows_between_stages):
    """
    At steady state, flow entering each stage = flow leaving it.
    In transient, the difference equals the rate of accumulation
    (which is captured by mass conservation).

    flows_between_stages: list of (Q_in, Q_out) per stage.
    Returns array of residuals.
    """
    residuals = []
    for q_in, q_out in flows_between_stages:
        residuals.append(q_in - q_out)
    return np.array(residuals)


def pressure_level_consistency(pressure_bottom, level, rho=998.0, g=9.81):
    """
    P_bottom = ρgh + P_atm is both physics AND math.
    Here we check the algebraic identity without modeling dynamics.

    Returns residual in Pascals.
    """
    predicted_p = rho * g * level
    return pressure_bottom - predicted_p


def ratio_invariant(sensor_a, sensor_b, expected_ratio):
    """
    Some sensor pairs maintain a fixed ratio set by plant geometry.
    Example: two flow sensors on the same pipe with different diameters.
        Q₁/Q₂ = (A₁/A₂) = constant

    Returns residual (should be ≈ 0).
    """
    if abs(sensor_b) < 1e-10:
        return 0.0
    actual_ratio = sensor_a / sensor_b
    return actual_ratio - expected_ratio


def total_water_mass_conservation(all_tank_levels, tank_areas, total_inflow_integral,
                                   total_outflow_integral, initial_total_volume):
    """
    Global invariant: total water in the system is conserved.
        Σ(Aᵢhᵢ) = V_initial + ∫Q_in dt - ∫Q_out dt

    This catches attacks that are locally consistent but violate the
    global mass balance.
    """
    current_total = sum(a * h for a, h in zip(tank_areas, all_tank_levels))
    expected_total = initial_total_volume + total_inflow_integral - total_outflow_integral
    return current_total - expected_total
