"""
Chemistry domain equations for water treatment.
Applies to Stage 2 (chemical dosing) and Stage 5 (RO/chlorination).
"""

import numpy as np


def ph_dosing_response(ph_current, dosing_rate, flow_rate, tank_volume,
                       buffer_capacity=0.01, target_ph=7.0, decay_rate=0.001):
    """
    pH change from chemical dosing in a mixing tank.

    Chemistry: Henderson-Hasselbalch + mixing dynamics.
    Simplified model:
        dpH/dt = (1/β) × (C_dose × Q_dose / V) - λ(pH - pH_eq)

    Where:
        β = buffer capacity (mol/L per pH unit) — how resistant the water is
        C_dose × Q_dose = moles of acid/base added per second
        λ = natural equilibration rate
        pH_eq = equilibrium pH of the source water

    Returns predicted dpH/dt.
    """
    dosing_term = dosing_rate / (buffer_capacity * tank_volume)
    equilibration_term = decay_rate * (ph_current - target_ph)
    return dosing_term - equilibration_term


def orp_chlorination_response(orp_current, chlorine_dose_rate, flow_rate,
                               tank_volume, k_cl=50.0, k_decay=0.005):
    """
    ORP response to chlorine dosing.

    Chemistry: Free chlorine is an oxidizer → raises ORP.
        dORP/dt = (k_Cl × Q_Cl - k_decay × ORP) / V

    k_Cl = ORP gain per unit chlorine flow
    k_decay = natural ORP decay (chlorine consumed by organics)
    """
    return (k_cl * chlorine_dose_rate - k_decay * orp_current) / tank_volume


def conductivity_mixing(sigma_current, sigma_inlet, flow_in, flow_out, tank_volume):
    """
    Conductivity is a conservative tracer — no chemical reaction changes it,
    only mixing/dilution.

        dσ/dt = Q_in(σ_in - σ) / V

    This is just the CSTR (continuously stirred tank reactor) dilution equation.
    If conductivity changes without a flow change, something is wrong.
    """
    return flow_in * (sigma_inlet - sigma_current) / tank_volume


def chlorine_decay(cl_current, k_decay=0.01, cl_demand=0.0):
    """
    First-order chlorine decay:
        dCl/dt = -k_decay × Cl - Cl_demand

    Chlorine decays as it reacts with organics in the water.
    """
    return -k_decay * cl_current - cl_demand
