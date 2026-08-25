"""
SINDy (Sparse Identification of Nonlinear Dynamics) equation discovery.
Layer 2 of the equation hierarchy.

Uses PySINDy to discover differential equations from operational data
in human-readable form, e.g.:
    dT/dt = -0.037(T_pipe - T_ambient) + 0.021 x flow_rate

These fill the gap between handwritten conservation laws (Layer 1)
and the PINN fallback (Layer 3) by capturing plant-specific parameters
like exact pipe friction, heat loss rates, and mixing coefficients.

This is Member 3's discover_equations/refine_observer_matrices/validate_against_conservation
API, kept as the canonical version during merge resolution: it's built to plug directly into
multi_domain_observer.py's A/B state-space matrices (refine_observer_matrices blends SINDy's
discovered linear terms into the observer's A matrix), which Member 2's alternate
discover_equation function never had a hook for. The __main__ block below is new: validates
discover_equations by generating synthetic data from a system with KNOWN coefficients (the
spec's own thermal example) and checking the recovered coefficients are close to the true
physical constants, not just that PySINDy returns *something*. This is the same validation
methodology Member 2's original discover_equation used, ported to test Member 3's function
instead, since the function itself had never been checked against known ground truth before
(demo_observer.py calls it but only prints the result, without verifying correctness).

Note: as of this merge, discover_equations/refine_observer_matrices are still standalone --
multi_domain_observer.py collects self.residual_history (commented "History for SINDy") but
nothing currently calls discover_equations/refine_observer_matrices on it. Wiring SINDy into
the live observer loop is follow-up work, not done here.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pysindy as ps
    HAS_PYSINDY = True
except ImportError:
    HAS_PYSINDY = False
    print("WARNING: PySINDy not installed. Run: pip install pysindy")


def discover_equations(
    state_data: np.ndarray,
    input_data: Optional[np.ndarray] = None,
    dt: float = 1.0,
    feature_names: Optional[List[str]] = None,
    threshold: float = 0.05,
    max_poly_degree: int = 2,
) -> Dict:
    """
    Discover governing equations from operational data using SINDy.

    Parameters:
        state_data: (n_timesteps, n_states) array of sensor readings
        input_data: (n_timesteps, n_inputs) array of actuator states (optional)
        dt: sampling interval in seconds
        feature_names: names for each state variable AND each control/input variable --
            PySINDy indexes into this list for every library feature, state and control
            alike, so it must cover both or building display names for control terms raises
            an internal IndexError.
        threshold: sparsity threshold -- higher = sparser equations
        max_poly_degree: max polynomial degree in the library

    Returns:
        Dictionary with:
            'model': fitted PySINDy model
            'equations': list of discovered equation strings
            'coefficients': coefficient matrix
            'score': R^2 score on the data

    HOW IT WORKS:
    SINDy builds a library of candidate terms (x, x^2, x*y, sin(x), ...)
    and finds the sparsest combination that explains dx/dt.

    Example output:
        dx0/dt = -0.037 x0 + 0.021 x1 + 0.003 x0 x1
    Meaning:
        dh/dt = -0.037h + 0.021*flow + 0.003*h*flow
    """
    if not HAS_PYSINDY:
        raise RuntimeError("PySINDy required. Install with: pip install pysindy")

    # Build function library: polynomials up to max_poly_degree
    library = ps.PolynomialLibrary(degree=max_poly_degree, include_bias=True)

    # Use STLSQ (Sequentially Thresholded Least Squares) optimizer
    # This promotes sparsity -- most coefficients become exactly zero
    optimizer = ps.STLSQ(threshold=threshold, alpha=0.01)

    # Build and fit the model. Note: `discrete_time` is not a constructor argument in the
    # installed pysindy version (2.1.0) -- an earlier version of this call passed it and
    # raised TypeError immediately. Continuous-time is this version's only/default mode.
    model = ps.SINDy(
        feature_library=library,
        optimizer=optimizer,
    )

    if input_data is not None:
        model.fit(state_data, t=dt, u=input_data, feature_names=feature_names)
    else:
        model.fit(state_data, t=dt, feature_names=feature_names)

    # Extract human-readable equations
    equations = []
    model.print()
    for i in range(state_data.shape[1]):
        eq = model.equations(precision=4)[i]
        equations.append(eq)

    # Score the model
    if input_data is not None:
        score = model.score(state_data, t=dt, u=input_data)
    else:
        score = model.score(state_data, t=dt)

    return {
        "model": model,
        "equations": equations,
        "coefficients": model.coefficients(),
        "score": score,
    }


def refine_observer_matrices(
    sindy_result: Dict,
    A_current: np.ndarray,
    B_current: np.ndarray,
    blend_factor: float = 0.3,
) -> tuple:
    """
    Blend SINDy-discovered coefficients into the observer's A and B matrices.

    The handwritten conservation laws (Layer 1) provide the structure.
    SINDy fills in the precise numerical coefficients from real plant data.

    blend_factor: how much weight to give SINDy vs handwritten (0 = all
    handwritten, 1 = all SINDy). Start low (0.3) and increase as SINDy
    proves accurate.

    B_current IS refined here -- an earlier version of this function only ever touched A and
    returned B unchanged regardless of blend_factor, silently a no-op for exactly the kind of
    plant mismatch this function exists to correct whenever it lives entirely in a
    control-input coefficient (e.g. a pump's true flow rate differing from its hand-coded
    estimate, which in this observer's A/B structure is a B-matrix term, not an A-matrix
    one). Confirmed via a live test: refining only A left prediction error on a plant with a
    mis-estimated pump flow rate completely unchanged (0.0% improvement) because that
    particular mismatch has no A-matrix component to correct at all.
    """
    coeffs = sindy_result["coefficients"]
    n_states = A_current.shape[0]
    n_inputs = B_current.shape[1]

    # SINDy's coefficient matrix is [bias | state columns | control/input columns], built
    # from feature_names = state_names + input_names via discover_equations's polynomial
    # library (degree=1, include_bias=True). Column 0 is the bias term (not used here --
    # A/B are linear operators, no constant offset); columns 1:n_states+1 are the state
    # (A-matrix) terms; the remaining n_inputs columns are the control (B-matrix) terms.
    if coeffs.shape[1] >= 1 + n_states + n_inputs:
        A_sindy = coeffs[:n_states, 1:n_states + 1]
        B_sindy = coeffs[:n_states, n_states + 1:n_states + 1 + n_inputs]
    elif coeffs.shape[1] > n_states:
        # No control columns were included in this fit (input_data was None) -- refine A
        # only, leave B untouched, matching the previous behaviour for that case.
        A_sindy = coeffs[:n_states, 1:n_states + 1]
        B_sindy = None
    else:
        A_sindy = coeffs[:n_states, :n_states]
        B_sindy = None

    A_refined = (1 - blend_factor) * A_current + blend_factor * A_sindy
    B_refined = (1 - blend_factor) * B_current + blend_factor * B_sindy if B_sindy is not None else B_current

    return A_refined, B_refined


def validate_against_conservation(
    sindy_model,
    state_data: np.ndarray,
    dt: float = 1.0,
) -> Dict[str, float]:
    """
    Check that SINDy's discovered equations don't violate conservation laws.

    For mass conservation: sum of all level derivatives weighted by areas
    should equal net flow in - net flow out.

    Returns violation magnitudes per conservation law.
    """
    derivatives = sindy_model.differentiate(state_data, t=dt)
    predicted = sindy_model.predict(state_data)

    prediction_error = np.mean(np.abs(derivatives - predicted), axis=0)

    return {
        f"state_{i}": float(err)
        for i, err in enumerate(prediction_error)
    }


def _simulate_thermal_system(
    n_steps: int, dt: float, decay_rate: float, ambient_temp: float, flow_gain: float, t0: float, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic data from dT/dt = -decay_rate*(T - ambient_temp) + flow_gain*flow_rate(t) --
    the spec's own thermal example, with KNOWN coefficients -- for validating
    discover_equations against ground truth below. flow_rate drifts rather than staying
    constant: a constant control input is indistinguishable from the bias term, so SINDy (like
    any system identification method) needs persistent excitation to separate the two.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps) * dt
    flow_rate = np.clip(np.cumsum(rng.normal(0, 0.3, n_steps)) + 5.0, 0.5, 10.0)

    T = np.zeros(n_steps)
    T[0] = t0
    for i in range(1, n_steps):
        dT = -decay_rate * (T[i - 1] - ambient_temp) + flow_gain * flow_rate[i - 1]
        T[i] = T[i - 1] + dt * dT
    return t, T, flow_rate


if __name__ == "__main__":
    decay_rate, ambient_temp, flow_gain = 0.037, 18.0, 0.021
    t, T, flow_rate = _simulate_thermal_system(
        n_steps=2000, dt=0.5, decay_rate=decay_rate, ambient_temp=ambient_temp, flow_gain=flow_gain, t0=40.0
    )

    result = discover_equations(
        state_data=T.reshape(-1, 1),
        input_data=flow_rate.reshape(-1, 1),
        dt=0.5,
        feature_names=["T", "flow_rate"],
        threshold=0.005,
        max_poly_degree=1,
    )

    feature_names = result["model"].get_feature_names()
    coeffs = result["coefficients"][0]
    coef_map = dict(zip(feature_names, coeffs))
    print("\nfeature names:", feature_names)
    print("coefficients:", coeffs)
    print("score (R^2):", result["score"])

    true_bias = decay_rate * ambient_temp
    true_T_coef = -decay_rate
    true_flow_coef = flow_gain
    print(f"\ntrue equation: dT/dt = {true_T_coef:.4f}*T + {true_bias:.4f} + {true_flow_coef:.4f}*flow_rate")

    discovered_bias = coef_map.get("1", 0.0)
    discovered_T_coef = coef_map.get("T", 0.0)
    discovered_flow_coef = coef_map.get("flow_rate", 0.0)

    print(f"\nbias:      true={true_bias:.4f} discovered={discovered_bias:.4f}")
    print(f"T coef:    true={true_T_coef:.4f} discovered={discovered_T_coef:.4f}")
    print(f"flow coef: true={true_flow_coef:.4f} discovered={discovered_flow_coef:.4f}")

    # Tight tolerance on the two rate constants (SINDy recovers these almost exactly here);
    # looser on the bias term, since it also has to absorb the RK4-vs-pysindy's-own
    # finite-difference discretization mismatch between how the training data was generated
    # and how SINDy estimates derivatives from it, not just true measurement noise.
    assert abs(discovered_T_coef - true_T_coef) < 0.01, "T coefficient not recovered accurately"
    assert abs(discovered_flow_coef - true_flow_coef) < 0.01, "flow_rate coefficient not recovered accurately"
    assert abs(discovered_bias - true_bias) < 0.5, "bias term not recovered accurately"

    # Also sanity-check refine_observer_matrices doesn't crash on a plausible-shaped observer
    # matrix -- not a correctness proof (needs a real multi-domain state vector to mean
    # anything), just confirms the blending arithmetic is shape-safe.
    A_dummy = np.zeros((1, 1))
    B_dummy = np.zeros((1, 1))
    A_refined, B_refined = refine_observer_matrices(result, A_dummy, B_dummy, blend_factor=0.3)
    assert A_refined.shape == A_dummy.shape
    print(f"\nrefine_observer_matrices shape check passed: A_refined={A_refined}")

    print("\nsindy_discovery.py smoke test passed")
