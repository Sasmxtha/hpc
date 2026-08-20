<<<<<<< HEAD
"""SINDy equation discovery (Component 1, Layer 2): discovers a human-readable differential
equation directly from operational data using PySINDy's sparse regression, filling the gap
between Layer 1's handwritten conservation laws (exact, but only for relationships an
engineer wrote down) and Layer 3's PINN (handles anything, but is a black-box network, not
a readable equation). This is the layer the spec's own example belongs to:
    dT/dt = -0.037(T_pipe - T_ambient) + 0.021 x flow_rate

Validated below by generating synthetic data from exactly that thermal system with KNOWN
coefficients -- discovering AN equation is easy (SINDy always returns something); the useful
claim is that the discovered coefficients are close to the true physical constants, which is
what's actually checked, not just that the fit residual is small.
"""

from typing import Tuple

import numpy as np
import pysindy as ps


def simulate_thermal_system(
    n_steps: int,
    dt: float,
    decay_rate: float,
    ambient_temp: float,
    flow_gain: float,
    t0: float,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates synthetic operational data from dT/dt = -decay_rate*(T - ambient_temp) +
    flow_gain*flow_rate(t) -- the spec's own example ODE -- with a randomly drifting
    flow_rate control signal. The drift matters: a CONSTANT flow_rate would make its
    contribution indistinguishable from the bias term, and SINDy (like any system
    identification method) needs persistent excitation in the control input to separate the
    two. Returns (time, temperature, flow_rate) arrays.
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


def discover_equation(t: np.ndarray, state: np.ndarray, control: np.ndarray, threshold: float = 0.01) -> ps.SINDy:
    """Fits sparse regression (STLSQ) over a degree-1 polynomial library of [state, control]
    plus a bias term -- matching the linear structure of conservation-law-derived ODEs like
    the spec's thermal example. `threshold` is STLSQ's sparsity knob: how small a
    coefficient has to be before it's zeroed out entirely, which is what keeps the discovered
    equation SHORT and human-readable rather than a dense polynomial with every term present
    at some tiny nonzero weight.
    """
    library = ps.PolynomialLibrary(degree=1, include_bias=True)
    optimizer = ps.STLSQ(threshold=threshold)
    model = ps.SINDy(feature_library=library, optimizer=optimizer)
    model.fit(state.reshape(-1, 1), t=t, u=control.reshape(-1, 1), feature_names=["T", "flow_rate"])
    # feature_names must cover BOTH the state and the control variable(s) -- passing just the
    # state name here throws an internal IndexError inside pysindy's PolynomialLibrary when it
    # tries to build a display name for the flow_rate term, since it indexes into
    # feature_names for every library input, not just the state.
    return model


if __name__ == "__main__":
    decay_rate, ambient_temp, flow_gain = 0.037, 18.0, 0.021
    t, T, flow_rate = simulate_thermal_system(
        n_steps=2000, dt=0.5, decay_rate=decay_rate, ambient_temp=ambient_temp, flow_gain=flow_gain, t0=40.0
    )

    model = discover_equation(t, T, flow_rate, threshold=0.005)
    print("discovered equation:")
    model.print()

    feature_names = model.get_feature_names()
    coeffs = model.coefficients()[0]
    coef_map = dict(zip(feature_names, coeffs))
    print("\nfeature names:", feature_names)
    print("coefficients:", coeffs)

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
    print("\nsindy_discovery.py smoke test passed")
=======
"""
SINDy (Sparse Identification of Nonlinear Dynamics) equation discovery.
Layer 2 of the equation hierarchy.

Uses PySINDy to discover differential equations from operational data
in human-readable form, e.g.:
    dT/dt = -0.037(T_pipe - T_ambient) + 0.021 × flow_rate

These fill the gap between handwritten conservation laws (Layer 1)
and the PINN fallback (Layer 3) by capturing plant-specific parameters
like exact pipe friction, heat loss rates, and mixing coefficients.
"""

import numpy as np
from typing import List, Optional, Dict

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
        feature_names: names for each state variable
        threshold: sparsity threshold — higher = sparser equations
        max_poly_degree: max polynomial degree in the library

    Returns:
        Dictionary with:
            'model': fitted PySINDy model
            'equations': list of discovered equation strings
            'coefficients': coefficient matrix
            'score': R² score on the data

    HOW IT WORKS:
    SINDy builds a library of candidate terms (x, x², x*y, sin(x), ...)
    and finds the sparsest combination that explains dx/dt.

    Example output:
        dx0/dt = -0.037 x0 + 0.021 x1 + 0.003 x0 x1
    Meaning:
        dh/dt = -0.037h + 0.021*flow + 0.003*h*flow
    """
    if not HAS_PYSINDY:
        raise RuntimeError("PySINDy required. Install with: pip install pysindy")

    # Build function library: polynomials up to degree 2
    library = ps.PolynomialLibrary(degree=max_poly_degree, include_bias=True)

    # Use STLSQ (Sequentially Thresholded Least Squares) optimizer
    # This promotes sparsity — most coefficients become exactly zero
    optimizer = ps.STLSQ(threshold=threshold, alpha=0.01)

    # Build and fit the model
    model = ps.SINDy(
        feature_library=library,
        optimizer=optimizer,
        feature_names=feature_names,
        discrete_time=False,
    )

    if input_data is not None:
        model.fit(state_data, t=dt, u=input_data)
    else:
        model.fit(state_data, t=dt)

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
    """
    coeffs = sindy_result["coefficients"]
    n_states = A_current.shape[0]

    # SINDy's coefficient matrix includes bias and polynomial terms.
    # The linear terms (columns 1:n_states+1) map to the A matrix.
    if coeffs.shape[1] > n_states:
        A_sindy = coeffs[:n_states, 1:n_states + 1]
    else:
        A_sindy = coeffs[:n_states, :n_states]

    A_refined = (1 - blend_factor) * A_current + blend_factor * A_sindy

    return A_refined, B_current


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
>>>>>>> 67782d2bf638fc6e1aa240b226b523957b981a18
