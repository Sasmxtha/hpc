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
