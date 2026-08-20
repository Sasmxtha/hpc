"""
PhysAttest — Layer 2: SINDy-discovered plant-specific equations.

Layer 1 (state_space_model.py) captures conservation laws that are exact by
construction. It will NOT capture plant idiosyncrasies -- e.g. a UF membrane
that fouls over time and changes the true flow/pressure relationship, or a
tank with a slow undocumented leak. SINDy discovers a compact ODE for the
RESIDUAL of Layer 1, in human-readable symbolic form, on clean/attack-free
data.

Usage pattern for the paper:
  1. Run PhysicsBlock over clean SWaT data -> get r_phys(t) time series.
  2. Feed r_phys(t) (and any candidate driving variables, e.g. neighbouring
     tank levels, flow) into SINDy.
  3. SINDy returns a short symbolic ODE, e.g.:
        d(r_LIT301)/dt = 0.52*(L_LIT101 - L_LIT301) + 0.02*FIT201
     which becomes the new Layer-2 correction term folded back into the
     observer, so the *next* residual is even closer to pure noise.
"""

import numpy as np
import pysindy as ps


def discover_plant_equations(
    X: np.ndarray,
    t: np.ndarray,
    feature_names: list[str],
    poly_degree: int = 2,
    threshold: float = 0.05,
) -> ps.SINDy:
    """
    X: (T, n_features) array of measured/derived state variables, e.g.
       columns = [L_LIT101, L_LIT301, FIT201, r_phys_LIT301, ...]
       -- put the quantity you want an equation FOR (e.g. residual or a
       state derivative target) as one of the columns; SINDy fits dX/dt for
       every column against a library built from all columns.
    t: (T,) time vector (SWaT data is 1 Hz, so t = np.arange(T)).
    feature_names: human-readable names, printed in the discovered equations.

    Returns a fitted pysindy.SINDy model. Call `.print()` to get the
    human-readable equations for the paper/appendix, and `.simulate()` to
    forward-simulate for validation.
    """
    library = ps.PolynomialLibrary(degree=poly_degree, include_bias=False)
    optimizer = ps.STLSQ(threshold=threshold)   # sparse regression -> few terms
    model = ps.SINDy(feature_library=library, optimizer=optimizer)
    model.fit(X, t=t, feature_names=feature_names)
    return model


def example_on_synthetic_two_tank():
    """
    Minimal runnable example: two coupled tanks with a *known* ground-truth
    coupling law, recovered purely from data. This is the sanity check to
    run before pointing SINDy at real SWaT residuals.
    """
    rng = np.random.default_rng(0)
    dt = 1.0
    T = 4000
    t = np.arange(T) * dt

    L1 = np.zeros(T)
    L2 = np.zeros(T)
    L1[0], L2[0] = 0.8, 0.3
    k_couple = 0.05   # ground truth coupling constant to recover
    inflow = 0.01

    for i in range(1, T):
        dL1 = -k_couple * (L1[i - 1] - L2[i - 1]) + inflow
        dL2 = k_couple * (L1[i - 1] - L2[i - 1]) - 0.008 * L2[i - 1]
        L1[i] = L1[i - 1] + dt * dL1 + rng.normal(0, 1e-4)
        L2[i] = L2[i - 1] + dt * dL2 + rng.normal(0, 1e-4)

    X = np.column_stack([L1, L2])
    model = discover_plant_equations(
        X, t, feature_names=["L1", "L2"], poly_degree=1, threshold=0.005
    )
    return model


if __name__ == "__main__":
    m = example_on_synthetic_two_tank()
    print("Discovered SINDy equations (should recover coupling ~0.05, decay ~0.008):")
    m.print()
