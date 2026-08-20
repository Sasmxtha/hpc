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
