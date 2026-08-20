# PhysAttest — Component 1: Multi-Domain Observer
## Mathematical Formulation for SWaT

## 1. Why three separate residuals, not one

The reason a single ML-on-raw-sensors model collapses under adaptive attack is
that it learns *one* joint distribution, and a strong enough attacker can find
a point that satisfies that single learned manifold. Physics, chemistry, and
math are three **independently and exactly** true constraints — an attacker
has to fool all three simultaneously, and (Theorem 2) the cost of doing so is
additive across domains: `K_physics + K_chemistry + K_math`. So the split
isn't just organizational, it's the thing that makes the impossibility bound
a *sum* rather than a single term.

## 2. State-space model (physics domain)

**State vector.** For each instrumented tank *i* in the plant graph, the
state is its water level `L_i` (meters). For a 6-stage SWaT model this is a
vector of ~6-10 levels depending on how finely you decompose each stage.

**Continuity equation (Layer 1 conservation of mass).** At constant density,
mass conservation reduces to volume conservation:

```
dL_i/dt = (1/A_i) * ( Σ_{pipes into i} Q_p  −  Σ_{pipes out of i} Q_p )
```

where `A_i` is the tank's cross-sectional area (m²) and `Q_p` is the
volumetric flow through pipe *p* (m³/s).

**Actuator model.** Pumps and motorized valves are modeled as switches on a
nominal flow: `Q_p = q_p^nom * u_p`, where `u_p ∈ {0,1}` (or a duty cycle in
`[0,1]` if you have PWM control) is the commanded actuator state — this is
known exactly, since it's the command *you* issued, not a measurement.

Putting it together, the physics state-space model is **linear**:

```
ẋ = A x + B u + w         (A = 0 for static tanks; all dynamics via B)
y = C x + v
```

with `w ~ N(0, Q)` process noise (leaks, evaporation, unmodeled turbulence)
and `v ~ N(0, R)` sensor noise (LIT sensor accuracy, ~±0.5% FS on SWaT).

**Residual.** Run a Kalman filter purely as a residual generator:

```
x̂(k|k-1) = x̂(k-1) + Δt (A x̂(k-1) + B u(k-1))
r_physics(k) = y(k) − C x̂(k|k-1)
```

Under honest operation, `r_physics ≈ 0` (bounded by sensor noise). Under
spoofing of sensor *j*, `y_j` no longer reflects the true state that the
*other* honest sensors and known actuator commands jointly imply — the
residual on that channel grows, which is Theorem 1's mechanism.

## 3. Chemistry domain

For each dosed/decaying chemical species (free chlorine, pH-active reagent),
mass balance in a well-mixed tank of volume `V` gives:

```
dC/dt = −k_decay * C + dosing_rate(t) / V
```

`k_decay` is a plant/chemistry constant (chlorine decay ~10⁻⁴–10⁻³ s⁻¹
depending on temperature/organic load — calibrate from clean data).
`dosing_rate` is **known** because it's the PLC's commanded pump setpoint,
exactly like `u` on the physics side. The chemistry residual is the same
innovation structure as physics: predict forward one step from the last
measurement + known dosing, compare to the next measurement.

```
r_chemistry(k) = C_measured(k) − Ĉ(k|k-1)
```

## 4. Math domain

This is the odd one out deliberately: it captures constraints that hold by
**algebra/logic**, independent of any dynamical model, so it's robust to
errors in your physics/chemistry parameter estimates. On SWaT the natural
math-domain constraint is actuator/flow-sensor consistency (a Kirchhoff-style
law):

```
r_math(k) = Q_measured(k) − q_p^nom * u_p(k)
```

For every instrumented pipe, the flow sensor reading must match "nominal
flow × commanded actuator state" up to noise. This catches attacks that spoof
a flow sensor while leaving the level sensors alone (which would otherwise
sail through the physics-only residual, since flow doesn't appear in the
level state vector directly except through B).

You can strengthen this domain further with global mass balance over the
whole plant (sum of all measured inflows = sum of all measured outflows +
rate of change of total stored volume) — this is a second, independent
algebraic check and a good second math-domain residual to add for the paper.

## 5. Layer 2 — SINDy

Layer 1 above is exact by construction but idealized (constant `A_i`, linear
actuator model, single decay constant). Layer 2 fits the **residual left
over from Layer 1** using SINDy's sparse regression over a polynomial library,
producing a short, human-readable correction ODE, e.g.:

```
d(r_LIT301)/dt = 0.52 * (L_LIT101 − L_LIT301) + 0.02 * FIT201
```

This equation itself becomes part of the observer (folded back as an
additional term in `A`/`B`), so what's left after Layer 1 + Layer 2 is
genuinely small and closer to pure sensor noise — which is what Layer 3
(PINN) then has to explain, and why it can afford to be small.

## 6. Validation criterion

On clean (attack-free) data, for each domain:

```
mean|r_domain(k)| / σ_noise,domain  ≈  O(1)
```

If this ratio is ≫ 1, the model has structural bias (wrong `A_i`, wrong
`k_decay`, mis-modeled actuator) — not noise, and Layer 1 needs correcting
before Layer 2 is even meaningful (SINDy will just try to fit your Layer-1
bias, which contaminates the "plant-specific detail" you actually want it
to discover).

## 7. Connection to Theorem 2

The three residual covariances `K_physics`, `K_chemistry`, `K_math` (the
sensitivity of each residual to a unit perturbation, i.e. `C` in the
observability sense for each domain) sum inside the log-det term precisely
because — under H0 — the three residual channels are (approximately)
independent given the plant state: they're derived from disjoint physical
mechanisms (mass conservation vs. chemical kinetics vs. actuator logic), so
their Fisher information about any hidden attack signal adds.
