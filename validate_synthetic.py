"""
Validation harness for Component 1 on synthetic "clean" data that mimics the
SWaT 4-tank cascade (P1->P3->P4->P6). This is a stand-in for real SWaT CSVs
(which require an access request to iTrust) -- swap `simulate_clean_plant`
for a SWaT CSV loader and everything downstream is unchanged.

Run:  python3 validate_synthetic.py
Expect: mean|r_physics|, mean|r_math|, mean|r_chemistry| all << sensor noise
std (near-zero, consistent with r ~ N(0, R) under H0 in Theorem 1/2).
"""

import numpy as np
from state_space_model import (
    TANKS, PIPES, CHEM_SPECIES,
    PhysicsBlock, ChemistryBlock, KalmanResidualGenerator, MultiDomainObserver,
)


def simulate_clean_plant(T=3600, dt=1.0, seed=0):
    """Ground-truth forward simulation + realistic sensor noise, no attacks."""
    rng = np.random.default_rng(seed)
    pb = PhysicsBlock(TANKS, PIPES)
    n, m = pb.n, pb.m

    x_true = np.zeros((T, n))
    x_true[0] = np.array([0.6, 0.6, 0.6, 0.4])   # initial levels (m)

    # simple bang-bang level control: pump ON if level below setpoint
    setpoints = np.array([0.8, 0.8, 0.8, 0.6])
    u_true = np.zeros((T, m))

    for k in range(1, T):
        u = np.zeros(m)
        for j, p in enumerate(PIPES):
            if p.src == "ENV" and p.dst in pb.tank_idx:
                i = pb.tank_idx[p.dst]
                u[j] = 1.0 if x_true[k - 1, i] < setpoints[i] else 0.0
            elif p.src in pb.tank_idx and p.dst in pb.tank_idx:
                # gravity-fed transfer pumps: ON if upstream has water and
                # downstream below its setpoint
                i_src, i_dst = pb.tank_idx[p.src], pb.tank_idx[p.dst]
                u[j] = 1.0 if (x_true[k - 1, i_src] > 0.15 and
                               x_true[k - 1, i_dst] < setpoints[i_dst]) else 0.0
            elif p.dst == "ENV" and p.src in pb.tank_idx:
                i = pb.tank_idx[p.src]
                u[j] = 1.0 if x_true[k - 1, i] > setpoints[i] * 0.9 else 0.0
        u_true[k] = u
        x_true[k] = np.clip(pb.step(x_true[k - 1], u, dt), 0.0, 3.0)

    # chemistry ground truth
    cb = ChemistryBlock(CHEM_SPECIES, tank_volume_m3={t.name: t.area_m2 * t.max_level_m for t in TANKS})
    c_true = np.zeros((T, len(CHEM_SPECIES)))
    c_true[0] = [2.0]  # mg/L free chlorine
    dosing = np.full((T, len(CHEM_SPECIES)), 0.5)  # constant dosing rate
    for k in range(1, T):
        c_true[k] = cb.step(c_true[k - 1], dosing[k], dt)

    # sensor noise
    level_noise_std = 0.003     # 3 mm, realistic for SWaT LIT sensors
    flow_noise_std = 3e-5       # m3/s
    chem_noise_std = 0.02       # mg/L

    y_levels = x_true + rng.normal(0, level_noise_std, x_true.shape)
    q_ideal = u_true * np.array([p.nominal_flow_m3s for p in PIPES])
    y_flows_arr = q_ideal + rng.normal(0, flow_noise_std, q_ideal.shape)
    y_chem = c_true + rng.normal(0, chem_noise_std, c_true.shape)

    return pb, cb, x_true, u_true, y_levels, y_flows_arr, y_chem, level_noise_std, flow_noise_std, chem_noise_std


def main():
    (pb, cb, x_true, u_true, y_levels, y_flows_arr, y_chem,
     lvl_std, flow_std, chem_std) = simulate_clean_plant()

    T, n = x_true.shape
    m = len(PIPES)

    kf = KalmanResidualGenerator(
        A=pb.A, B=pb.B, C=np.eye(n),
        Q=np.eye(n) * 1e-6,
        R=np.eye(n) * lvl_std**2,
        x0=x_true[0].copy(),
        P0=np.eye(n) * 1e-4,
    )
    observer = MultiDomainObserver(pb, kf, chem_block=cb, c0=y_chem[0])

    r_phys_hist, r_math_hist, r_chem_hist = [], [], []
    for k in range(1, T):
        y_flows = {p.flow_sensor: y_flows_arr[k, j]
                   for j, p in enumerate(PIPES) if p.flow_sensor is not None}
        y_chem_dict = {s.sensor: y_chem[k, i] for i, s in enumerate(CHEM_SPECIES)}
        dosing_cmd = np.full(len(CHEM_SPECIES), 0.5)   # known PLC dosing setpoint
        res = observer.step(u_true[k], y_levels[k], y_flows, y_chem_dict,
                             dosing_cmd=dosing_cmd, dt=1.0)
        r_phys_hist.append(res["r_physics"])
        r_math_hist.append(res["r_math"])
        r_chem_hist.append(res["r_chemistry"])

    r_phys_hist = np.array(r_phys_hist)
    r_math_hist = np.array(r_math_hist)
    r_chem_hist = np.array(r_chem_hist)

    print("=== Clean-data residual validation (H0: no attack) ===")
    print(f"physics   residual: mean|r| = {np.mean(np.abs(r_phys_hist)):.6f} m   "
          f"(sensor noise std = {lvl_std:.6f} m)")
    print(f"math      residual: mean|r| = {np.mean(np.abs(r_math_hist)):.8f} m3/s "
          f"(sensor noise std = {flow_std:.8f} m3/s)")
    print(f"chemistry residual: mean|r| = {np.mean(np.abs(r_chem_hist)):.6f} mg/L "
          f"(sensor noise std = {chem_std:.6f} mg/L)")
    print()
    print("PASS criterion: mean|r| should sit within ~1x the sensor noise std")
    print("(i.e. residual is explained by measurement noise alone, not model bias).")

    for label, r, std in [("physics", r_phys_hist, lvl_std),
                           ("math", r_math_hist, flow_std),
                           ("chemistry", r_chem_hist, chem_std)]:
        ratio = np.mean(np.abs(r)) / std
        status = "PASS" if ratio < 1.5 else "CHECK MODEL"
        print(f"  {label:9s}: mean|r| / noise_std = {ratio:.2f}  -> {status}")


if __name__ == "__main__":
    main()
