#!/usr/bin/env python
"""Run a single closed-loop powered-descent landing simulation.

Example
-------
    .venv/bin/python scripts/run_landing.py --guidance zemzev --controller pid --nav ekf

Writes ``results/landing_<g>_<c>_<nav>.npz`` (full time history),
``results/landing_<g>_<c>_<nav>.json`` (touchdown metrics) and the figure
``docs/figures/landing_<g>_<c>_<nav>.png``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from sixdof.actuators import ActuatorSuite, GimbalActuator, RCSActuator, ThrottleActuator
from sixdof.control import CONTROLLERS, make_controller
from sixdof.guidance import GUIDANCE, make_guidance
from sixdof.navigation import EKF
from sixdof.scenarios import landing_scenario, landing_success
from sixdof.sensors import Barometer, GPS, IMU, RadarAltimeter, SensorSuite
from sixdof.simulation import PerfectNavigator, Simulation
from sixdof.math.quaternion import quat_rotate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DESCENT_GUIDANCE = [g for g in GUIDANCE if g != "gravity_turn"]


def build_sim(guidance_name: str, controller_name: str, nav_kind: str,
              seed: int, dt: float, control_rate_hz: float):
    """Construct a landing simulation; returns (sim, vehicle, x0)."""
    veh, atm, wind, x0, meta = landing_scenario(np.random.default_rng(seed))
    if nav_kind == "ekf":
        nav = EKF(
            x0[0:3] + np.array([3.0, -2.0, 4.0]),
            x0[3:6] + np.array([0.3, -0.2, 0.2]),
            x0[6:10],
            accel_noise_density=100e-6 * 9.80665,
            gyro_noise_density=np.radians(0.005),
            accel_bias_rw=1e-5, gyro_bias_rw=1e-7,
        )
        sensors = SensorSuite(
            imu=IMU(np.random.default_rng(seed + 1)),
            gps=GPS(np.random.default_rng(seed + 2)),
            barometer=Barometer(np.random.default_rng(seed + 3), atm),
            radar=RadarAltimeter(np.random.default_rng(seed + 4)),
        )
    else:
        nav = PerfectNavigator()
        sensors = None
    eng = veh.active_stage_config.engine
    act = ActuatorSuite(
        gimbal=GimbalActuator(eng.gimbal_max, eng.gimbal_rate_max),
        throttle=ThrottleActuator(eng.throttle_tau, eng.thrust_min_frac,
                                delay_s=0.05),
        rcs=RCSActuator(),
    )
    sim = Simulation(
        veh, atm, wind, sensors=sensors, navigator=nav,
        guidance=make_guidance(guidance_name),
        controller=make_controller(controller_name),
        actuators=act, dt=dt, control_rate_hz=control_rate_hz,
        rng=np.random.default_rng(seed + 100),
    )
    return sim, veh, x0


def tilt_history(res) -> np.ndarray:
    """Body +x-axis angle from vertical [deg] along the trajectory."""
    x_axis = np.stack([quat_rotate(q, np.array([1.0, 0.0, 0.0]))
                       for q in res.x[:, 6:10]])
    return np.degrees(np.arccos(np.clip(x_axis[:, 2], -1.0, 1.0)))


def make_figure(res, title: str, path: str) -> str:
    """Standard landing figure: track, altitude, velocities, tilt, commands."""
    t, x, u = res.t, res.x, res.u
    tilt = tilt_history(res)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle(title)

    ax = axes[0, 0]
    ax.plot(x[:, 0], x[:, 1], lw=1.2)
    ax.add_patch(Circle((0, 0), 10.0, fill=False, ls="--", color="k",
                        label="pad r=10 m"))
    ax.plot(x[-1, 0], x[-1, 1], "rx", ms=10, mew=2, label="touchdown")
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
    ax.set_title("ground track"); ax.legend(); ax.axis("equal")

    ax = axes[0, 1]
    ax.plot(t, x[:, 2])
    ax.set_xlabel("t [s]"); ax.set_ylabel("altitude [m]")
    ax.set_title("altitude")

    ax = axes[0, 2]
    ax.plot(t, -x[:, 5], label="|v_z|")
    ax.plot(t, np.hypot(x[:, 3], x[:, 4]), label="|v_lat|")
    ax.axhline(3.0, color="r", ls="--", lw=0.8)
    ax.axhline(1.5, color="r", ls=":", lw=0.8)
    ax.set_xlabel("t [s]"); ax.set_ylabel("speed [m/s]")
    ax.set_title("velocity components"); ax.legend()

    ax = axes[1, 0]
    ax.plot(t, tilt)
    ax.axhline(5.0, color="r", ls="--", lw=0.8)
    ax.set_xlabel("t [s]"); ax.set_ylabel("tilt [deg]"); ax.set_title("tilt")

    ax = axes[1, 1]
    ax.plot(t, u[:, 0])
    ax.set_xlabel("t [s]"); ax.set_ylabel("throttle"); ax.set_title("throttle")

    ax = axes[1, 2]
    ax.plot(t, np.degrees(u[:, 1]), label="gimbal_y")
    ax.plot(t, np.degrees(u[:, 2]), label="gimbal_z")
    ax.set_xlabel("t [s]"); ax.set_ylabel("gimbal [deg]"); ax.set_title("gimbal")
    ax.legend()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--guidance", choices=DESCENT_GUIDANCE, default="zemzev")
    p.add_argument("--controller", choices=list(CONTROLLERS), default="pid")
    p.add_argument("--nav", choices=["perfect", "ekf"], default="perfect")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--control-hz", type=float, default=50.0)
    p.add_argument("--t-end", type=float, default=90.0)
    p.add_argument("--tag", type=str, default=None,
                   help="artifact filename suffix (default: <g>_<c>_<nav>)")
    p.add_argument("--outdir", type=str, default=os.path.join(ROOT, "results"))
    p.add_argument("--figdir", type=str,
                   default=os.path.join(ROOT, "docs", "figures"))
    p.add_argument("--no-figs", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    tag = args.tag or f"{args.guidance}_{args.controller}_{args.nav}"

    sim, veh, x0 = build_sim(args.guidance, args.controller, args.nav,
                             args.seed, args.dt, args.control_hz)
    t0 = time.perf_counter()
    res = sim.run(x0, t_end=args.t_end, stop_on_touchdown=True)
    wall = time.perf_counter() - t0

    ok, mode = landing_success(res.touchdown, veh.prop_remaining)
    summary = res.summary(landing_success_fn=landing_success,
                          prop_remaining=veh.prop_remaining)
    summary.update({
        "guidance": args.guidance, "controller": args.controller,
        "nav": args.nav, "seed": args.seed, "wall_time_s": wall,
        "events": [[float(t), str(e)] for t, e in res.events],
    })

    os.makedirs(args.outdir, exist_ok=True)
    npz_path = os.path.join(args.outdir, f"landing_{tag}.npz")
    res.save_npz(npz_path)
    json_path = os.path.join(args.outdir, f"landing_{tag}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    fig_path = None
    if not args.no_figs:
        fig_path = make_figure(
            res, f"landing: {args.guidance} / {args.controller} / {args.nav}",
            os.path.join(args.figdir, f"landing_{tag}.png"))

    print(f"===== landing: {args.guidance} + {args.controller} + {args.nav} "
          f"(wall {wall:.1f}s) =====")
    if res.touchdown is not None:
        td = res.touchdown
        print(f"touchdown at t={res.t[-1]:.1f}s")
        print(f"  vertical speed : {td['vertical_speed']:.3f} m/s   (<= 3)")
        print(f"  lateral speed  : {td['lateral_speed']:.3f} m/s   (<= 1.5)")
        print(f"  tilt           : {td['tilt_angle_deg']:.2f} deg    (<= 5)")
        print(f"  lateral offset : {td['lateral_offset']:.2f} m      (<= 10)")
        print(f"  prop remaining : {veh.prop_remaining:.0f} kg")
    else:
        print("no touchdown (t_end reached)")
    print(f"result: {'SUCCESS' if ok else 'FAILED'} ({mode})")
    print(f"npz: {npz_path}\njson: {json_path}")
    if fig_path:
        print(f"figure: {fig_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
