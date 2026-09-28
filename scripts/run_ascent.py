#!/usr/bin/env python
"""Run the two-stage gravity-turn ascent closed-loop simulation.

Example
-------
    .venv/bin/python scripts/run_ascent.py --controller pid --t-end 520

Writes ``results/ascent_<controller>.npz``, ``results/ascent_<controller>.json``
and the figure ``docs/figures/ascent_<controller>.png``.
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

from sixdof.actuators import ActuatorSuite, GimbalActuator, RCSActuator, ThrottleActuator
from sixdof.control import CONTROLLERS, make_controller
from sixdof.guidance import GravityTurnGuidance
from sixdof.scenarios import ascent_scenario
from sixdof.simulation import PerfectNavigator, Simulation
from sixdof.math.quaternion import quat_rotate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_sim(controller_name: str, dt: float, control_rate_hz: float,
              h_target: float, seed: int = 0):
    veh, atm, wind, x0, meta = ascent_scenario()
    eng = veh.active_stage_config.engine
    act = ActuatorSuite(
        gimbal=GimbalActuator(eng.gimbal_max, eng.gimbal_rate_max),
        throttle=ThrottleActuator(eng.throttle_tau, eng.thrust_min_frac),
        rcs=RCSActuator(),
    )
    sim = Simulation(
        veh, atm, wind, sensors=None, navigator=PerfectNavigator(),
        guidance=GravityTurnGuidance(h_target=h_target),
        controller=make_controller(controller_name),
        actuators=act, dt=dt, control_rate_hz=control_rate_hz,
        rng=np.random.default_rng(seed),
        staging_callback=lambda v, t: v.stage(t),
    )
    return sim, veh, x0


def make_figure(res, title: str, path: str) -> str:
    t, x, u, aux = res.t, res.x, res.u, res.aux
    alt = x[:, 2] / 1e3
    speed = np.linalg.norm(x[:, 3:6], axis=1)
    downrange = np.hypot(x[:, 0], x[:, 1]) / 1e3
    x_axis = np.stack([quat_rotate(q, np.array([1.0, 0.0, 0.0]))
                       for q in x[:, 6:10]])
    pitch = np.degrees(np.arctan2(np.hypot(x_axis[:, 0], x_axis[:, 1]),
                                  x_axis[:, 2]))
    fpa = np.degrees(np.arctan2(x[:, 5], np.hypot(x[:, 3], x[:, 4])))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle(title)

    ax = axes[0, 0]
    ax.plot(downrange, alt)
    ax.set_xlabel("downrange [km]"); ax.set_ylabel("altitude [km]")
    ax.set_title("trajectory")

    ax = axes[0, 1]
    ax.plot(t, alt)
    ax.set_xlabel("t [s]"); ax.set_ylabel("altitude [km]"); ax.set_title("altitude")

    ax = axes[0, 2]
    ax.plot(t, speed / 1e3)
    ax.set_xlabel("t [s]"); ax.set_ylabel("speed [km/s]"); ax.set_title("speed")

    ax = axes[1, 0]
    ax.plot(t, pitch, label="pitch (from vertical)")
    ax.plot(t, fpa, label="flight-path angle")
    ax.set_xlabel("t [s]"); ax.set_ylabel("deg"); ax.legend()
    ax.set_title("attitude / flight path")

    ax = axes[1, 1]
    ax.plot(t, aux[:, 2] / 1e3, label="q_dyn")
    ax.plot(t, aux[:, 0], label="Mach")
    ax.set_xlabel("t [s]"); ax.legend(); ax.set_title("aero (kPa / Mach)")

    ax = axes[1, 2]
    ax.plot(t, u[:, 0], label="throttle")
    ax.plot(t, np.degrees(u[:, 1]), label="gimbal_y [deg]")
    ax.plot(t, np.degrees(u[:, 2]), label="gimbal_z [deg]")
    ax.set_xlabel("t [s]"); ax.legend(); ax.set_title("controls")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--controller", choices=list(CONTROLLERS), default="pid")
    p.add_argument("--h-target", type=float, default=200e3,
                   help="target apogee for the tangent-steering cutoff heuristic [m]")
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--control-hz", type=float, default=50.0)
    p.add_argument("--t-end", type=float, default=520.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default=os.path.join(ROOT, "results"))
    p.add_argument("--figdir", type=str,
                   default=os.path.join(ROOT, "docs", "figures"))
    p.add_argument("--no-figs", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    sim, veh, x0 = build_sim(args.controller, args.dt, args.control_hz,
                             args.h_target, args.seed)
    t0 = time.perf_counter()
    res = sim.run(x0, t_end=args.t_end, stop_on_touchdown=False)
    wall = time.perf_counter() - t0

    t, x, aux = res.t, res.x, res.aux
    v = np.linalg.norm(x[:, 3:6], axis=1)
    alt = x[:, 2]
    i_qmax = int(np.argmax(aux[:, 2]))
    # Vertical-energy apogee estimate at each instant (flat-Earth sim:
    # horizontal velocity does not raise apogee).
    apogee_est = alt + x[:, 5] ** 2 / (2 * 9.80665)
    summary = {
        "controller": args.controller,
        "t_end": float(t[-1]),
        "final_alt_km": float(alt[-1] / 1e3),
        "final_speed_km_s": float(v[-1] / 1e3),
        "final_downrange_km": float(np.hypot(x[-1, 0], x[-1, 1]) / 1e3),
        "apogee_est_final_km": float(apogee_est[-1] / 1e3),
        "h_target_km": args.h_target / 1e3,
        "reached_h_target": bool(apogee_est[-1] >= args.h_target),
        "q_max_kPa": float(aux[i_qmax, 2] / 1e3),
        "t_maxq_s": float(t[i_qmax]),
        "alpha_max_deg": float(np.degrees(np.max(aux[:, 1]))),
        "prop_remaining_kg": float(veh.prop_remaining),
        "active_stage": int(veh.active_stage),
        "fuel_used_kg": float(res.prop_used),
        "wall_time_s": wall,
        "events": [[float(te), str(e)] for te, e in res.events],
        "phases": sorted(set(str(p) for p in res.phase)),
    }

    os.makedirs(args.outdir, exist_ok=True)
    npz_path = os.path.join(args.outdir, f"ascent_{args.controller}.npz")
    res.save_npz(npz_path)
    json_path = os.path.join(args.outdir, f"ascent_{args.controller}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    fig_path = None
    if not args.no_figs:
        fig_path = make_figure(res, f"ascent: gravity_turn / {args.controller}",
                               os.path.join(args.figdir,
                                            f"ascent_{args.controller}.png"))

    print(f"===== ascent: gravity_turn + {args.controller} (wall {wall:.1f}s) =====")
    print(f"final alt {summary['final_alt_km']:.1f} km, "
          f"speed {summary['final_speed_km_s']:.3f} km/s, "
          f"apogee est {summary['apogee_est_final_km']:.0f} km "
          f"(target {summary['h_target_km']:.0f} km)")
    print(f"max-q {summary['q_max_kPa']:.1f} kPa @ t={summary['t_maxq_s']:.0f}s, "
          f"max alpha {summary['alpha_max_deg']:.1f} deg")
    print(f"events: {summary['events']}")
    print(f"npz: {npz_path}\njson: {json_path}")
    if fig_path:
        print(f"figure: {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
