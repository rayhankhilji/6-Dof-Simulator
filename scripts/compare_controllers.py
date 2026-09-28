#!/usr/bin/env python
"""Compare TVC controllers on the same powered-descent landing scenario.

Runs each requested controller with the same guidance law, navigation mode,
seed and vehicle, then writes a metrics table
(``results/compare_controllers_<g>.json``) and an overlay figure
(``docs/figures/compare_controllers_<g>.png``).

Example
-------
    .venv/bin/python scripts/compare_controllers.py --guidance zemzev
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

from sixdof.control import CONTROLLERS
from sixdof.scenarios import landing_success

from run_landing import DESCENT_GUIDANCE, build_sim, tilt_history

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_figure(results: dict, guidance: str, path: str) -> str:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle(f"controller comparison -- guidance: {guidance}")

    ax = axes[0, 0]
    ax.add_patch(Circle((0, 0), 10.0, fill=False, ls="--", color="k"))
    for name, res in results.items():
        ax.plot(res.x[:, 0], res.x[:, 1], lw=1.0, label=name)
        ax.plot(res.x[-1, 0], res.x[-1, 1], "x", ms=8)
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
    ax.set_title("ground track"); ax.legend(); ax.axis("equal")

    ax = axes[0, 1]
    for name, res in results.items():
        ax.plot(res.t, res.x[:, 2], label=name)
    ax.set_xlabel("t [s]"); ax.set_ylabel("altitude [m]"); ax.legend()
    ax.set_title("altitude")

    ax = axes[0, 2]
    for name, res in results.items():
        ax.plot(res.t, np.hypot(res.x[:, 0], res.x[:, 1]), label=name)
    ax.set_xlabel("t [s]"); ax.set_ylabel("offset [m]"); ax.legend()
    ax.set_title("lateral offset")

    ax = axes[1, 0]
    for name, res in results.items():
        ax.plot(res.t, -res.x[:, 5], label=name)
    ax.set_xlabel("t [s]"); ax.set_ylabel("|v_z| [m/s]"); ax.legend()
    ax.set_title("descent rate")

    ax = axes[1, 1]
    for name, res in results.items():
        ax.plot(res.t, tilt_history(res), label=name)
    ax.axhline(5.0, color="r", ls="--", lw=0.8)
    ax.set_xlabel("t [s]"); ax.set_ylabel("tilt [deg]"); ax.legend()
    ax.set_title("tilt")

    ax = axes[1, 2]
    for name, res in results.items():
        if res.compute_time is not None:
            ax.plot(res.t, res.compute_time * 1e3, lw=0.7, label=name)
    ax.set_xlabel("t [s]"); ax.set_ylabel("ms"); ax.legend()
    ax.set_title("controller compute time")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--guidance", choices=DESCENT_GUIDANCE, default="zemzev")
    p.add_argument("--controllers", nargs="+", choices=list(CONTROLLERS),
                   default=list(CONTROLLERS))
    p.add_argument("--nav", choices=["perfect", "ekf"], default="perfect")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--control-hz", type=float, default=50.0)
    p.add_argument("--t-end", type=float, default=90.0)
    p.add_argument("--outdir", type=str, default=os.path.join(ROOT, "results"))
    p.add_argument("--figdir", type=str,
                   default=os.path.join(ROOT, "docs", "figures"))
    p.add_argument("--no-figs", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    results, table = {}, {}
    for cname in args.controllers:
        sim, veh, x0 = build_sim(args.guidance, cname, args.nav,
                                 args.seed, args.dt, args.control_hz)
        t0 = time.perf_counter()
        res = sim.run(x0, t_end=args.t_end, stop_on_touchdown=True)
        wall = time.perf_counter() - t0
        ok, mode = landing_success(res.touchdown, veh.prop_remaining)
        results[cname] = res
        table[cname] = res.summary(
            landing_success_fn=landing_success,
            prop_remaining=veh.prop_remaining)
        table[cname]["wall_time_s"] = wall
        td = res.touchdown or {}
        print(f"{cname:10s} wall={wall:6.1f}s  "
              f"vz={td.get('vertical_speed', float('nan')):6.2f}  "
              f"vlat={td.get('lateral_speed', float('nan')):6.2f}  "
              f"tilt={td.get('tilt_angle_deg', float('nan')):6.2f}  "
              f"off={td.get('lateral_offset', float('nan')):7.2f}  "
              f"-> {'OK' if ok else 'FAIL:' + mode}", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    json_path = os.path.join(args.outdir,
                             f"compare_controllers_{args.guidance}.json")
    payload = {"guidance": args.guidance, "nav": args.nav,
               "seed": args.seed, "controllers": table}
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    fig_path = None
    if not args.no_figs:
        fig_path = make_figure(
            results, args.guidance,
            os.path.join(args.figdir,
                         f"compare_controllers_{args.guidance}.png"))

    n_ok = sum(1 for v in table.values() if v.get("success"))
    print(f"\n{n_ok}/{len(table)} controllers landed successfully")
    print(f"json: {json_path}")
    if fig_path:
        print(f"figure: {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
