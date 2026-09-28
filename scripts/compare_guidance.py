#!/usr/bin/env python
"""Compare powered-descent guidance laws on the same landing scenario.

Runs each requested guidance law with the same controller, navigation mode,
seed and vehicle, then writes a metrics table
(``results/compare_guidance_<c>.json``) and an overlay figure
(``docs/figures/compare_guidance_<c>.png``).

Example
-------
    .venv/bin/python scripts/compare_guidance.py --controller pid
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


def make_figure(results: dict, controller: str, path: str) -> str:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle(f"guidance comparison -- controller: {controller}")

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
        ax.plot(res.t, res.u[:, 0], lw=0.8, label=name)
    ax.set_xlabel("t [s]"); ax.set_ylabel("throttle"); ax.legend()
    ax.set_title("throttle")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--controller", choices=list(CONTROLLERS), default="pid")
    p.add_argument("--guidance", nargs="+", choices=DESCENT_GUIDANCE,
                   default=DESCENT_GUIDANCE)
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
    for gname in args.guidance:
        sim, veh, x0 = build_sim(gname, args.controller, args.nav,
                                 args.seed, args.dt, args.control_hz)
        t0 = time.perf_counter()
        res = sim.run(x0, t_end=args.t_end, stop_on_touchdown=True)
        wall = time.perf_counter() - t0
        ok, mode = landing_success(res.touchdown, veh.prop_remaining)
        results[gname] = res
        table[gname] = res.summary(
            landing_success_fn=landing_success,
            prop_remaining=veh.prop_remaining)
        table[gname]["wall_time_s"] = wall
        td = res.touchdown or {}
        print(f"{gname:10s} wall={wall:6.1f}s  "
              f"vz={td.get('vertical_speed', float('nan')):6.2f}  "
              f"vlat={td.get('lateral_speed', float('nan')):6.2f}  "
              f"tilt={td.get('tilt_angle_deg', float('nan')):6.2f}  "
              f"off={td.get('lateral_offset', float('nan')):7.2f}  "
              f"fuel={res.prop_used:8.0f} kg  "
              f"-> {'OK' if ok else 'FAIL:' + mode}", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    json_path = os.path.join(args.outdir,
                             f"compare_guidance_{args.controller}.json")
    payload = {"controller": args.controller, "nav": args.nav,
               "seed": args.seed, "guidance": table}
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    fig_path = None
    if not args.no_figs:
        fig_path = make_figure(
            results, args.controller,
            os.path.join(args.figdir,
                         f"compare_guidance_{args.controller}.png"))

    n_ok = sum(1 for v in table.values() if v.get("success"))
    print(f"\n{n_ok}/{len(table)} guidance laws landed successfully")
    print(f"json: {json_path}")
    if fig_path:
        print(f"figure: {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
