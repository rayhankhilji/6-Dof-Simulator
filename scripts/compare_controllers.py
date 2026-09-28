#!/usr/bin/env python
"""Compare TVC controllers on the powered-descent landing scenario.

Two modes:

* single-seed overlay (default): each requested controller on the same
  nominal scenario -> ``results/compare_controllers_<g>.json`` +
  ``docs/figures/compare_controllers_<g>.png``.
* dispersed sweep (``--sweep``): n dispersed seeds per controller
  (defaults: guidance=optimal, nav=ekf, seeds 100..119, wind U(0,12) m/s +
  gusts, mass/thrust +-3%, density +-10%, pos +-50 m, vel +-5 m/s, 0.1 s
  measurement delay, 10 s mid-descent GPS outage on ~half the seeds),
  parallelized over a process pool ->
  ``results/controller_comparison.json``,
  ``docs/figures/controller_comparison.png`` and
  ``docs/figures/controller_tracking.png``.

Example
-------
    .venv/bin/python scripts/compare_controllers.py --guidance zemzev
    .venv/bin/python scripts/compare_controllers.py --sweep --workers 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from sixdof.control import CONTROLLERS
from sixdof.scenarios import landing_success

from run_landing import (DESCENT_GUIDANCE, build_sim, dispersion_draws,
                         make_comparison_figure, run_trial, summarize_runs,
                         tilt_history)

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


def make_tracking_figure(traces: dict, title: str, path: str) -> str:
    """Nominal-seed tilt + lateral-offset vs time, all controllers overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    fig.suptitle(title)
    for name, tr in traces.items():
        axes[0].plot(tr["t"], tr["tilt_deg"], lw=1.0, label=name)
        axes[1].plot(tr["t"], tr["lateral_offset"], lw=1.0, label=name)
    axes[0].axhline(5.0, color="r", ls="--", lw=0.8, label="5 deg limit")
    axes[0].set_xlabel("t [s]"); axes[0].set_ylabel("tilt [deg]")
    axes[0].set_title("tilt angle"); axes[0].legend()
    axes[1].axhline(10.0, color="r", ls="--", lw=0.8, label="pad r=10 m")
    axes[1].set_xlabel("t [s]"); axes[1].set_ylabel("lateral offset [m]")
    axes[1].set_yscale("log")
    axes[1].set_title("lateral offset"); axes[1].legend()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--guidance", choices=DESCENT_GUIDANCE, default=None,
                   help="default: zemzev (single-seed) / optimal (--sweep)")
    p.add_argument("--controllers", nargs="+", choices=list(CONTROLLERS),
                   default=list(CONTROLLERS))
    p.add_argument("--nav", choices=["perfect", "ekf", "ukf"], default=None,
                   help="default: perfect (single-seed) / ekf (--sweep)")
    p.add_argument("--seed", type=int, default=0,
                   help="single-seed / nominal-tracking seed")
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--control-hz", type=float, default=50.0)
    p.add_argument("--t-end", type=float, default=90.0)
    p.add_argument("--sweep", action="store_true",
                   help="dispersed Monte-Carlo comparison over --n-seeds seeds")
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--seed0", type=int, default=100,
                   help="first dispersed seed (seeds are seed0..seed0+n-1)")
    p.add_argument("--workers", type=int, default=4,
                   help="process-pool workers for --sweep")
    p.add_argument("--outdir", type=str, default=os.path.join(ROOT, "results"))
    p.add_argument("--figdir", type=str,
                   default=os.path.join(ROOT, "docs", "figures"))
    p.add_argument("--no-figs", action="store_true")
    return p.parse_args(argv)


def run_sweep(args) -> int:
    """Dispersed n-seed-per-controller comparison via a process pool."""
    guidance = args.guidance or "optimal"
    nav = args.nav or "ekf"
    names = list(args.controllers)
    seeds = list(range(args.seed0, args.seed0 + args.n_seeds))

    jobs = [
        {"kind": "sweep", "guidance": guidance, "controller": c, "nav": nav,
         "seed": s, "dt": args.dt, "control_hz": args.control_hz,
         "t_end": args.t_end, "dispersions": dispersion_draws(s, args.seed0)}
        for c in names for s in seeds
    ]
    # Nominal (undispersed) runs for the tracking overlay figure.
    jobs += [
        {"kind": "nominal", "guidance": guidance, "controller": c, "nav": nav,
         "seed": args.seed, "dt": args.dt, "control_hz": args.control_hz,
         "t_end": args.t_end, "dispersions": None, "return_trace": True}
        for c in names
    ]

    t0 = time.perf_counter()
    records: list = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_trial, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            rec = fut.result()
            records.append(rec)
            done += 1
            td = f"seed={rec['seed']} " if rec["kind"] == "sweep" else "nominal "
            print(f"[{done}/{len(jobs)}] {rec['controller']:>9s} {td}"
                  f"{'OK ' if rec.get('success') else 'FAIL:' + str(rec.get('failure_mode')):>18s}"
                  f" wall={rec['wall_time_s']:.1f}s", flush=True)
    wall = time.perf_counter() - t0

    table = {}
    for c in names:
        recs = [r for r in records
                if r["controller"] == c and r["kind"] == "sweep"]
        table[c] = summarize_runs(recs)
    traces = {r["controller"]: r["trace"] for r in records
              if r["kind"] == "nominal" and r.get("trace")}

    os.makedirs(args.outdir, exist_ok=True)
    json_path = os.path.join(args.outdir, "controller_comparison.json")
    payload = {
        "mode": "dispersed_mc",
        "guidance": guidance, "nav": nav,
        "seeds": seeds, "dt": args.dt, "control_hz": args.control_hz,
        "dispersion_spec": {
            "wind_speed": "U(0,12) m/s", "gust_sigma": 1.5,
            "density_scale": "U(-0.10,0.10)", "thrust_scale": "U(0.97,1.03)",
            "mass_offset": "U(-3%,3%) x 37000 kg",
            "pos_offset": "U(-50,50) m/axis", "vel_offset": "U(-5,5) m/s/axis",
            "meas_delay": 0.1,
            "gps_outage": "10 s window at U(10,18) s on even-indexed seeds",
        },
        "controllers": table,
        "wall_time_s": wall,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    fig_paths = []
    if not args.no_figs:
        fig_paths.append(make_comparison_figure(
            table, names,
            f"controller comparison: {guidance} / {nav}, "
            f"{args.n_seeds} dispersed seeds",
            os.path.join(args.figdir, "controller_comparison.png")))
        if traces:
            fig_paths.append(make_tracking_figure(
                traces,
                f"nominal tracking: {guidance} / {nav} (seed {args.seed})",
                os.path.join(args.figdir, "controller_tracking.png")))

    print(f"\n===== controller sweep ({guidance} / {nav}) "
          f"[wall {wall:.0f}s] =====")
    for c in names:
        s = table[c]
        print(f"{c:>9s}: {s['successes']}/{s['n']} ok "
              f"({s['success_rate']:.0%} CI95 "
              f"[{s['wilson95'][0]:.2f},{s['wilson95'][1]:.2f}]) "
              f"vs_p95={s['vs_p95']} fuel={s['fuel_used_mean']} kg "
              f"ct={s['compute_time_ms_mean']} ms "
              f"modes={s['failure_modes']}")
    print(f"json: {json_path}")
    for p_ in fig_paths:
        print(f"figure: {p_}")
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.sweep:
        return run_sweep(args)

    guidance = args.guidance or "zemzev"
    nav = args.nav or "perfect"
    results, table = {}, {}
    for cname in args.controllers:
        sim, veh, x0, meta = build_sim(guidance, cname, nav,
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
                             f"compare_controllers_{guidance}.json")
    payload = {"guidance": guidance, "nav": nav,
               "seed": args.seed, "controllers": table}
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    fig_path = None
    if not args.no_figs:
        fig_path = make_figure(
            results, guidance,
            os.path.join(args.figdir,
                         f"compare_controllers_{guidance}.png"))

    n_ok = sum(1 for v in table.values() if v.get("success"))
    print(f"\n{n_ok}/{len(table)} controllers landed successfully")
    print(f"json: {json_path}")
    if fig_path:
        print(f"figure: {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
