#!/usr/bin/env python
"""Compare powered-descent guidance laws on the landing scenario.

Two modes:

* single-seed overlay (default): each requested guidance law with the same
  controller/nav/seed -> ``results/compare_guidance_<c>.json`` +
  ``docs/figures/compare_guidance_<c>.png``.
* dispersed sweep (``--sweep``): n dispersed seeds per guidance law
  (defaults: controller=mpc, nav=ekf, seeds 100..119; same dispersion set
  as ``compare_controllers.py --sweep``), parallelized over a process pool
  -> ``results/guidance_comparison.json``,
  ``docs/figures/guidance_comparison.png`` and
  ``docs/figures/guidance_fuel.png``.

Example
-------
    .venv/bin/python scripts/compare_guidance.py --controller pid
    .venv/bin/python scripts/compare_guidance.py --sweep --workers 5
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


def make_fuel_figure(table: dict, names: list, title: str, path: str) -> str:
    """Scatter: fuel used vs initial lateral offset, colored by guidance."""
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    fig.suptitle(title)
    colors = plt.get_cmap("tab10")
    for i, name in enumerate(names):
        runs = table[name]["runs"]
        ok_x = [r["initial_lateral_offset"] for r in runs
                if r.get("success") and r.get("initial_lateral_offset") is not None]
        ok_y = [r["fuel_used_kg"] for r in runs
                if r.get("success") and r.get("initial_lateral_offset") is not None]
        f_x = [r["initial_lateral_offset"] for r in runs
               if not r.get("success") and r.get("initial_lateral_offset") is not None]
        f_y = [r["fuel_used_kg"] for r in runs
               if not r.get("success") and r.get("initial_lateral_offset") is not None]
        ax.scatter(ok_x, ok_y, s=28, color=colors(i), alpha=0.85,
                   edgecolor="k", lw=0.4, label=name)
        if f_x:
            ax.scatter(f_x, f_y, s=40, color=colors(i), marker="x",
                       lw=1.2, label=f"{name} (fail)")
    ax.set_xlabel("initial lateral offset [m]")
    ax.set_ylabel("fuel used [kg]")
    ax.grid(alpha=0.3)
    ax.legend()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--controller", choices=list(CONTROLLERS), default=None,
                   help="default: pid (single-seed) / mpc (--sweep)")
    p.add_argument("--guidance", nargs="+", choices=DESCENT_GUIDANCE,
                   default=None, help="default: all descent laws")
    p.add_argument("--nav", choices=["perfect", "ekf", "ukf"], default=None,
                   help="default: perfect (single-seed) / ekf (--sweep)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--control-hz", type=float, default=50.0)
    p.add_argument("--t-end", type=float, default=90.0)
    p.add_argument("--sweep", action="store_true",
                   help="dispersed Monte-Carlo comparison over --n-seeds seeds")
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--seed0", type=int, default=100)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--outdir", type=str, default=os.path.join(ROOT, "results"))
    p.add_argument("--figdir", type=str,
                   default=os.path.join(ROOT, "docs", "figures"))
    p.add_argument("--no-figs", action="store_true")
    return p.parse_args(argv)


def run_sweep(args) -> int:
    """Dispersed n-seed-per-guidance comparison via a process pool."""
    controller = args.controller or "mpc"
    nav = args.nav or "ekf"
    names = list(args.guidance or DESCENT_GUIDANCE)
    seeds = list(range(args.seed0, args.seed0 + args.n_seeds))

    jobs = [
        {"kind": "sweep", "guidance": g, "controller": controller, "nav": nav,
         "seed": s, "dt": args.dt, "control_hz": args.control_hz,
         "t_end": args.t_end, "dispersions": dispersion_draws(s, args.seed0)}
        for g in names for s in seeds
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
            print(f"[{done}/{len(jobs)}] {rec['guidance']:>8s} "
                  f"seed={rec['seed']} "
                  f"{'OK ' if rec.get('success') else 'FAIL:' + str(rec.get('failure_mode')):>18s}"
                  f" wall={rec['wall_time_s']:.1f}s", flush=True)
    wall = time.perf_counter() - t0

    table = {}
    for g in names:
        recs = [r for r in records if r["guidance"] == g]
        table[g] = summarize_runs(recs)

    os.makedirs(args.outdir, exist_ok=True)
    json_path = os.path.join(args.outdir, "guidance_comparison.json")
    payload = {
        "mode": "dispersed_mc",
        "controller": controller, "nav": nav,
        "seeds": seeds, "dt": args.dt, "control_hz": args.control_hz,
        "dispersion_spec": {
            "wind_speed": "U(0,12) m/s", "gust_sigma": 1.5,
            "density_scale": "U(-0.10,0.10)", "thrust_scale": "U(0.97,1.03)",
            "mass_offset": "U(-3%,3%) x 37000 kg",
            "pos_offset": "U(-50,50) m/axis", "vel_offset": "U(-5,5) m/s/axis",
            "meas_delay": 0.1,
            "gps_outage": "10 s window at U(10,18) s on even-indexed seeds",
        },
        "guidance": table,
        "wall_time_s": wall,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    fig_paths = []
    if not args.no_figs:
        fig_paths.append(make_comparison_figure(
            table, names,
            f"guidance comparison: {controller} / {nav}, "
            f"{args.n_seeds} dispersed seeds",
            os.path.join(args.figdir, "guidance_comparison.png")))
        fig_paths.append(make_fuel_figure(
            table, names,
            f"fuel use vs initial offset: {controller} / {nav}",
            os.path.join(args.figdir, "guidance_fuel.png")))

    print(f"\n===== guidance sweep ({controller} / {nav}) "
          f"[wall {wall:.0f}s] =====")
    for g in names:
        s = table[g]
        print(f"{g:>8s}: {s['successes']}/{s['n']} ok "
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

    controller = args.controller or "pid"
    nav = args.nav or "perfect"
    guidance_list = args.guidance or DESCENT_GUIDANCE
    results, table = {}, {}
    for gname in guidance_list:
        sim, veh, x0, meta = build_sim(gname, controller, nav,
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
                             f"compare_guidance_{controller}.json")
    payload = {"controller": controller, "nav": nav,
               "seed": args.seed, "guidance": table}
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    fig_path = None
    if not args.no_figs:
        fig_path = make_figure(
            results, controller,
            os.path.join(args.figdir,
                         f"compare_guidance_{controller}.png"))

    n_ok = sum(1 for v in table.values() if v.get("success"))
    print(f"\n{n_ok}/{len(table)} guidance laws landed successfully")
    print(f"json: {json_path}")
    if fig_path:
        print(f"figure: {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
