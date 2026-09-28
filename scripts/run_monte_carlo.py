#!/usr/bin/env python
"""Run the Phase-4 Monte Carlo landing dispersion study.

Example
-------
    .venv/bin/python scripts/run_monte_carlo.py --n 100000 --chunks 4

Writes ``results/monte_carlo.npz``, ``results/mc_summary.json`` and the
figure set under ``docs/figures/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sixdof.batch import (
    BatchMonteCarlo,
    DispersionConfig,
    FAILURE_MODES,
    MCResult,
    make_figures,
    summarize,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunks", type=int, default=1,
                   help="split n into this many sequential chunks "
                        "(bounds peak memory)")
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--wind-max", type=float, default=12.0)
    p.add_argument("--density-sigma", type=float, default=0.10)
    p.add_argument("--mass-sigma", type=float, default=0.03)
    p.add_argument("--thrust-sigma", type=float, default=0.03)
    p.add_argument("--gps-outage-prob", type=float, default=0.2)
    p.add_argument("--engine-fail-prob", type=float, default=0.02)
    p.add_argument("--control-noise", type=float, default=0.002,
                   help="sigma [rad] on gimbal commands")
    p.add_argument("--gust-sigma", type=float, default=2.0,
                   help="per-run gust sigma ~ U(0, gust-sigma) [m/s]")
    p.add_argument("--ic-sigma-pos", type=float, default=5.0)
    p.add_argument("--ic-sigma-vel", type=float, default=1.0)
    p.add_argument("--meas-delay", type=float, default=0.3,
                   help="max meas-lag tau [s]; 0 disables")
    p.add_argument("--t-end", type=float, default=120.0)
    p.add_argument("--out", type=str,
                   default=os.path.join(ROOT, "results", "monte_carlo.npz"))
    p.add_argument("--summary", type=str,
                   default=os.path.join(ROOT, "results", "mc_summary.json"))
    p.add_argument("--figdir", type=str,
                   default=os.path.join(ROOT, "docs", "figures"))
    p.add_argument("--no-figs", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    disp = DispersionConfig(
        wind_max=args.wind_max,
        density_sigma=args.density_sigma,
        mass_sigma=args.mass_sigma,
        thrust_sigma=args.thrust_sigma,
        gps_outage_prob=args.gps_outage_prob,
        engine_fail_prob=args.engine_fail_prob,
        control_noise=args.control_noise,
        gust_sigma=args.gust_sigma,
        ic_sigma_pos=args.ic_sigma_pos,
        ic_sigma_vel=args.ic_sigma_vel,
        meas_delay=args.meas_delay,
    )

    n = args.n
    chunks = max(1, args.chunks)
    per = n // chunks
    sizes = [per] * chunks
    sizes[-1] += n - per * chunks  # remainder in last chunk

    parts = []
    t0 = time.perf_counter()
    for ci, nn in enumerate(sizes):
        if nn <= 0:
            continue
        mc = BatchMonteCarlo(n=nn, dt=args.dt, seed=args.seed + ci,
                             dispersion=disp, t_end=args.t_end)
        parts.append(mc.run())
        print(f"[chunk {ci + 1}/{chunks}] n={nn} "
              f"runtime={parts[-1].runtime_s:.1f}s "
              f"success={100 * parts[-1].success.mean():.2f}%",
              flush=True)
    res = MCResult.concatenate(parts)
    wall = time.perf_counter() - t0
    res.runtime_s = wall

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    res.save_npz(args.out)

    summary = summarize(res)
    summary["wall_time_s"] = wall
    summary["args"] = vars(args)
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)

    if not args.no_figs:
        figs = make_figures(res, args.figdir)
        print("figures:", ", ".join(figs.values()))

    # ---- compact stdout summary ----
    print("\n===== Monte Carlo summary =====")
    print(f"runs: {res.n}   wall: {wall:.1f}s   "
          f"({1e6 * wall / max(res.n, 1):.0f} us/run)")
    print(f"success rate: {100 * summary['success_rate']:.2f}% "
          f"({summary['n_success']}/{res.n})")
    print("failure modes:")
    for mode in FAILURE_MODES:
        c = summary["failure_counts"][mode]
        if mode == "success" or c == 0:
            continue
        print(f"  {mode:22s} {c:8d}  ({100 * c / res.n:5.2f}%)")
    td = summary["touchdown"]
    print("touchdown metrics (p50 / p95 / p99):")
    for key, label in [("vertical_speed", "|v_z| [m/s]"),
                       ("lateral_speed", "lat speed [m/s]"),
                       ("tilt_deg", "tilt [deg]"),
                       ("lateral_offset", "offset [m]"),
                       ("touchdown_time", "t_td [s]")]:
        s = td[key]
        print(f"  {label:16s} {s['p50']:8.3f} {s['p95']:8.3f} {s['p99']:8.3f}")
    fs = summary["fuel_remaining_frac"]["success"]
    print(f"fuel remaining (successes): p5-ish mean={fs['mean']:.3f} "
          f"p50={fs['p50']:.3f} p95={fs['p95']:.3f}")
    print(f"npz: {args.out}\nsummary: {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
