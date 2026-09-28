#!/usr/bin/env python
"""Run the Phase-5b guidance-parameter policy search (CEM).

Example
-------
    .venv/bin/python scripts/run_policy_search.py \
        --pop 16 --gens 12 --n-eval 1000 --n-final 50000 --jobs 8

Writes ``results/policy_search.json``, ``docs/figures/policy_learning.png``
and ``docs/figures/policy_comparison.png``.

The search uses common random numbers (one shared batch seed per
generation) and injects each candidate theta through
``BatchMonteCarlo(guidance_params=...)``.  Population slot 0 is always the
baseline theta, giving a paired baseline curve on the same eval budget.
After the search, the baseline and learned theta are both evaluated on the
SAME fresh ``--n-final`` seed so the comparison is fully paired.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sixdof.batch import DispersionConfig, FAILURE_MODES, summarize, wilson_ci
from sixdof.batch.analysis import _MODE_COLORS
from sixdof.batch.policy_search import (
    HI, LO, PARAM_NAMES, PARAMS, THETA0, _run_star, cem,
    theta_to_guidance_params,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pop", type=int, default=16)
    p.add_argument("--gens", type=int, default=12)
    p.add_argument("--n-eval", type=int, default=1000,
                   help="batch runs per candidate evaluation")
    p.add_argument("--n-final", type=int, default=50_000,
                   help="fresh-seed runs for the baseline/learned final eval")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--elite", type=int, default=5)
    p.add_argument("--sigma0-frac", type=float, default=0.3,
                   help="initial sigma as a fraction of (hi - lo)")
    p.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1),
                   help="worker processes for population evaluation")
    p.add_argument("--budget", type=float, default=9_000.0,
                   help="CEM wall-clock cap [s] (~2.5 h)")
    p.add_argument("--control-noise", type=float, default=0.002,
                   help="gimbal-command noise sigma [rad] (matches "
                        "run_monte_carlo study)")
    p.add_argument("--out", type=str,
                   default=os.path.join(ROOT, "results", "policy_search.json"))
    p.add_argument("--figdir", type=str,
                   default=os.path.join(ROOT, "docs", "figures"))
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_learning(history: dict, theta_best: np.ndarray, path: str,
                 pop: int = 0) -> str:
    """Reward learning curve (elite-mean band) + per-param mu/sigma grid."""
    gens = np.asarray(history["gen"])
    best = np.asarray(history["best_reward"])
    mean = np.asarray(history["mean_reward"])
    em = np.asarray(history["elite_mean_reward"])
    es = np.asarray(history["elite_std"])
    base = np.asarray(history["baseline_reward"])
    mu = np.asarray(history["mu"])          # (gens, d)
    sig = np.asarray(history["sigma"])      # (gens, d)

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(4, 4, hspace=0.55, wspace=0.35,
                          left=0.06, right=0.985, top=0.93, bottom=0.08)
    ax = fig.add_subplot(gs[:, :2])

    ax.fill_between(gens, em - es, em + es, color="#1f77b4", alpha=0.2,
                    label="elite $\\pm 1\\sigma$")
    ax.plot(gens, mean, color="#9ecae1", lw=1.2, label="pop mean")
    ax.plot(gens, em, color="#1f77b4", lw=2.0, label="elite mean")
    ax.plot(gens, best, color="#d62728", lw=1.6, ls="--", label="gen best")
    ax.plot(gens, base, color="#2ca02c", lw=1.4, ls=":",
            label="baseline $\\theta_0$ (paired)")
    ax.set_xlabel("generation")
    ax.set_ylabel("reward")
    ax.set_title("CEM policy search -- reward per generation "
                 "(common random numbers within each gen)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    if pop > 0:
        ax2 = ax.secondary_xaxis(
            "top", functions=(lambda g, p=pop: (g + 1) * p,
                              lambda e, p=pop: e / p - 1))
        ax2.set_xlabel("cumulative evaluations")

    # Per-parameter convergence grid (4 x 2).
    for i, prm in enumerate(PARAMS):
        a = fig.add_subplot(gs[i // 2, 2 + i % 2])
        a.fill_between(gens, mu[:, i] - sig[:, i], mu[:, i] + sig[:, i],
                       color="#1f77b4", alpha=0.25)
        a.plot(gens, mu[:, i], color="#1f77b4", lw=1.5, label="$\\mu$")
        a.axhline(prm.default, color="#2ca02c", ls=":", lw=1.2,
                  label="default")
        a.axhline(theta_best[i], color="#d62728", ls="--", lw=1.2,
                  label="learned")
        a.set_ylim(prm.lo, prm.hi)
        a.set_title(f"{prm.name}  [{prm.lo:g}, {prm.hi:g}]", fontsize=9)
        a.grid(alpha=0.3)
        a.tick_params(labelsize=7)
        if i // 2 == 3:
            a.set_xlabel("gen", fontsize=8)
        if i == 0:
            a.legend(fontsize=7, loc="best")
    fig.suptitle("Guidance-parameter convergence ($\\mu \\pm \\sigma$ "
                 "within bounds)", fontsize=11, y=0.975)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_comparison(res0, res1, path: str) -> str:
    """Baseline vs learned: success CI, fuel, |v_z| CDF, failure modes."""
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8))
    labels = ["baseline", "learned"]
    res_list = [res0, res1]
    colors = ["#1f77b4", "#d62728"]

    # (a) success rate with Wilson CI
    ax = axes[0]
    for i, res in enumerate(res_list):
        k = int(res.success.sum())
        lo, hi = wilson_ci(k, res.n)
        p = k / res.n
        ax.bar(i, 100 * p, color=colors[i], alpha=0.85, width=0.55)
        ax.errorbar(i, 100 * p,
                    yerr=[[max(0.0, 100 * (p - lo))],
                          [max(0.0, 100 * (hi - p))]],
                    color="black", capsize=5, lw=1.4)
        ax.text(i, 100 * p + 2.5, f"{100 * p:.2f}%", ha="center",
                fontsize=10)
    ax.set_xticks(range(2))
    ax.set_xticklabels(labels)
    ax.set_ylabel("success rate [%]")
    ax.set_ylim(0, 105)
    ax.set_title(f"Success rate (n={res0.n:,}, Wilson 95% CI)")
    ax.grid(alpha=0.3, axis="y")

    # (b) fuel-remaining distribution
    ax = axes[1]
    bins = np.linspace(-0.02, 1.02, 60)
    for res, c, lab in zip(res_list, colors, labels):
        ax.hist(np.clip(res.fuel_remaining_frac, -0.05, 1.05), bins=bins,
                density=True, histtype="step", lw=1.8, color=c, label=lab)
    ax.set_xlabel("fuel remaining / initial landing propellant")
    ax.set_ylabel("density")
    ax.set_title("Fuel remaining at touchdown")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (c) touchdown |v_z| empirical CDF
    ax = axes[2]
    for res, c, lab in zip(res_list, colors, labels):
        vz = res.vertical_speed[np.isfinite(res.vertical_speed)
                                & np.isfinite(res.touchdown_time)]
        vz = np.sort(vz)
        ax.plot(vz, np.linspace(0, 1, vz.size), color=c, lw=1.8, label=lab)
    ax.axvline(3.0, color="black", ls="--", lw=1.2, label="|v_z| limit 3 m/s")
    ax.set_xlabel("touchdown $|v_z|$ [m/s]")
    ax.set_ylabel("CDF")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.set_title("Touchdown $|v_z|$ CDF")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)

    # (d) failure-mode stacked bars (normalized, success excluded)
    ax = axes[3]
    modes = [m for m in FAILURE_MODES if m != "success"]
    bottoms = np.zeros(2)
    for mode in modes:
        mi = FAILURE_MODES.index(mode)
        frac = np.array([np.mean(res.failure_mode == mi)
                         for res in res_list])
        ax.bar([0, 1], 100 * frac, bottom=100 * bottoms, width=0.55,
               color=_MODE_COLORS[mode], label=mode)
        bottoms += frac
    ax.set_xticks(range(2))
    ax.set_xticklabels(labels)
    ax.set_ylabel("share of runs [%]")
    ax.set_title("Failure-mode mix (success excluded)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3, axis="y")

    sr0, sr1 = res0.success.mean(), res1.success.mean()
    fig.suptitle(f"Policy comparison: baseline {100 * sr0:.2f}% vs learned "
                 f"{100 * sr1:.2f}% success "
                 f"($\\Delta$={100 * (sr1 - sr0):+.2f} pts)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def main(argv=None):
    args = parse_args(argv)
    t_all = time.perf_counter()
    disp = DispersionConfig(control_noise=args.control_noise)

    sigma0 = args.sigma0_frac * (HI - LO)
    print(f"[policy_search] pop={args.pop} gens={args.gens} "
          f"n_eval={args.n_eval} elite={args.elite} jobs={args.jobs} "
          f"seed={args.seed}", flush=True)
    out = cem(THETA0, sigma0, args.n_eval, pop=args.pop, elite=args.elite,
              gens=args.gens, seed=args.seed, dispersion=disp,
              n_jobs=args.jobs, budget_s=args.budget, verbose=True)
    theta_best = out["theta_best"]
    print(f"[policy_search] search done: gens_run={out['gens_run']} "
          f"wall={out['wall_time_s'] / 60:.1f} min "
          f"best_reward={out['best_reward']:.4f}", flush=True)
    print("learned theta:", dict(zip(PARAM_NAMES,
                                     np.round(theta_best, 4).tolist())),
          flush=True)
    print("guidance_params:", theta_to_guidance_params(theta_best),
          flush=True)

    # ---- final paired evaluation on the SAME fresh seed ----------------
    final_seed = args.seed + 10_000
    t_fin = time.perf_counter()
    jobs = max(1, args.jobs)
    if jobs >= 2:
        with ProcessPoolExecutor(max_workers=2) as ex:
            futs = [
                ex.submit(_run_star,
                          (THETA0, args.n_final, final_seed, disp,
                           0.02, 120.0)),
                ex.submit(_run_star,
                          (theta_best, args.n_final, final_seed, disp,
                           0.02, 120.0)),
            ]
            res_base, res_learned = futs[0].result(), futs[1].result()
    else:
        res_base = _run_star((THETA0, args.n_final, final_seed, disp,
                              0.02, 120.0))
        res_learned = _run_star((theta_best, args.n_final, final_seed, disp,
                                 0.02, 120.0))
    fin_wall = time.perf_counter() - t_fin
    print(f"[policy_search] final eval n={args.n_final} x2 "
          f"wall={fin_wall / 60:.1f} min  "
          f"baseline={100 * res_base.success.mean():.2f}%  "
          f"learned={100 * res_learned.success.mean():.2f}%", flush=True)

    # ---- JSON -----------------------------------------------------------
    wall = time.perf_counter() - t_all
    history = {k: ([x.tolist() if isinstance(x, np.ndarray) else x
                    for x in v] if isinstance(v, list) else v)
               for k, v in out["history"].items()}
    doc = {
        "args": vars(args),
        "theta": {
            "names": PARAM_NAMES,
            "baseline": THETA0.tolist(),
            "learned": theta_best.tolist(),
            "bounds_lo": LO.tolist(),
            "bounds_hi": HI.tolist(),
            "guidance_params": theta_to_guidance_params(theta_best),
        },
        "baseline": summarize(res_base),
        "learned": summarize(res_learned),
        "history": history,
        "meta": {
            "gens_run": out["gens_run"],
            "search_wall_s": out["wall_time_s"],
            "final_wall_s": fin_wall,
            "wall_time_s": wall,
            "n_eval": args.n_eval,
            "final_seed": final_seed,
            "best_reward": out["best_reward"],
            "final_mu": out["mu"].tolist(),
            "final_sigma": out["sigma"].tolist(),
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2, default=_jsonable)

    # ---- figures ---------------------------------------------------------
    os.makedirs(args.figdir, exist_ok=True)
    p1 = fig_learning(out["history"], theta_best,
                      os.path.join(args.figdir, "policy_learning.png"),
                      pop=args.pop)
    p2 = fig_comparison(res_base, res_learned,
                        os.path.join(args.figdir, "policy_comparison.png"))

    # ---- stdout report ---------------------------------------------------
    sb, sl = doc["baseline"], doc["learned"]
    print("\n===== policy search summary =====")
    print(f"gens run: {out['gens_run']}   evals: "
          f"{out['gens_run'] * args.pop}   wall: {wall / 60:.1f} min")
    print(f"success:  baseline {100 * sb['success_rate']:.2f}%  ->  "
          f"learned {100 * sl['success_rate']:.2f}%")
    fs_b = sb["fuel_remaining_frac"]["success"]
    fs_l = sl["fuel_remaining_frac"]["success"]
    print(f"fuel remaining (successes) mean: {fs_b['mean']:.3f} -> "
          f"{fs_l['mean']:.3f}")
    print("failure modes (baseline -> learned):")
    for m in FAILURE_MODES:
        if m == "success":
            continue
        cb = sb["failure_counts"][m]
        cl = sl["failure_counts"][m]
        if cb or cl:
            print(f"  {m:20s} {cb:7d} -> {cl:7d}")
    print(f"json: {args.out}\nfigs: {p1}, {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
