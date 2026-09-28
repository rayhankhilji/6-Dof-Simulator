#!/usr/bin/env python
"""Train the gate-state landing-success classifier on Monte Carlo data.

Example
-------
    .venv/bin/python scripts/train_success_model.py \
        --npz results/monte_carlo_ml.npz --gate 1500 --feature-set both

Writes a joblib bundle (model + scaler + feature_names + meta), a metrics
JSON, and the ``ml_*.png`` figure set under ``docs/figures/``.  With
multiple gates / ``--feature-set both``, non-primary combos get a
``_g<alt>_<feature_set>`` suffix on the output paths; ``--out``/figures keep
the requested names for the first combo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sixdof.ml.features import build_features
from sixdof.ml.train import probability_map, train_success_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", type=str,
                   default=os.path.join(ROOT, "results", "monte_carlo_ml.npz"))
    p.add_argument("--gate", type=float, nargs="+", default=[1500.0],
                   help="altitude gate(s) [m] to train at")
    p.add_argument("--feature-set", choices=["observable", "oracle", "both"],
                   default="observable")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str,
                   default=os.path.join(ROOT, "results",
                                        "ml_success_model.pkl"))
    p.add_argument("--metrics-out", type=str,
                   default=os.path.join(ROOT, "results", "ml_metrics.json"))
    p.add_argument("--figdir", type=str,
                   default=os.path.join(ROOT, "docs", "figures"))
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _fig_roc(combos, path, base_rate):
    """ROC + PR panels; one entry per trained combo (model curves inside)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"observable": "tab:blue", "oracle": "tab:orange"}
    for label, res in combos:
        c = colors.get(res["metrics"]["feature_set"], None)
        curves = res["curves"]
        for mname, ls in (("hgb", "-"), ("logreg", "--")):
            if mname not in curves["roc"]:
                continue
            fpr, tpr, _ = curves["roc"][mname]
            axes[0].plot(fpr, tpr, ls, color=c, lw=1.4,
                         label=f"{label} {mname}")
            prec, rec, _ = curves["pr"][mname]
            axes[1].plot(rec, prec, ls, color=c, lw=1.4,
                         label=f"{label} {mname}")
        fpr, tpr, _ = curves["roc_cal"]
        axes[0].plot(fpr, tpr, "-", color=c, lw=2.4, alpha=0.45,
                     label=f"{label} calibrated")
        prec, rec, _ = curves["pr_cal"]
        axes[1].plot(rec, prec, "-", color=c, lw=2.4, alpha=0.45,
                     label=f"{label} calibrated")
    axes[0].plot([0, 1], [0, 1], "k:", lw=0.8)
    axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC")
    axes[1].axhline(base_rate, color="k", ls=":", lw=0.8,
                    label=f"base rate {base_rate:.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision",
                title="Precision-Recall")
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_calibration(res, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cal = res["curves"]["calibration"]
    axes[0].plot([0, 1], [0, 1], "k:", lw=0.8)
    axes[0].plot(cal["mean_pred"], cal["frac_observed"], "o-",
                 label="calibrated (test)")
    axes[0].set(xlabel="mean predicted P(success)", ylabel="observed rate",
                title="Reliability diagram")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    prob, yt = res["curves"]["prob_test"], res["curves"]["y_test"]
    bins = np.linspace(0, 1, 41)
    axes[1].hist(prob[yt == 1], bins=bins, density=True, alpha=0.6,
                 label="successes")
    axes[1].hist(prob[yt == 0], bins=bins, density=True, alpha=0.6,
                 label="failures")
    axes[1].set(xlabel="predicted P(success)", ylabel="density",
                title="Predicted-probability histogram")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_importance(res, path):
    imp = res["metrics"]["permutation_importance"]
    names = [d["name"] for d in imp][::-1]
    mean = [d["mean"] for d in imp][::-1]
    std = [d["std"] for d in imp][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(names) + 1.5))
    ax.barh(names, mean, xerr=std, color="tab:blue", alpha=0.8)
    ax.set(xlabel="permutation importance (PR-AUC drop)",
           title="Top-20 permutation importance (test)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_prob_map(res, raw, feature_names, feature_set, gate_alt, path):
    model = res["bundle"]["model"]

    def pp(X):
        return model.predict_proba(np.asarray(X, dtype=np.float64))

    v_abs = np.abs(raw["v"][:, 2])
    vz_max = float(np.percentile(v_abs, 99))
    z_grid = np.linspace(max(0.05 * gate_alt, 5.0), gate_alt, 40)
    vz_grid = np.linspace(0.0, max(vz_max, 1.0), 40)
    zz, vv, prob = probability_map(pp, raw, feature_names, feature_set,
                                   z_grid, vz_grid)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    pcm = ax.pcolormesh(vv, zz, prob, cmap="RdYlGn", vmin=0.0, vmax=1.0,
                        shading="auto")
    fig.colorbar(pcm, ax=ax, label="P(success)")
    ok = raw["y"] == 1
    vz_runs = np.abs(raw["v"][:, 2])
    rng = np.random.default_rng(0)
    zjit = raw["r"][:, 2] + rng.uniform(-8.0, 8.0, size=raw["r"].shape[0])
    ax.scatter(vz_runs[ok], zjit[ok], s=4, c="k", marker=".", alpha=0.4,
               label="success")
    ax.scatter(vz_runs[~ok], zjit[~ok], s=4, c="tab:red", marker=".",
               alpha=0.25, label="failure")
    ax.set(xlabel="|v_z| at gate [m/s]", ylabel="altitude [m]",
           title=f"P(success | state) -- gate {gate_alt:g} m, {feature_set}")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(vz_grid.min(), vz_grid.max())
    ax.set_ylim(z_grid.min(), z_grid.max())
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_threshold(res, path):
    dec = res["metrics"]["deciles"]
    mp = [d["mean_pred"] for d in dec]
    ob = [d["observed"] for d in dec]
    ct = [d["count"] for d in dec]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    sc = ax.scatter(mp, ob, c=np.log10(np.maximum(ct, 1)), cmap="viridis",
                    s=60, zorder=3)
    fig.colorbar(sc, ax=ax, label="log10(count)")
    ax.plot(mp, ob, "-", lw=1, alpha=0.6)
    ax.set(xlabel="predicted P(success) (decile mean)",
           ylabel="observed success rate",
           title="Observed success rate vs predicted-prob decile")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    t_start = time.perf_counter()
    feature_sets = (["observable", "oracle"]
                    if args.feature_set == "both" else [args.feature_set])

    combos = []          # (label, result, raw)
    bundles = {}
    for gate in args.gate:
        for fs in feature_sets:
            label = f"g{gate:g}_{fs}"
            print(f"[features] gate={gate:g} feature_set={fs}", flush=True)
            X, y, names, meta = build_features(args.npz, gate, fs)
            print(f"  X={X.shape} success_rate={meta['success_rate']:.4f} "
                  f"dropped={meta['n_dropped']}", flush=True)
            res = train_success_model(X, y, names, meta, seed=args.seed)
            m = res["metrics"]
            print(f"  winner={m['winner']} "
                  f"ROC-AUC={m['test_calibrated']['roc_auc']:.4f} "
                  f"PR-AUC={m['test_calibrated']['pr_auc']:.4f} "
                  f"Brier={m['test_calibrated']['brier']:.4f}", flush=True)
            res["_label"] = label
            res["_gate"] = gate
            res["_fs"] = fs
            res["_raw"] = meta["raw"]
            res["_raw"]["y"] = y
            combos.append((label, res))
            bundles[label] = res["bundle"]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_out) or ".", exist_ok=True)
    os.makedirs(args.figdir, exist_ok=True)

    # ---- model bundle ----
    if len(bundles) == 1:
        joblib.dump(next(iter(bundles.values())), args.out)
    else:
        joblib.dump({"combos": bundles, "primary": combos[0][0]}, args.out)
    print(f"[saved] {args.out}")

    # ---- metrics json ----
    metrics = {
        "npz": os.path.abspath(args.npz),
        "seed": args.seed,
        "combos": {label: res["metrics"] for label, res in combos},
    }
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[saved] {args.metrics_out}")

    # ---- figures ----
    fig_paths = []
    for ci, (label, res) in enumerate(combos):
        suffix = "" if ci == 0 else f"_{label}"
        gate, fs, raw = res["_gate"], res["_fs"], res["_raw"]
        base = res["metrics"]["base_rate_test"]
        sub = [c for c in combos if c[1]["_gate"] == gate]
        p = os.path.join(args.figdir, f"ml_roc{suffix}.png")
        _fig_roc(sub, p, base)
        fig_paths.append(p)
        for fname, fn in (
            ("ml_calibration", lambda pp: _fig_calibration(res, pp)),
            ("ml_importance", lambda pp: _fig_importance(res, pp)),
            ("ml_threshold", lambda pp: _fig_threshold(res, pp)),
        ):
            p = os.path.join(args.figdir, f"{fname}{suffix}.png")
            fn(p)
            fig_paths.append(p)
        p = os.path.join(args.figdir, f"ml_prob_map{suffix}.png")
        _fig_prob_map(res, raw, res["bundle"]["feature_names"], fs, gate, p)
        fig_paths.append(p)
    for p in fig_paths:
        print(f"[fig] {p}")

    print(f"done in {time.perf_counter() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
