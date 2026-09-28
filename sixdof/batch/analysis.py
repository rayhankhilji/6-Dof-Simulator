"""Monte Carlo result summarization and figure builders (Phase 4).

``summarize(res)`` returns a JSON-serializable dict: success rate,
per-failure-mode counts, touchdown-metric percentiles, fuel stats, and
3-sigma dispersion ellipse parameters at each altitude gate.

``make_figures(res, outdir)`` writes the standard PNG set:

* ``mc_dispersion.png`` -- top-down scatter at the altitude gates with
  3-sigma covariance ellipses.
* ``mc_touchdown.png`` -- touchdown map (colored by failure mode, pad
  circle r = 10 m) plus histograms of the four touchdown metrics with
  the success thresholds drawn.
* ``mc_fuel.png`` -- fuel-remaining distribution, successes vs failures.
* ``mc_factors.png`` -- P(success) vs wind speed, mass offset, thrust
  scale, density scale, and gust sigma, with Wilson binomial CIs.
* ``mc_failures.png`` -- failure-mode bar chart and touchdown-time
  distribution stacked by mode (proxy for when failures occur).
"""

from __future__ import annotations

import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle

from .batch_mc import FAILURE_MODES, MCResult


# ---------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / den
    h = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return (float(c - h), float(c + h))


def _ellipse_params(xy: np.ndarray) -> dict:
    """3-sigma dispersion ellipse from a (m,2) position sample."""
    out = {"n": int(xy.shape[0]), "cx": float("nan"), "cy": float("nan"),
           "semi_major": float("nan"), "semi_minor": float("nan"),
           "angle_deg": float("nan")}
    if xy.shape[0] < 3:
        return out
    c = xy.mean(axis=0)
    cov = np.cov((xy - c).T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 0.0)
    vecs = vecs[:, order]
    out.update(
        cx=float(c[0]), cy=float(c[1]),
        semi_major=float(3.0 * np.sqrt(vals[0])),
        semi_minor=float(3.0 * np.sqrt(vals[1])),
        angle_deg=float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))),
    )
    return out


def summarize(res: MCResult) -> dict:
    """Aggregate an ``MCResult`` into a JSON-serializable summary dict."""
    n = res.n
    touched = ~np.isnan(res.touchdown_time)
    succ = res.success

    counts = {m: int(np.sum(res.failure_mode == i))
              for i, m in enumerate(FAILURE_MODES)}

    def pct(x):
        x = x[np.isfinite(x)]
        if x.size == 0:
            return {"p50": float("nan"), "p95": float("nan"),
                    "p99": float("nan"), "mean": float("nan")}
        return {"p50": float(np.percentile(x, 50)),
                "p95": float(np.percentile(x, 95)),
                "p99": float(np.percentile(x, 99)),
                "mean": float(np.mean(x))}

    td = {
        "vertical_speed": pct(res.vertical_speed[touched]),
        "lateral_speed": pct(res.lateral_speed[touched]),
        "tilt_deg": pct(res.tilt_deg[touched]),
        "lateral_offset": pct(res.lateral_offset[touched]),
        "touchdown_time": pct(res.touchdown_time[touched]),
    }

    fuel = res.fuel_remaining_frac
    fuel_stats = {
        "all": pct(fuel),
        "success": pct(fuel[succ]),
        "failure": pct(fuel[~succ]),
    }

    gates = []
    for gi, alt in enumerate(res.gate_altitudes):
        xy = res.gate_positions[:, gi, :2]
        ok = np.isfinite(xy).all(axis=1)
        ep = _ellipse_params(xy[ok])
        ep["altitude_m"] = float(alt)
        gates.append(ep)

    return {
        "n": int(n),
        "runtime_s": float(res.runtime_s),
        "dt": float(res.dt),
        "seed": int(res.seed),
        "success_rate": float(np.mean(succ)),
        "n_success": int(np.sum(succ)),
        "failure_counts": counts,
        "failure_rates": {m: c / n for m, c in counts.items()},
        "touchdown": td,
        "fuel_remaining_frac": fuel_stats,
        "gates": gates,
        "failure_modes": list(FAILURE_MODES),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

_MODE_COLORS = {
    "success": "#2ca02c", "hard_landing": "#d62728",
    "lateral_velocity": "#ff7f0e", "tipover": "#9467bd",
    "miss_pad": "#8c564b", "fuel_exhausted": "#e377c2",
    "engine_failure": "#7f7f7f", "gps_outage_diverged": "#bcbd22",
    "timeout": "#17becf",
}


def _subsample(idx, max_pts, rng):
    if idx.size <= max_pts:
        return idx
    return rng.choice(idx, size=max_pts, replace=False)


def fig_dispersion(res: MCResult, path: str, max_pts: int = 4000) -> str:
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.get_cmap("viridis")
    alts = res.gate_altitudes
    for gi, alt in enumerate(alts):
        xy = res.gate_positions[:, gi, :2]
        ok = np.isfinite(xy).all(axis=1)
        idx = _subsample(np.flatnonzero(ok), max_pts, rng)
        color = cmap(gi / max(len(alts) - 1, 1))
        ax.scatter(xy[idx, 0], xy[idx, 1], s=1, alpha=0.25, color=color,
                   label=f"z={alt:.0f} m", rasterized=True)
        ep = _ellipse_params(xy[ok])
        if np.isfinite(ep["semi_major"]):
            ax.add_patch(Ellipse(
                (ep["cx"], ep["cy"]), 2 * ep["semi_major"],
                2 * ep["semi_minor"], angle=ep["angle_deg"],
                fill=False, color=color, lw=2))
    ax.add_patch(Circle((0, 0), 10.0, fill=False, color="red", lw=1.5,
                        ls="--", label="pad r=10 m"))
    ax.set_xlabel("x East [m]")
    ax.set_ylabel("y North [m]")
    ax.set_title(f"Dispersion at altitude gates (n={res.n}, 3$\\sigma$ ellipses)")
    ax.legend(markerscale=8, fontsize=8, loc="upper right")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_touchdown(res: MCResult, path: str, max_pts: int = 20000) -> str:
    rng = np.random.default_rng(1)
    touched = ~np.isnan(res.touchdown_time)
    fig, axes = plt.subplots(1, 5, figsize=(24, 5.2))
    ax = axes[0]
    xy = res.gate_positions[:, -1, :2]
    ok = np.isfinite(xy).all(axis=1) & touched
    for i, mode in enumerate(FAILURE_MODES):
        idx = np.flatnonzero(ok & (res.failure_mode == i))
        idx = _subsample(idx, max_pts, rng)
        if idx.size == 0:
            continue
        ax.scatter(xy[idx, 0], xy[idx, 1], s=2, alpha=0.35,
                   color=_MODE_COLORS[mode], label=mode, rasterized=True)
    ax.add_patch(Circle((0, 0), 10.0, fill=False, color="black", lw=1.5,
                        ls="--", label="pad r=10 m"))
    ax.set_xlabel("x East [m]")
    ax.set_ylabel("y North [m]")
    ax.set_title("Touchdown map")
    ax.legend(markerscale=6, fontsize=7)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    metrics = [
        ("vertical_speed", res.vertical_speed, 3.0, "|v_z| [m/s]"),
        ("lateral_speed", res.lateral_speed, 1.5, "lateral speed [m/s]"),
        ("tilt_deg", res.tilt_deg, 5.0, "tilt [deg]"),
        ("lateral_offset", res.lateral_offset, 10.0, "offset [m]"),
    ]
    for ax, (name, vals, thresh, label) in zip(axes[1:], metrics):
        vv = vals[touched & np.isfinite(vals)]
        vv_clip = vv[np.isfinite(vv)]
        hi = np.percentile(vv_clip, 99) if vv_clip.size else thresh * 2
        bins = np.linspace(0, max(hi, thresh * 1.5), 60)
        ax.hist(vv, bins=bins, color="#1f77b4", alpha=0.8)
        ax.axvline(thresh, color="red", lw=1.5, ls="--",
                   label=f"limit {thresh}")
        frac = float(np.mean(vv <= thresh)) if vv.size else float("nan")
        ax.set_title(f"{label}  (P(pass)={frac:.3f})")
        ax.set_xlabel(label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Touchdown metrics (n={res.n}, "
                 f"success={100 * np.mean(res.success):.1f}%)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_fuel(res: MCResult, path: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    fuel = np.clip(res.fuel_remaining_frac, -0.05, 1.05)
    bins = np.linspace(-0.02, 1.02, 60)
    for mask, color, name in (
            (res.success, "#2ca02c", "success"),
            (~res.success, "#d62728", "failure")):
        vals = fuel[mask]
        if vals.size == 0:
            continue
        ax.hist(vals, bins=bins, alpha=0.7 if name == "success" else 0.6,
                color=color, label=f"{name} (n={vals.size})", density=True)
    ax.set_xlabel("fuel remaining / initial landing propellant")
    ax.set_ylabel("density")
    ax.set_title("Fuel remaining at touchdown")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _factor_panel(ax, x, succ, label, n_bins=8, logx=False):
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    x, succ = x[ok], succ[ok]
    if x.size == 0 or np.ptp(x) == 0:
        ax.text(0.5, 0.5, "no dispersion", ha="center", transform=ax.transAxes)
        ax.set_title(label)
        return
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    edges = np.unique(edges)
    centers, p_lo, p_hi, p_mid = [], [], [], []
    for i in range(len(edges) - 1):
        m = (x > edges[i]) & (x <= edges[i + 1])
        k = int(np.sum(succ[m]))
        nn = int(m.sum())
        lo, hi = wilson_ci(k, nn)
        centers.append(float(np.mean(x[m])) if nn else np.nan)
        p_mid.append(k / nn if nn else np.nan)
        p_lo.append(lo)
        p_hi.append(hi)
    centers = np.array(centers)
    ax.errorbar(centers, p_mid,
                yerr=[np.array(p_mid) - np.array(p_lo),
                      np.array(p_hi) - np.array(p_mid)],
                fmt="o-", capsize=3, lw=1.2)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(label)
    ax.set_ylabel("P(success)")
    ax.set_title(label)
    ax.grid(alpha=0.3)


def fig_factors(res: MCResult, path: str) -> str:
    factors = [
        ("wind speed [m/s]", res.wind_speed),
        ("mass scale", res.mass_scale),
        ("thrust scale", res.thrust_scale),
        ("density scale", res.density_scale),
        ("gust sigma [m/s]", res.gust_sigma),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(24, 4.5))
    for ax, (label, x) in zip(axes, factors):
        _factor_panel(ax, x, res.success, label)
    fig.suptitle(f"Success probability vs dispersion factors (n={res.n})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_failures(res: MCResult, path: str) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    counts = [int(np.sum(res.failure_mode == i))
              for i in range(len(FAILURE_MODES))]
    colors = [_MODE_COLORS[m] for m in FAILURE_MODES]
    ax1.bar(range(len(FAILURE_MODES)), counts, color=colors)
    ax1.set_xticks(range(len(FAILURE_MODES)))
    ax1.set_xticklabels(FAILURE_MODES, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("count")
    ax1.set_title("Failure-mode counts")
    ax1.grid(alpha=0.3, axis="y")
    for i, c in enumerate(counts):
        ax1.text(i, c, str(c), ha="center", va="bottom", fontsize=8)

    # Stacked touchdown-time distribution by mode (failure-time proxy).
    tt = res.touchdown_time
    ok = np.isfinite(tt)
    bins = np.linspace(0, np.nanmax(tt[ok]) if ok.any() else res.dt, 50)
    bottom = np.zeros(len(bins) - 1)
    for i, mode in enumerate(FAILURE_MODES):
        if mode == "success":
            continue
        h, _ = np.histogram(tt[ok & (res.failure_mode == i)], bins=bins)
        ax2.bar(bins[:-1], h, width=np.diff(bins), bottom=bottom,
                color=_MODE_COLORS[mode], label=mode, align="edge")
        bottom += h
    ax2.set_xlabel("touchdown time [s]")
    ax2.set_ylabel("count")
    ax2.set_title("When failures occur (touchdown time by mode)")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_figures(res: MCResult, outdir: str) -> dict:
    """Write all standard Monte Carlo figures under ``outdir``."""
    os.makedirs(outdir, exist_ok=True)
    return {
        "dispersion": fig_dispersion(res, os.path.join(outdir, "mc_dispersion.png")),
        "touchdown": fig_touchdown(res, os.path.join(outdir, "mc_touchdown.png")),
        "fuel": fig_fuel(res, os.path.join(outdir, "mc_fuel.png")),
        "factors": fig_factors(res, os.path.join(outdir, "mc_factors.png")),
        "failures": fig_failures(res, os.path.join(outdir, "mc_failures.png")),
    }
