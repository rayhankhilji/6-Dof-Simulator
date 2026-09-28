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
from sixdof.navigation import EKF, UKF
from sixdof.scenarios import landing_scenario, landing_success
from sixdof.sensors import Barometer, GPS, IMU, RadarAltimeter, SensorSuite
from sixdof.simulation import PerfectNavigator, Simulation
from sixdof.math.quaternion import quat_error, quat_rotate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DESCENT_GUIDANCE = [g for g in GUIDANCE if g != "gravity_turn"]

# Nominal wet mass of the small landing vehicle (27 t dry + 10 t prop).
NOMINAL_TOTAL_MASS = 37_000.0

# Chi-square 99% gates used by the EKF/UKF measurement updates.
_CHI2_3DOF = 11.3449
_CHI2_1DOF = 6.6349


def build_sim(guidance_name: str, controller_name: str, nav_kind: str,
              seed: int, dt: float, control_rate_hz: float,
              dispersions: dict | None = None):
    """Construct a landing simulation; returns (sim, vehicle, x0, meta).

    ``dispersions`` are forwarded to ``landing_scenario`` (wind, gusts,
    mass/thrust/density scales, initial pos/vel offsets) and the returned
    ``meta`` is wired into the sensor/actuator models: ``gps_outage_windows``
    and ``meas_delay`` configure the GPS, ``thrust_scale``/``engine_fail_t``
    the throttle actuator, ``sensor_noise_scale`` the sensor sigmas, and
    ``control_noise_sigma`` the actuator command noise.
    """
    veh, atm, wind, x0, meta = landing_scenario(
        np.random.default_rng(seed), **(dispersions or {}))
    sns_scale = float(meta.sensor_noise_scale or 1.0)
    if nav_kind in ("ekf", "ukf"):
        cls = EKF if nav_kind == "ekf" else UKF
        nav = cls(
            x0[0:3] + np.array([3.0, -2.0, 4.0]),
            x0[3:6] + np.array([0.3, -0.2, 0.2]),
            x0[6:10],
            accel_noise_density=100e-6 * 9.80665,
            gyro_noise_density=np.radians(0.005),
            accel_bias_rw=1e-5, gyro_bias_rw=1e-7,
            gps_pos_sigma=tuple(np.array([1.5, 1.5, 3.0]) * sns_scale),
            gps_vel_sigma=0.1 * sns_scale,
        )
        imu = IMU(np.random.default_rng(seed + 1))
        imu.accel_noise *= sns_scale
        imu.gyro_noise *= sns_scale
        sensors = SensorSuite(
            imu=imu,
            gps=GPS(np.random.default_rng(seed + 2),
                    pos_sigma=tuple(np.array([1.5, 1.5, 3.0]) * sns_scale),
                    vel_sigma=0.1 * sns_scale,
                    delay_s=(meta.meas_delay
                             if meta.meas_delay is not None else 0.1),
                    outage_windows=[tuple(w) for w in
                                    (meta.gps_outage_windows or [])]),
            barometer=Barometer(np.random.default_rng(seed + 3), atm,
                                noise_sigma=10.0 * sns_scale,
                                bias_sigma=20.0 * sns_scale),
            radar=RadarAltimeter(np.random.default_rng(seed + 4)),
        )
    else:
        nav = PerfectNavigator()
        sensors = None
    eng = veh.active_stage_config.engine
    act = ActuatorSuite(
        gimbal=GimbalActuator(eng.gimbal_max, eng.gimbal_rate_max),
        throttle=ThrottleActuator(eng.throttle_tau, eng.thrust_min_frac,
                                delay_s=0.05,
                                fail_at_t=meta.engine_fail_t,
                                thrust_scale=meta.thrust_scale),
        rcs=RCSActuator(),
        control_noise_sigma=float(meta.control_noise_sigma or 0.0),
        rng=np.random.default_rng(seed + 5),
    )
    sim = Simulation(
        veh, atm, wind, sensors=sensors, navigator=nav,
        guidance=make_guidance(guidance_name),
        controller=make_controller(controller_name),
        actuators=act, dt=dt, control_rate_hz=control_rate_hz,
        rng=np.random.default_rng(seed + 100),
    )
    return sim, veh, x0, meta


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


def nav_errors(res) -> dict:
    """Navigation error history vs truth: pos [m], vel [m/s], att [deg],
    plus the filter's own 1-sigma bands (error-state ordering
    ``[dr, dv, dtheta, db_a, db_g]``)."""
    xn, pd, x = res.x_nav, res.P_diag, res.x
    mask = np.isfinite(xn[:, 0])
    idx = np.nonzero(mask)[0]
    e_att = np.array([
        quat_error(xn[i, 6:10], x[i, 6:10]) for i in idx
    ]) if idx.size else np.zeros((0, 3))
    sig = np.sqrt(np.maximum(pd[mask], 0.0))
    return {
        "t": res.t[mask],
        "pos_err": xn[mask, 0:3] - x[mask, 0:3],
        "vel_err": xn[mask, 3:6] - x[mask, 3:6],
        "att_err_deg": np.degrees(e_att),
        "pos_sig": sig[:, 0:3],
        "vel_sig": sig[:, 3:6],
        "att_sig_deg": np.degrees(sig[:, 6:9]),
    }


def _shade_outages(ax, outage_windows) -> None:
    for j, w in enumerate(outage_windows or []):
        ax.axvspan(float(w[0]), float(w[1]), color="r", alpha=0.12,
                   label="GPS outage" if j == 0 else None)


def make_nav_error_figure(res, title: str, path: str,
                          outage_windows=None) -> str:
    """Nav pos/vel/att errors vs +-3-sigma bands, outages shaded."""
    ne = nav_errors(res)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                             constrained_layout=True)
    fig.suptitle(title)
    cols = ("tab:blue", "tab:orange", "tab:green")
    rows = [
        ("pos_err", "pos_sig", "position error [m]"),
        ("vel_err", "vel_sig", "velocity error [m/s]"),
        ("att_err_deg", "att_sig_deg", "attitude error [deg]"),
    ]
    for ax, (ek, sk, ylab) in zip(axes, rows):
        err, sig = ne[ek], ne[sk]
        for k, lab in enumerate(("x", "y", "z")):
            ax.plot(ne["t"], err[:, k], lw=0.8, color=cols[k],
                    label=f"err {lab}")
            ax.plot(ne["t"], 3.0 * sig[:, k], ls="--", lw=0.7,
                    color=cols[k], alpha=0.55,
                    label="+-3sigma" if k == 0 else None)
            ax.plot(ne["t"], -3.0 * sig[:, k], ls="--", lw=0.7,
                    color=cols[k], alpha=0.55)
        _shade_outages(ax, outage_windows)
        ax.set_ylabel(ylab)
        ax.legend(ncol=5, fontsize=8)
    axes[-1].set_xlabel("t [s]")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def make_nis_figure(navigator, title: str, path: str,
                    outage_windows=None) -> str | None:
    """Measurement NIS vs time (per update kind) with 99% chi-square gates."""
    log = getattr(navigator, "innovation_log", None) or []
    if not log:
        return None
    series: dict[str, list] = {}
    for entry in log:
        t_u, kind, innov, S = entry
        innov = np.atleast_1d(np.asarray(innov, dtype=float))
        nis = float(innov @ np.linalg.solve(np.asarray(S, dtype=float), innov))
        series.setdefault(str(kind), []).append((float(t_u), nis))

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    fig.suptitle(title)
    gates_drawn = set()
    for kind, rows in sorted(series.items()):
        tt = [r[0] for r in rows]
        nn = [r[1] for r in rows]
        gate = _CHI2_3DOF if kind.startswith("gps") else _CHI2_1DOF
        n_over = int(np.sum(np.asarray(nn) > gate))
        ax.scatter(tt, nn, s=5, alpha=0.6,
                   label=f"{kind} ({n_over}/{len(nn)} gated)")
        dof = "3" if kind.startswith("gps") else "1"
        if dof not in gates_drawn:
            ax.axhline(gate, color="r", ls="--", lw=0.9,
                       label=f"99% gate, {dof}-dof")
            gates_drawn.add(dof)
    _shade_outages(ax, outage_windows)
    ax.set_yscale("log")
    ax.set_xlabel("t [s]"); ax.set_ylabel("NIS")
    ax.legend(fontsize=8)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Dispersed Monte-Carlo comparison helpers (shared by compare_*.py)
# ---------------------------------------------------------------------------

def dispersion_draws(seed: int, seed0: int = 100) -> dict:
    """Per-seed landing dispersion set for the comparison sweeps.

    wind ~ U(0,12) m/s + 1.5 m/s gusts, density +-10%, thrust +-3%,
    mass +-3% of the 37 t nominal wet mass, initial pos +-50 m and
    vel +-5 m/s per axis, GPS transport delay 0.1 s, and a 10 s GPS
    outage window mid-descent on ~50% of seeds (even sweep indices).
    """
    rng = np.random.default_rng(seed)
    d = {
        "wind_speed": float(rng.uniform(0.0, 12.0)),
        "wind_dir": float(rng.uniform(0.0, 2.0 * np.pi)),
        "gust_sigma": 1.5,
        "density_scale": float(rng.uniform(-0.10, 0.10)),
        "thrust_scale": float(rng.uniform(0.97, 1.03)),
        "mass_offset": float(rng.uniform(-0.03, 0.03) * NOMINAL_TOTAL_MASS),
        "pos_offset": [float(v) for v in rng.uniform(-50.0, 50.0, 3)],
        "vel_offset": [float(v) for v in rng.uniform(-5.0, 5.0, 3)],
        "meas_delay": 0.1,
    }
    if (seed - seed0) % 2 == 0:
        t0 = float(rng.uniform(10.0, 18.0))  # mid-descent (nominal TD ~32 s)
        d["gps_outage_windows"] = [[t0, t0 + 10.0]]
    return d


def run_trial(job: dict) -> dict:
    """Worker: one closed-loop landing run -> a JSON-safe per-run record.

    Never raises: sim exceptions are caught and recorded as
    ``failure_mode="exception"`` so one bad seed cannot kill a sweep.
    ``job`` keys: guidance, controller, nav, seed, dt, control_hz, t_end,
    dispersions (dict or None), kind ("sweep"/"nominal"), return_trace.
    """
    import traceback

    rec = {
        "kind": job.get("kind", "sweep"),
        "guidance": job["guidance"], "controller": job["controller"],
        "nav": job["nav"], "seed": int(job["seed"]),
        "dispersions": job.get("dispersions") or {},
    }
    t0 = time.perf_counter()
    try:
        sim, veh, x0, meta = build_sim(
            job["guidance"], job["controller"], job["nav"], job["seed"],
            job.get("dt", 0.01), job.get("control_hz", 50.0),
            dispersions=job.get("dispersions"))
        rec["initial_pos"] = [float(v) for v in x0[0:3]]
        rec["initial_vel"] = [float(v) for v in x0[3:6]]
        rec["initial_lateral_offset"] = float(np.hypot(x0[0], x0[1]))
        res = sim.run(x0, t_end=job.get("t_end", 90.0),
                      stop_on_touchdown=True)
        ok, mode = landing_success(res.touchdown, veh.prop_remaining)
        rec.update(res.summary(landing_success_fn=landing_success,
                               prop_remaining=veh.prop_remaining))
        rec["success"], rec["failure_mode"] = bool(ok), str(mode)
        rec["prop_remaining_kg"] = float(veh.prop_remaining)
        ct = res.compute_time
        if ct is not None:
            ct = np.asarray(ct, dtype=float)
            ct = ct[np.isfinite(ct)]
            if ct.size:
                rec["compute_time_p95_ms"] = float(np.percentile(ct * 1e3, 95))
        rec["events"] = [[float(te), str(e)] for te, e in res.events]
        if job.get("return_trace"):
            step = max(1, len(res.t) // 2000)
            rec["trace"] = {
                "t": [float(v) for v in res.t[::step]],
                "tilt_deg": [float(v) for v in tilt_history(res)[::step]],
                "lateral_offset": [float(v) for v in
                                   np.hypot(res.x[:, 0], res.x[:, 1])[::step]],
            }
    except Exception as exc:  # noqa: BLE001 - record and move on
        rec["success"] = False
        rec["failure_mode"] = "exception"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()[-2000:]
    rec["wall_time_s"] = float(time.perf_counter() - t0)
    return _jsonable(rec)


def _jsonable(obj):
    """Recursively convert numpy types/arrays; non-finite floats -> None."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _finite(vals) -> np.ndarray:
    out = []
    for v in vals:
        if v is None:
            continue
        v = float(v)
        if np.isfinite(v):
            out.append(v)
    return np.asarray(out, dtype=float)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    h = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def summarize_runs(records: list) -> dict:
    """Aggregate per-run records into the comparison statistics block."""
    n = len(records)
    k = sum(1 for r in records if r.get("success"))

    def pct(key, q):
        v = _finite(r.get(key) for r in records)
        return float(np.percentile(v, q)) if v.size else None

    def mean_of(key):
        v = _finite(r.get(key) for r in records)
        return float(np.mean(v)) if v.size else None

    fuel = _finite(r.get("fuel_used_kg") for r in records)
    modes: dict = {}
    for r in records:
        m = str(r.get("failure_mode") or "unknown")
        modes[m] = modes.get(m, 0) + 1

    return {
        "n": n,
        "successes": k,
        "success_rate": k / n if n else 0.0,
        "wilson95": [wilson_ci(k, n)[0], wilson_ci(k, n)[1]],
        "vs_p50": pct("vertical_speed", 50),
        "vs_p95": pct("vertical_speed", 95),
        "lateral_speed_p95": pct("lateral_speed", 95),
        "tilt_p95": pct("tilt_angle_deg", 95),
        "offset_p50": pct("lateral_offset", 50),
        "offset_p95": pct("lateral_offset", 95),
        "fuel_used_mean": float(np.mean(fuel)) if fuel.size else None,
        "fuel_used_std": float(np.std(fuel)) if fuel.size else None,
        "control_effort_mean": mean_of("control_effort"),
        "compute_time_ms_mean": mean_of("compute_time_mean_ms"),
        "compute_time_ms_p95": pct("compute_time_p95_ms", 95),
        "failure_modes": modes,
        "runs": records,
    }


def make_comparison_figure(table: dict, names: list, title: str,
                           path: str) -> str:
    """4-panel dispersed-comparison figure shared by both compare scripts:
    success-rate bars w/ Wilson CI | touchdown |v_z| + offset boxplots |
    fuel-used boxplots | compute-time boxplots."""
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    fig.suptitle(title)
    gs = fig.add_gridspec(2, 2)
    colors = plt.get_cmap("tab10")

    # (a) success rate bars + Wilson 95% CI whiskers
    axa = fig.add_subplot(gs[0, 0])
    rates = np.array([table[c]["success_rate"] for c in names])
    lo = np.array([max(0.0, r - table[c]["wilson95"][0])
                   for r, c in zip(rates, names)])
    hi = np.array([max(0.0, table[c]["wilson95"][1] - r)
                   for r, c in zip(rates, names)])
    axa.bar(names, rates, yerr=np.vstack([lo, hi]), capsize=4,
            color=[colors(i) for i in range(len(names))], alpha=0.8,
            edgecolor="k")
    for i, r in enumerate(rates):
        axa.text(i, min(r + 0.06, 1.0), f"{r:.0%}",
                 ha="center", fontsize=9)
    axa.set_ylim(0, 1.12)
    axa.set_ylabel("success rate")
    axa.set_title("landing success (Wilson 95% CI)")
    axa.grid(axis="y", alpha=0.3)

    def _series(key):
        out = []
        for c in names:
            v = _finite(r.get(key) for r in table[c]["runs"])
            out.append(v if v.size else np.array([np.nan]))
        return out

    def _box(ax, data, ylab, hline=None, log=False):
        bp = ax.boxplot(data, tick_labels=names, showfliers=True,
                        patch_artist=True)
        for i, b in enumerate(bp["boxes"]):
            b.set_facecolor(colors(i)); b.set_alpha(0.6)
        if hline is not None:
            ax.axhline(hline, color="r", ls="--", lw=0.8)
        if log:
            ax.set_yscale("log")
        ax.set_ylabel(ylab)
        ax.grid(axis="y", alpha=0.3)
        for tick in ax.get_xticklabels():
            tick.set_rotation(20)

    # (b) touchdown |v_z| + lateral offset boxplots (split panel)
    gs2 = gs[0, 1].subgridspec(1, 2)
    axb1 = fig.add_subplot(gs2[0, 0])
    _box(axb1, _series("vertical_speed"), "|v_z| [m/s]", hline=3.0)
    axb1.set_title("touchdown |v_z|")
    axb2 = fig.add_subplot(gs2[0, 1], sharey=None)
    _box(axb2, _series("lateral_offset"), "offset [m]", hline=10.0,
         log=True)
    axb2.set_title("touchdown offset")

    # (c) fuel used
    axc = fig.add_subplot(gs[1, 0])
    _box(axc, _series("fuel_used_kg"), "fuel used [kg]")
    axc.set_title("propellant used")

    # (d) per-run mean controller compute time
    axd = fig.add_subplot(gs[1, 1])
    _box(axd, _series("compute_time_mean_ms"), "compute time [ms]",
         log=True)
    axd.set_title("controller compute time (per-run mean)")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--guidance", choices=DESCENT_GUIDANCE, default="zemzev")
    p.add_argument("--controller", choices=list(CONTROLLERS), default="pid")
    p.add_argument("--nav", choices=["perfect", "ekf", "ukf"], default="perfect")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--control-hz", type=float, default=50.0)
    p.add_argument("--t-end", type=float, default=90.0)
    # Dispersions (forwarded to landing_scenario / wired into sensors+actuators)
    p.add_argument("--pos-offset", type=float, nargs=3,
                   metavar=("DX", "DY", "DZ"), default=None,
                   help="initial position offset [m]")
    p.add_argument("--vel-offset", type=float, nargs=3,
                   metavar=("DVX", "DVY", "DVZ"), default=None,
                   help="initial velocity offset [m/s]")
    p.add_argument("--mass-offset", type=float, default=0.0,
                   help="initial mass offset [kg]")
    p.add_argument("--thrust-scale", type=float, default=1.0)
    p.add_argument("--density-scale", type=float, default=0.0,
                   help="fractional air-density offset")
    p.add_argument("--wind-speed", type=float, default=0.0)
    p.add_argument("--wind-dir", type=float, default=None,
                   help="steady-wind direction [deg]; drawn randomly if unset")
    p.add_argument("--gust-sigma", type=float, default=0.0,
                   help="Dryden gust sigma [m/s]")
    p.add_argument("--gps-outage", type=float, nargs=2,
                   metavar=("T_START", "T_END"), default=None,
                   help="GPS outage window [s]")
    p.add_argument("--meas-delay", type=float, default=None,
                   help="GPS measurement transport delay [s]")
    p.add_argument("--sensor-noise-scale", type=float, default=1.0)
    p.add_argument("--control-noise", type=float, default=0.0,
                   help="sigma [rad/throttle-frac] noise on actuator commands")
    p.add_argument("--engine-fail-t", type=float, default=None,
                   help="time [s] at which the engine permanently fails")
    p.add_argument("--tag", type=str, default=None,
                   help="artifact filename suffix (default: <g>_<c>_<nav>)")
    p.add_argument("--outdir", type=str, default=os.path.join(ROOT, "results"))
    p.add_argument("--figdir", type=str,
                   default=os.path.join(ROOT, "docs", "figures"))
    p.add_argument("--no-figs", action="store_true")
    return p.parse_args(argv)


def dispersions_from_args(args) -> dict:
    """Collect the CLI dispersion flags into ``landing_scenario`` kwargs."""
    d = {
        "wind_speed": args.wind_speed,
        "gust_sigma": args.gust_sigma,
        "mass_offset": args.mass_offset,
        "thrust_scale": args.thrust_scale,
        "density_scale": args.density_scale,
        "sensor_noise_scale": args.sensor_noise_scale,
        "control_noise_sigma": args.control_noise,
    }
    if args.pos_offset is not None:
        d["pos_offset"] = list(args.pos_offset)
    if args.vel_offset is not None:
        d["vel_offset"] = list(args.vel_offset)
    if args.wind_dir is not None:
        d["wind_dir"] = np.radians(args.wind_dir)
    if args.gps_outage is not None:
        d["gps_outage_windows"] = [tuple(args.gps_outage)]
    if args.meas_delay is not None:
        d["meas_delay"] = args.meas_delay
    if args.engine_fail_t is not None:
        d["engine_fail_t"] = args.engine_fail_t
    return d


def main(argv=None):
    args = parse_args(argv)
    tag = args.tag or f"{args.guidance}_{args.controller}_{args.nav}"

    sim, veh, x0, meta = build_sim(
        args.guidance, args.controller, args.nav,
        args.seed, args.dt, args.control_hz,
        dispersions=dispersions_from_args(args))
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
    nav_fig_paths = []
    if not args.no_figs:
        fig_path = make_figure(
            res, f"landing: {args.guidance} / {args.controller} / {args.nav}",
            os.path.join(args.figdir, f"landing_{tag}.png"))
        if args.nav in ("ekf", "ukf"):
            outage = getattr(meta, "gps_outage_windows", None) or []
            nav_fig_paths.append(make_nav_error_figure(
                res,
                f"nav errors: {args.guidance} / {args.controller} / {args.nav}",
                os.path.join(args.figdir, f"landing_{tag}_nav_errors.png"),
                outage_windows=outage))
            nis_path = make_nis_figure(
                sim.navigator,
                f"measurement NIS: {args.guidance} / {args.controller} / {args.nav}",
                os.path.join(args.figdir, f"landing_{tag}_nis.png"),
                outage_windows=outage)
            if nis_path:
                nav_fig_paths.append(nis_path)

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
    for p_ in nav_fig_paths:
        print(f"figure: {p_}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
