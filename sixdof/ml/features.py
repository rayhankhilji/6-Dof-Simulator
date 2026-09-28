"""Feature engineering for the gate-state landing-success model.

``build_features`` extracts the per-run state recorded by the batch Monte
Carlo at the crossing of a chosen altitude gate and turns it into a feature
matrix for success classification.

Two feature sets are supported:

* ``observable`` -- quantities an onboard navigation filter could plausibly
  know: the (measured-ish) gate state ``r, v``, vehicle mass, a baro-derived
  nominal density, and kinematic/dynamic proxies built from them.  True
  dispersion draws (wind, thrust scale, ...) are excluded.
* ``oracle`` -- the observable set plus the per-run dispersion draws, giving
  an upper bound on what perfect knowledge of the environment would buy.

Requires an npz (or ``MCResult``) produced after the gate recorder was
extended to write ``gate_states`` (n, 7, 7) = ``[r_I, v_I, m]`` and
``gate_times`` (n, 7); runs that never crossed the gate (NaN) are dropped.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

import numpy as np

from ..environment.atmosphere import USStandardAtmosphere1976

# ---------------------------------------------------------------------------
# Constants (mirror small_landing_vehicle / batch_mc)
# ---------------------------------------------------------------------------

T_MAX_VAC = 845e3          # single-stage vacuum thrust [N]
DRY_MASS_NOM = 27_000.0    # nominal dry mass [kg]
G0 = 9.80665               # standard gravity [m/s^2]

OBSERVABLE_FEATURES = [
    "r_x", "r_y", "z",
    "v_x", "v_y", "v_z",
    "speed", "lateral_speed", "abs_vz",
    "lateral_offset", "t_gate",
    "flight_path_sin",          # -v_z / |v|  (sin of flight-path angle)
    "required_decel",           # v_z^2 / (2 z)  [m/s^2]
    "avail_decel",              # T_max/m - g    [m/s^2]
    "decel_margin",             # avail_decel - required_decel
    "dyn_pressure",             # 0.5 rho_nom(z) |v|^2  [Pa]
    "energy",                   # v^2/2 + g z   [m^2/s^2]
    "t_go_est",                 # z / |v_z|     [s]
    "mass",
    "prop_proxy",               # m - dry_nom
]

ORACLE_EXTRA = [
    "wind_speed", "wind_dir_sin", "wind_dir_cos", "gust_sigma",
    "mass_scale", "thrust_scale", "density_scale", "meas_delay_tau",
    "gps_outage", "engine_fail", "engine_fail_t",
]

# npz keys consumed by the oracle columns (wind_dir -> sin/cos derived).
_ORACLE_KEYS = (
    "wind_speed", "wind_dir", "gust_sigma", "mass_scale", "thrust_scale",
    "density_scale", "meas_delay_tau", "gps_outage", "engine_fail",
    "engine_fail_t",
)

_NOMINAL_ATM = USStandardAtmosphere1976(1.0)


# ---------------------------------------------------------------------------
# Column computation
# ---------------------------------------------------------------------------

def compute_feature_columns(r, v, m, t_gate, dispersions=None):
    """Return ``{name: (n,) column}`` for all computable features.

    Parameters
    ----------
    r, v : (n, 3) gate-crossing position/velocity (inertial frame, z = alt).
    m : (n,) vehicle mass at crossing [kg].
    t_gate : (n,) time since simulation start at crossing [s].
    dispersions : dict of (n,) arrays or None.  Oracle columns are only
        populated when the corresponding key is present; missing keys are
        omitted (callers select the columns they need).
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    m = np.asarray(m, dtype=float)
    t_gate = np.asarray(t_gate, dtype=float)

    z = r[:, 2]
    vz = v[:, 2]
    speed = np.sqrt(np.einsum("ij,ij->i", v, v))
    lat_speed = np.hypot(v[:, 0], v[:, 1])
    abs_vz = np.abs(vz)
    z_safe = np.maximum(z, 1.0)

    rho = _NOMINAL_ATM.density(z)          # nominal (baro-style) density
    required = vz * vz / (2.0 * z_safe)
    avail = T_MAX_VAC / np.maximum(m, 1.0) - G0

    cols = {
        "r_x": r[:, 0],
        "r_y": r[:, 1],
        "z": z,
        "v_x": v[:, 0],
        "v_y": v[:, 1],
        "v_z": vz,
        "speed": speed,
        "lateral_speed": lat_speed,
        "abs_vz": abs_vz,
        "lateral_offset": np.hypot(r[:, 0], r[:, 1]),
        "t_gate": t_gate,
        "flight_path_sin": np.where(speed > 1e-6, -vz / np.maximum(speed, 1e-6), 1.0),
        "required_decel": required,
        "avail_decel": avail,
        "decel_margin": avail - required,
        "dyn_pressure": 0.5 * rho * speed * speed,
        "energy": 0.5 * speed * speed + G0 * z,
        "t_go_est": z_safe / np.maximum(abs_vz, 0.5),
        "mass": m,
        "prop_proxy": m - DRY_MASS_NOM,
    }

    if dispersions:
        wd = np.asarray(dispersions.get("wind_dir", np.zeros_like(z)),
                        dtype=float)
        oracle_map = {
            "wind_speed": dispersions.get("wind_speed"),
            "wind_dir_sin": np.sin(wd),
            "wind_dir_cos": np.cos(wd),
            "gust_sigma": dispersions.get("gust_sigma"),
            "mass_scale": dispersions.get("mass_scale"),
            "thrust_scale": dispersions.get("thrust_scale"),
            "density_scale": dispersions.get("density_scale"),
            "meas_delay_tau": dispersions.get("meas_delay_tau"),
            "gps_outage": dispersions.get("gps_outage"),
            "engine_fail": dispersions.get("engine_fail"),
            "engine_fail_t": dispersions.get("engine_fail_t"),
        }
        for name, col in oracle_map.items():
            if col is None:
                continue
            col = np.asarray(col, dtype=float)
            cols[name] = col
    return cols


def feature_names_for(feature_set: str) -> list[str]:
    """Ordered feature names for ``feature_set`` ('observable' | 'oracle')."""
    if feature_set == "observable":
        return list(OBSERVABLE_FEATURES)
    if feature_set == "oracle":
        return list(OBSERVABLE_FEATURES) + list(ORACLE_EXTRA)
    raise ValueError(f"unknown feature_set {feature_set!r}")


# ---------------------------------------------------------------------------
# Source loading / gate extraction
# ---------------------------------------------------------------------------

def _as_mapping(source):
    """Accept an npz path, an NpzFile/dict, or an MCResult-like object."""
    if isinstance(source, (str, os.PathLike)):
        return np.load(source, allow_pickle=True)
    if isinstance(source, Mapping) or hasattr(source, "files"):
        return source
    # MCResult-like: build a view over dataclass fields.
    keys = ("success", "gate_altitudes", "gate_states", "gate_times",
            "gate_positions") + _ORACLE_KEYS
    out = {}
    for k in keys:
        if hasattr(source, k):
            out[k] = getattr(source, k)
    if out:
        return out
    raise TypeError(f"cannot interpret feature source of type {type(source)!r}")


def gate_state_arrays(source, gate_alt):
    """Extract (r, v, m, t, valid_mask, dispersions, gate_index) at a gate.

    ``valid_mask`` marks runs that actually crossed the gate (finite state
    and time).  Raises ``ValueError`` if the source lacks ``gate_states``.
    """
    data = _as_mapping(source)
    if "gate_states" not in data or "gate_times" not in data:
        raise ValueError(
            "source has no gate_states/gate_times; regenerate the Monte "
            "Carlo npz with the extended gate recorder in batch_mc.py")

    galt = np.asarray(data["gate_altitudes"], dtype=float)
    gi = int(np.argmin(np.abs(galt - float(gate_alt))))
    if abs(galt[gi] - float(gate_alt)) > 1.0:
        raise ValueError(
            f"requested gate {gate_alt} not in gate_altitudes {galt.tolist()}")

    gs = np.asarray(data["gate_states"], dtype=float)   # (n, 7, 7)
    gt = np.asarray(data["gate_times"], dtype=float)    # (n, 7)
    valid = np.isfinite(gt[:, gi]) & np.isfinite(gs[:, gi, :]).all(axis=1)

    r = gs[valid, gi, 0:3]
    v = gs[valid, gi, 3:6]
    m = gs[valid, gi, 6]
    t = gt[valid, gi]

    dispersions = {}
    for k in _ORACLE_KEYS:
        if k in data:
            dispersions[k] = np.asarray(data[k])[valid]
    return r, v, m, t, valid, dispersions, gi


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features(npz_path_or_result, gate_alt, feature_set="observable"):
    """Build ``(X, y, feature_names, meta)`` for success prediction.

    Parameters
    ----------
    npz_path_or_result : path to a Monte Carlo npz, an NpzFile/mapping, or
        an ``MCResult``.  Must contain ``gate_states``, ``gate_times``,
        ``gate_altitudes``, ``success`` (and the dispersion draws for the
        ``oracle`` feature set).
    gate_alt : float
        Altitude gate [m]; the nearest entry in ``gate_altitudes`` is used
        (first crossing of that altitude, NaN for runs that never got there).
    feature_set : "observable" | "oracle"

    Returns
    -------
    X : (n_used, k) float64 feature matrix (finite).
    y : (n_used,) int labels (1 = success).
    feature_names : list[str] of length k.
    meta : dict with gate/feature-set bookkeeping.  ``meta["raw"]`` holds the
        sliced gate arrays (r, v, m, t, dispersions) for downstream figure
        generation (e.g. the probability map); drop it before JSON dumps.
    """
    data = _as_mapping(npz_path_or_result)
    if "success" not in data:
        raise ValueError("source lacks 'success' labels")

    r, v, m, t, valid, dispersions, gi = gate_state_arrays(data, gate_alt)
    galt = float(np.asarray(data["gate_altitudes"], dtype=float)[gi])

    names = feature_names_for(feature_set)
    if feature_set == "oracle":
        missing = [k for k in _ORACLE_KEYS if k not in dispersions]
        if missing:
            raise ValueError(
                f"oracle feature set requested but dispersion keys missing: "
                f"{missing}")

    cols = compute_feature_columns(r, v, m, t, dispersions)
    X = np.column_stack([cols[name] for name in names]).astype(np.float64)
    y = np.asarray(data["success"])[valid].astype(np.int64)

    meta = {
        "gate_alt": galt,
        "gate_index": gi,
        "feature_set": feature_set,
        "n_total": int(np.asarray(data["success"]).shape[0]),
        "n_used": int(valid.sum()),
        "n_dropped": int((~valid).sum()),
        "success_rate": float(y.mean()) if y.size else float("nan"),
        "raw": {"r": r, "v": v, "m": m, "t": t, "dispersions": dispersions},
    }
    return X, y, names, meta
