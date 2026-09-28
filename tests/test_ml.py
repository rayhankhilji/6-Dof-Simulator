"""Tests for the ML success-prediction component (sixdof.ml)."""

from __future__ import annotations

import os

import joblib
import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, roc_auc_score

from sixdof.batch.batch_mc import GATE_ALTITUDES
from sixdof.ml.features import (
    OBSERVABLE_FEATURES,
    ORACLE_EXTRA,
    build_features,
)
from sixdof.ml.train import stratified_split, train_success_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ_CANDIDATES = [
    os.path.join(ROOT, "results", "monte_carlo_ml.npz"),
    os.path.join(ROOT, "results", "monte_carlo.npz"),
]


def _synthetic_npz(n=2000, seed=0):
    """Minimal mapping mimicking the MC npz keys the feature builder needs."""
    rng = np.random.default_rng(seed)
    gs = np.full((n, len(GATE_ALTITUDES), 7), np.nan)
    gt = np.full((n, len(GATE_ALTITUDES)), np.nan)
    gi = int(np.argmin(np.abs(GATE_ALTITUDES - 1500.0)))
    gs[:, gi, 0:3] = np.column_stack([
        rng.normal(-100, 60, n), rng.normal(50, 60, n),
        np.full(n, GATE_ALTITUDES[gi])])
    vz = -rng.uniform(40, 160, n)
    gs[:, gi, 3:6] = np.column_stack([
        rng.normal(0, 15, n), rng.normal(0, 15, n), vz])
    gs[:, gi, 6] = rng.uniform(31000, 36000, n)
    gt[:, gi] = rng.uniform(6, 12, n)
    # Logistic boundary: success iff |vz| small and offset small.
    logit = (6.0 - 0.08 * np.abs(vz)
             - 0.03 * np.hypot(gs[:, gi, 0], gs[:, gi, 1]))
    success = rng.random(n) < 1.0 / (1.0 + np.exp(-logit))
    return {
        "success": success,
        "gate_altitudes": GATE_ALTITUDES.copy(),
        "gate_states": gs,
        "gate_times": gt,
        "wind_speed": rng.uniform(0, 12, n),
        "wind_dir": rng.uniform(0, 2 * np.pi, n),
        "gust_sigma": rng.uniform(0, 2, n),
        "mass_scale": rng.normal(1, 0.03, n),
        "thrust_scale": rng.normal(1, 0.03, n),
        "density_scale": np.exp(0.1 * rng.standard_normal(n)),
        "meas_delay_tau": rng.uniform(0.1, 0.3, n),
        "gps_outage": rng.random(n) < 0.2,
        "engine_fail": rng.random(n) < 0.02,
        "engine_fail_t": rng.uniform(5, 40, n),
    }


def _synthetic_xy(n=3000, k=None, seed=1):
    """Feature matrix with a logistic decision boundary."""
    rng = np.random.default_rng(seed)
    names = OBSERVABLE_FEATURES if k is None else [f"f{i}" for i in range(k)]
    k = len(names)
    X = rng.standard_normal((n, k))
    w = np.zeros(k)
    w[:4] = [2.5, -2.0, 1.2, 1.0]
    logit = X @ w + 0.2 * rng.standard_normal(n)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    meta = {"gate_alt": 1500.0, "feature_set": "observable",
            "n_used": n, "n_dropped": 0, "success_rate": float(y.mean())}
    return X, y, names, meta


# ---------------------------------------------------------------------------

def test_feature_builder_shapes_and_names():
    data = _synthetic_npz()
    X, y, names, meta = build_features(data, 1500.0, "observable")
    assert X.shape == (2000, len(OBSERVABLE_FEATURES))
    assert names == OBSERVABLE_FEATURES
    assert len(set(names)) == len(names)
    assert np.isfinite(X).all()
    assert y.shape == (2000,)
    assert meta["gate_alt"] == 1500.0
    assert meta["feature_set"] == "observable"
    assert meta["n_used"] == 2000
    assert meta["n_dropped"] == 0


def test_feature_builder_oracle_extra_and_gate_select():
    data = _synthetic_npz()
    Xo, _, names_o, meta_o = build_features(data, 1500.0, "oracle")
    assert Xo.shape[1] == len(OBSERVABLE_FEATURES) + len(ORACLE_EXTRA)
    assert names_o[-len(ORACLE_EXTRA):] == ORACLE_EXTRA
    assert meta_o["feature_set"] == "oracle"
    # Unknown gate -> error; missing gate_states -> error.
    with pytest.raises(ValueError):
        build_features(data, 12345.0, "observable")
    with pytest.raises(ValueError):
        build_features({"success": np.ones(4), "gate_altitudes": GATE_ALTITUDES},
                       1500.0, "observable")


def test_feature_builder_drops_uncrossed():
    data = _synthetic_npz()
    data["gate_times"][:37, 1] = np.nan   # 37 runs never reached 1500 m
    X, y, _, meta = build_features(data, 1500.0, "observable")
    assert X.shape[0] == 2000 - 37
    assert meta["n_dropped"] == 37


def test_train_high_auc_on_logistic_boundary():
    X, y, names, meta = _synthetic_xy()
    res = train_success_model(X, y, names, meta, seed=0)
    auc = res["metrics"]["test_calibrated"]["roc_auc"]
    assert auc > 0.9
    assert res["metrics"]["winner"] in ("hgb", "hgb_balanced", "logreg")
    assert len(res["metrics"]["permutation_importance"]) == 20


def test_calibration_not_worse_than_uncalibrated_on_val():
    X, y, names, meta = _synthetic_xy(n=4000, seed=2)
    splits = stratified_split(y, seed=0)
    i_va = splits["val"]
    res = train_success_model(X, y, names, meta, seed=0)
    cal = res["bundle"]["model"]
    p_cal = cal.predict_proba(X[i_va])[:, 1]
    # Refit the winner standalone for an uncalibrated comparison on val.
    from sixdof.ml.train import _candidates
    est = _candidates(0)[res["metrics"]["winner"]]
    est.fit(X[splits["train"]], y[splits["train"]])
    p_raw = est.predict_proba(X[i_va])[:, 1]
    b_cal = brier_score_loss(y[i_va], p_cal)
    b_raw = brier_score_loss(y[i_va], p_raw)
    assert b_cal <= b_raw + 1e-6


def test_joblib_round_trip(tmp_path):
    X, y, names, meta = _synthetic_xy(n=1500, seed=3)
    res = train_success_model(X, y, names, meta, seed=0)
    path = tmp_path / "bundle.pkl"
    joblib.dump(res["bundle"], path)
    b2 = joblib.load(path)
    assert b2["feature_names"] == res["bundle"]["feature_names"]
    p1 = res["bundle"]["model"].predict_proba(X[:50])
    p2 = b2["model"].predict_proba(X[:50])
    np.testing.assert_allclose(p1, p2)


def test_smoke_on_monte_carlo_npz():
    npz = next((p for p in NPZ_CANDIDATES if os.path.exists(p)), None)
    if npz is None:
        pytest.skip("no Monte Carlo npz present")
    try:
        X, y, names, meta = build_features(npz, 1500.0, "observable")
    except ValueError as exc:
        pytest.skip(f"npz lacks gate states: {exc}")
    rng = np.random.default_rng(0)
    sub = rng.choice(len(y), size=min(4000, len(y)), replace=False)
    res = train_success_model(X[sub], y[sub], names,
                              {**meta, "raw": None}, seed=0)
    auc = res["metrics"]["test_calibrated"]["roc_auc"]
    assert auc > 0.6, f"smoke AUC too low: {auc}"
