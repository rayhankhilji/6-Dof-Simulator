"""Training/evaluation for the gate-state landing-success model.

Pipeline: stratified 70/15/15 split -> candidate models
(HistGradientBoosting, plain and class-balanced; LogisticRegression on
standardized features as a linear baseline) -> winner selected by
validation PR-AUC (the honest metric under the ~8 % success base rate) ->
isotonic recalibration on the validation fold -> test-set metrics,
permutation importance and a P(success) map over altitude x |v_z|.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import compute_feature_columns

THRESHOLDS = (0.5, 0.8, 0.95)


# ---------------------------------------------------------------------------
# Splitting / metrics
# ---------------------------------------------------------------------------

def stratified_split(y, seed=0):
    """70/15/15 stratified train/val/test index split."""
    n = len(y)
    idx = np.arange(n)
    i_tr, i_tmp = train_test_split(
        idx, test_size=0.30, random_state=seed, stratify=y)
    i_va, i_te = train_test_split(
        i_tmp, test_size=0.50, random_state=seed, stratify=y[i_tmp])
    return {"train": i_tr, "val": i_va, "test": i_te}


def evaluate_probs(prob, y, thresholds=THRESHOLDS):
    """Scalar metrics + per-threshold confusion for probability column."""
    prob = np.asarray(prob, dtype=float)
    y = np.asarray(y, dtype=int)
    out = {
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
        "thresholds": {},
    }
    for th in thresholds:
        pred = prob >= th
        tn, fp, fn, tp = confusion_matrix(
            y, pred, labels=[0, 1]).ravel()
        out["thresholds"][f"{th:g}"] = {
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "precision": float(tp / max(tp + fp, 1)),
            "recall": float(tp / max(tp + fn, 1)),
            "flagged_frac": float(pred.mean()),
        }
    return out


def decile_table(prob, y, n_bins=10):
    """Observed success rate vs predicted-prob decile (quantile bins)."""
    prob = np.asarray(prob, dtype=float)
    y = np.asarray(y, dtype=int)
    edges = np.quantile(prob, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    b = np.clip(np.searchsorted(edges, prob, side="right") - 1, 0, n_bins - 1)
    rows = []
    for k in range(n_bins):
        sel = b == k
        if not sel.any():
            continue
        rows.append({
            "bin": int(k),
            "count": int(sel.sum()),
            "mean_pred": float(prob[sel].mean()),
            "observed": float(y[sel].mean()),
            "lo": float(edges[k]) if np.isfinite(edges[k]) else float(prob[sel].min()),
            "hi": float(edges[k + 1]) if np.isfinite(edges[k + 1]) else float(prob[sel].max()),
        })
    return rows


# ---------------------------------------------------------------------------
# Candidate models
# ---------------------------------------------------------------------------

def _candidates(seed):
    """name -> (estimator, needs_scaling_flag_for_meta)."""
    return {
        "hgb": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=40, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=20,
            random_state=seed),
        "hgb_balanced": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=40, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=20,
            class_weight="balanced",
            random_state=seed),
        "logreg": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
        ]),
    }


# ---------------------------------------------------------------------------
# Probability map over altitude x |v_z|
# ---------------------------------------------------------------------------

def probability_map(predict_proba, raw, feature_names, feature_set,
                    z_grid, vz_grid):
    """P(success) on a (z, |v_z|) grid, other state vars at medians.

    ``raw`` is ``meta["raw"]`` from ``build_features`` (the sliced gate
    arrays); medians anchor the state while z / |v_z| sweep the grid.
    """
    r_med = np.median(raw["r"], axis=0)
    v_med = np.median(raw["v"], axis=0)
    m_med = float(np.median(raw["m"]))
    t_med = float(np.median(raw["t"]))
    d_med = {k: np.full(z_grid.size * vz_grid.size, np.median(val))
             for k, val in raw.get("dispersions", {}).items()}

    zz, vv = np.meshgrid(z_grid, vz_grid, indexing="ij")
    n_g = zz.size
    r = np.tile(r_med, (n_g, 1))
    v = np.tile(v_med, (n_g, 1))
    r[:, 2] = zz.ravel()
    v[:, 2] = -np.abs(vv.ravel())
    m = np.full(n_g, m_med)
    t = np.full(n_g, t_med)

    cols = compute_feature_columns(r, v, m, t, d_med)
    X = np.column_stack([cols[name] for name in feature_names])
    out = np.asarray(predict_proba(X))
    prob = out[:, 1] if out.ndim == 2 else out
    return zz, vv, prob.reshape(zz.shape)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train_success_model(X, y, feature_names, meta, seed=0):
    """Train, select, calibrate and evaluate the success classifier.

    Returns a result dict:
      ``bundle``  -- joblib-ready {model, scaler, feature_names, meta};
                     ``model.predict_proba(X_raw)`` takes unscaled features.
      ``metrics`` -- JSON-ready metrics block for this combo.
      ``curves``  -- arrays for the figure helpers (ROC/PR/calibration/etc.).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    classes = np.unique(y)
    if classes.size < 2:
        raise ValueError("need both success and failure rows to train")

    splits = stratified_split(y, seed)
    i_tr, i_va, i_te = splits["train"], splits["val"], splits["test"]
    Xtr, Xva, Xte = X[i_tr], X[i_va], X[i_te]
    ytr, yva, yte = y[i_tr], y[i_va], y[i_te]

    scaler = StandardScaler().fit(Xtr)  # saved for reference / LR path

    cand = _candidates(seed)
    fitted, val_metrics = {}, {}
    for name, est in cand.items():
        est.fit(Xtr, ytr)
        fitted[name] = est
        pv = est.predict_proba(Xva)[:, 1]
        val_metrics[name] = evaluate_probs(pv, yva)

    # Winner by validation PR-AUC (majority-failure regime).
    winner_name = max(val_metrics, key=lambda k: val_metrics[k]["pr_auc"])
    winner = fitted[winner_name]

    # Isotonic recalibration on the validation fold (prefit model frozen).
    cal = CalibratedClassifierCV(FrozenEstimator(winner), method="isotonic")
    cal.fit(Xva, yva)

    p_cal_te = cal.predict_proba(Xte)[:, 1]
    test_metrics = {
        "winner": winner_name,
        "calibrated": evaluate_probs(p_cal_te, yte),
        "uncalibrated_winner": evaluate_probs(
            winner.predict_proba(Xte)[:, 1], yte),
        "candidates_val": val_metrics,
        "candidates_test": {
            name: evaluate_probs(est.predict_proba(Xte)[:, 1], yte)
            for name, est in fitted.items()},
    }

    # Permutation importance on the test fold (PR-AUC scoring).
    perm = permutation_importance(
        cal, Xte, yte, scoring="average_precision",
        n_repeats=5, random_state=seed)
    order = np.argsort(-perm.importances_mean)[:20]
    importance = [{
        "name": feature_names[i],
        "mean": float(perm.importances_mean[i]),
        "std": float(perm.importances_std[i]),
    } for i in order]

    # Calibration/decile bookkeeping for figures.
    frac_obs, mean_pred = calibration_curve(
        yte, p_cal_te, n_bins=10, strategy="quantile")
    curves = {
        "roc": {name: roc_curve(yte, est.predict_proba(Xte)[:, 1])
                for name, est in fitted.items()},
        "pr": {name: precision_recall_curve(yte, est.predict_proba(Xte)[:, 1])
               for name, est in fitted.items()},
        "roc_cal": roc_curve(yte, p_cal_te),
        "pr_cal": precision_recall_curve(yte, p_cal_te),
        "calibration": {"frac_observed": frac_obs, "mean_pred": mean_pred},
        "deciles": decile_table(p_cal_te, yte),
        "prob_test": p_cal_te,
        "y_test": yte,
    }

    slim_meta = {k: v for k, v in meta.items() if k != "raw"}
    slim_meta.update({
        "winner": winner_name,
        "model_class": type(winner).__name__,
        "seed": seed,
        "n_train": int(i_tr.size), "n_val": int(i_va.size),
        "n_test": int(i_te.size),
    })
    bundle = {
        "model": cal,
        "scaler": scaler,
        "feature_names": list(feature_names),
        "meta": slim_meta,
    }

    metrics = {
        **{k: slim_meta[k] for k in
           ("gate_alt", "feature_set", "n_used", "n_dropped",
            "success_rate", "n_train", "n_val", "n_test")},
        "base_rate_test": float(yte.mean()),
        "winner": winner_name,
        "val": {k: {m: v for m, v in val_metrics[k].items()
                    if m != "thresholds"} for k in val_metrics},
        "test_calibrated": test_metrics["calibrated"],
        "test_uncalibrated_winner": test_metrics["uncalibrated_winner"],
        "test_candidates": {k: {m: v for m, v in mt.items()
                                if m != "thresholds"}
                            for k, mt in test_metrics["candidates_test"].items()},
        "permutation_importance": importance,
        "deciles": curves["deciles"],
    }
    return {"bundle": bundle, "metrics": metrics, "curves": curves}
