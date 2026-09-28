"""Tests for the Phase-5b CEM guidance-parameter search."""

import numpy as np
import pytest

from sixdof.batch import BatchMonteCarlo, DispersionConfig
from sixdof.batch.policy_search import (
    HI, LO, PARAMS, THETA0, cem, clip_theta, evaluate,
    theta_to_guidance_params,
)


# ---------------------------------------------------------------------------
def test_theta_mapping_and_clipping():
    gp = theta_to_guidance_params(THETA0)
    assert gp["t_go_scale"] == pytest.approx(1.15)
    assert gp["kv_lat"] == pytest.approx(0.45)
    assert gp["v_lat_max"] == pytest.approx(25.0)
    # degree-valued theta entries map to radians on the attributes
    assert gp["tilt_cap"] == pytest.approx(np.radians(35.0))
    assert gp["tilt_cap_min"] == pytest.approx(np.radians(3.0))
    assert gp["ignite_margin"] == pytest.approx(150.0)
    assert gp["accel_clamp"] == pytest.approx(0.95)
    assert gp["lat_filt_tau"] == pytest.approx(1.5)
    assert len(gp) == len(PARAMS)

    th = clip_theta(np.array([999.0, -999.0, 25.0, 35.0, 3.0,
                              150.0, 0.95, 1.5]))
    assert th[0] == HI[0] and th[1] == LO[1]
    assert np.all(th <= HI) and np.all(th >= LO)


def test_guidance_params_validation():
    with pytest.raises(KeyError):
        BatchMonteCarlo(n=10, dispersion=DispersionConfig.nominal(),
                        guidance_params={"bogus_key": 1.0})


# ---------------------------------------------------------------------------
def test_cem_converges_quadratic_bowl():
    """CEM on a 2-D quadratic bowl reaches the known optimum in <8 gens."""
    target = np.array([1.5, -2.0])

    def bowl(theta, n, seed):
        return -float(np.sum((np.asarray(theta, dtype=float) - target) ** 2))

    out = cem(theta0=[0.0, 0.0], sigma0=[2.0, 2.0], n_eval=1,
              pop=12, elite=4, gens=8, seed=0,
              lo=np.array([-5.0, -5.0]), hi=np.array([5.0, 5.0]),
              evaluate_fn=bowl, bonus_gens=None, budget_s=None,
              verbose=False)
    assert out["gens_run"] <= 8
    assert np.linalg.norm(out["theta_best"] - target) < 0.5
    assert np.linalg.norm(out["mu"] - target) < 0.5
    h = out["history"]
    assert len(h["gen"]) == out["gens_run"]
    # Reward improves over the run.
    assert h["best_reward"][-1] > h["best_reward"][0]


# ---------------------------------------------------------------------------
def test_guidance_param_injection_changes_outcome():
    """Same seed, only ``ignition_margin`` differs -> different trajectories.

    With the nominal (dispersion-free) config every run in a batch is
    identical, so the two batches differ iff the injected parameter
    actually reaches the guidance.  A larger margin raises the hoverslam
    ignition altitude (earlier ignition, longer burn -> less prop).
    """
    disp = DispersionConfig.nominal()
    base = theta_to_guidance_params(THETA0)
    kw = dict(n=24, dispersion=disp, seed=7, dt=0.04)  # coarse dt for speed
    ra = BatchMonteCarlo(guidance_params=dict(base, ignite_margin=50.0),
                         **kw).run()
    rb = BatchMonteCarlo(guidance_params=dict(base, ignite_margin=400.0),
                         **kw).run()
    # Descent differs before touchdown: v_z at the 500 m gate (index 3)
    # differs, and total propellant burned differs.
    assert not np.allclose(ra.gate_states[:, 3, 5], rb.gate_states[:, 3, 5])
    assert not np.allclose(ra.prop_remaining, rb.prop_remaining)
    assert not np.allclose(ra.touchdown_time, rb.touchdown_time)
    # Earlier ignition burns more propellant.
    assert rb.prop_remaining.mean() < ra.prop_remaining.mean()


def test_evaluate_returns_finite_scalar():
    r = evaluate(THETA0, n=24, seed=0, dt=0.04,
                 dispersion=DispersionConfig.nominal())
    assert isinstance(r, float)
    assert np.isfinite(r)
    # reward = +1 minus fuel shaping on success; negative weights on fail
    assert -1.0 - 0.3 <= r <= 1.0
