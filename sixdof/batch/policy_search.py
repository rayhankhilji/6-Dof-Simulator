"""CEM search over the batch ZEM/ZEV guidance constants (Phase 5b).

Optimizes an 8-parameter vector ``theta`` injected into
:class:`~sixdof.batch.batch_mc.BatchMonteCarlo` via its ``guidance_params``
argument.  The searched parameters (defaults -> bounds) are:

===================  ========  =========  ====================================
name                 default   bounds     batch attribute (SI)
===================  ========  =========  ====================================
``t_go_scale``       1.15      [0.9,1.6]  ``t_go_scale``
``kv_lat``           0.45      [.15,1.2]  ``kv_lat`` [1/s]
``v_lat_max``        25        [10,40]    ``v_lat_max`` [m/s]
``tilt_cap_hi``      35 deg    [20,60]    ``tilt_cap`` [rad]
``tilt_cap_lo``      3 deg     [1,8]      ``tilt_cap_min`` [rad]
``ignition_margin``  150       [50,400]   ``ignite_margin`` [m]
``accel_clamp``      0.95      [.85,.99]  ``accel_clamp``
``lat_filt_tau``     1.5       [0.2,3.0]  ``lat_filt_tau`` [s]
===================  ========  =========  ====================================

Fitness per run: ``r = +1`` on success else ``-w[mode]`` with
``w`` = hard_landing 1.0, tipover 0.7, miss_pad 0.7, lateral_velocity 0.5,
fuel_exhausted 0.8, engine_failure / gps_outage_diverged 0.6, timeout 1.0;
successful runs are additionally shaped by ``-0.3 * fuel_used / 10000``
(fuel_used = propellant burned [kg] against the 10 t nominal load) so the
search prefers lower-burn landings, not just survivable ones.

:func:`cem` is a standard cross-entropy-method loop with sigma smoothing
(0.3 blend), bound clipping, and common random numbers: every candidate in
a generation is evaluated on the same seed (``seed + eval_index``, where
``eval_index`` counts generations), so elite selection compares candidates
on identical dispersion draws.  ``evaluate`` wall time is instrumented:
when the amortized per-evaluation time is below ``fast_eval_s`` the
generation count is raised to ``bonus_gens``; whenever the projected
remaining runtime would exceed ``budget_s`` the remaining generations are
cut.  Population member 0 is always the baseline ``theta0``, giving a
paired baseline curve on the same eval budget for free.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from .batch_mc import BatchMonteCarlo, DispersionConfig, FAILURE_MODES, MCResult

# ---------------------------------------------------------------------------
# Parameter spec
# ---------------------------------------------------------------------------

_DEG = np.pi / 180.0
PROP_NOM = 10_000.0   # nominal landing propellant load [kg]
FUEL_COEF = 0.3       # fuel-shaping weight on successful runs


@dataclass(frozen=True)
class PolicyParam:
    """One tunable guidance constant.

    ``attr`` names the ``BatchMonteCarlo`` attribute overridden through
    ``guidance_params``; ``scale`` maps the reported theta value to the
    attribute's SI units (degrees -> radians for the tilt caps).
    """

    name: str
    attr: str
    default: float
    lo: float
    hi: float
    scale: float = 1.0


PARAMS: list[PolicyParam] = [
    PolicyParam("t_go_scale", "t_go_scale", 1.15, 0.9, 1.6),
    PolicyParam("kv_lat", "kv_lat", 0.45, 0.15, 1.2),
    PolicyParam("v_lat_max", "v_lat_max", 25.0, 10.0, 40.0),
    PolicyParam("tilt_cap_hi", "tilt_cap", 35.0, 20.0, 60.0, scale=_DEG),
    PolicyParam("tilt_cap_lo", "tilt_cap_min", 3.0, 1.0, 8.0, scale=_DEG),
    PolicyParam("ignition_margin", "ignite_margin", 150.0, 50.0, 400.0),
    PolicyParam("accel_clamp", "accel_clamp", 0.95, 0.85, 0.99),
    PolicyParam("lat_filt_tau", "lat_filt_tau", 1.5, 0.2, 3.0),
]
PARAM_NAMES = [p.name for p in PARAMS]
THETA0 = np.array([p.default for p in PARAMS])
LO = np.array([p.lo for p in PARAMS])
HI = np.array([p.hi for p in PARAMS])


def theta_to_guidance_params(theta) -> dict:
    """Map a theta vector to a ``guidance_params`` dict (SI values)."""
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (len(PARAMS),):
        raise ValueError(f"theta must have shape ({len(PARAMS)},)")
    return {p.attr: float(p.scale * theta[i]) for i, p in enumerate(PARAMS)}


def clip_theta(theta):
    """Clip a theta vector (or population matrix) to the parameter bounds."""
    return np.clip(np.asarray(theta, dtype=float), LO, HI)


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

FAILURE_WEIGHTS = {
    "success": 0.0,
    "hard_landing": 1.0,
    "lateral_velocity": 0.5,
    "tipover": 0.7,
    "miss_pad": 0.7,
    "fuel_exhausted": 0.8,
    "engine_failure": 0.6,
    "gps_outage_diverged": 0.6,
    "timeout": 1.0,
}
_WEIGHT_VEC = np.array([FAILURE_WEIGHTS[m] for m in FAILURE_MODES])


def reward_per_run(res: MCResult) -> np.ndarray:
    """Per-run fitness ``r = +1`` (success, fuel-shaped) else ``-w[mode]``."""
    r = -_WEIGHT_VEC[res.failure_mode].astype(float)
    succ = res.success
    r[succ] = 1.0
    # Fuel shaping on successes: fuel_used = initial prop - remaining [kg].
    prop0 = PROP_NOM * res.mass_scale
    fuel_used = np.clip(prop0 - res.prop_remaining, 0.0, None)
    r[succ] -= FUEL_COEF * fuel_used[succ] / PROP_NOM
    return r


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def run_batch(theta, n: int, seed: int,
              dispersion: DispersionConfig | None = None,
              dt: float = 0.02, t_end: float = 120.0) -> MCResult:
    """Run an ``n``-run batch at ``seed`` with theta's guidance overrides."""
    gp = theta_to_guidance_params(theta)
    mc = BatchMonteCarlo(n=n, dt=dt, seed=seed,
                         dispersion=dispersion or DispersionConfig(),
                         t_end=t_end, guidance_params=gp)
    return mc.run()


def evaluate(theta, n: int = 1000, seed: int = 0,
             dispersion: DispersionConfig | None = None,
             dt: float = 0.02, t_end: float = 120.0) -> float:
    """Mean reward of ``theta`` over ``n`` batch runs at ``seed``."""
    return float(np.mean(reward_per_run(
        run_batch(theta, n, seed, dispersion, dt, t_end))))


def _eval_star(args):
    """Picklable worker for parallel population evaluation."""
    theta, n, seed, dispersion, dt, t_end = args
    return evaluate(theta, n=n, seed=seed, dispersion=dispersion,
                    dt=dt, t_end=t_end)


def _run_star(args):
    """Picklable worker returning the full ``MCResult`` (final evals)."""
    theta, n, seed, dispersion, dt, t_end = args
    return run_batch(theta, n=n, seed=seed, dispersion=dispersion,
                     dt=dt, t_end=t_end)


# ---------------------------------------------------------------------------
# Cross-entropy method
# ---------------------------------------------------------------------------

def cem(theta0=None, sigma0=None, n_eval: int = 1000,
        pop: int = 20, elite: int = 5, gens: int = 15, seed: int = 0,
        *, lo=None, hi=None, sigma0_frac: float = 0.3,
        sigma_blend: float = 0.3,
        dispersion: DispersionConfig | None = None,
        dt: float = 0.02, t_end: float = 120.0,
        evaluate_fn=None, n_jobs: int = 1,
        fast_eval_s: float = 20.0, bonus_gens: int | None = 18,
        budget_s: float | None = 9000.0,
        verbose: bool = True) -> dict:
    """Cross-entropy-method search over the guidance parameter vector.

    Parameters
    ----------
    theta0 : array-like, optional
        Center of the initial sampling distribution (default ``THETA0``);
        also evaluated every generation as the paired baseline (pop slot 0).
    sigma0 : array-like | float, optional
        Initial per-parameter std (default ``sigma0_frac * (hi - lo)``).
    n_eval : int
        Batch runs per candidate evaluation.
    pop, elite, gens, seed : int
        CEM population size, elite count, generation count, RNG seed.
    lo, hi : array-like, optional
        Parameter bounds (default ``LO``/``HI``).
    sigma_blend : float
        Fraction of the previous sigma retained each update (0.3).
    evaluate_fn : callable, optional
        ``fn(theta, n_eval, seed) -> float`` override (e.g. analytic test
        problems); requires ``n_jobs == 1``.
    n_jobs : int
        Worker processes for population evaluation (default 1 = serial;
        only valid with the default evaluator).
    fast_eval_s : float
        If the amortized per-eval time after the first generation is below
        this, raise the generation target to ``bonus_gens``.
    bonus_gens : int | None
        Raised generation target for fast evals; ``None`` disables.
    budget_s : float | None
        Wall-clock cap; remaining generations are cut to fit. ``None``
        disables the cap.
    verbose : bool
        Log best/mean/elite-mean reward and theta each generation.

    Returns
    -------
    dict
        ``theta_best`` (best evaluated candidate), ``best_reward``,
        final ``mu``/``sigma``, ``gens_run``, ``wall_time_s``, and the
        per-generation ``history`` lists.
    """
    lo = LO.copy() if lo is None else np.asarray(lo, dtype=float)
    hi = HI.copy() if hi is None else np.asarray(hi, dtype=float)
    theta0 = THETA0.copy() if theta0 is None else np.asarray(theta0, float)
    elite = int(min(elite, pop))
    d = int(theta0.size)
    if sigma0 is None:
        sigma = sigma0_frac * (hi - lo)
    else:
        sigma = np.broadcast_to(np.asarray(sigma0, dtype=float), (d,)).copy()
    sigma = np.maximum(sigma, 1e-9)
    mu = np.clip(theta0, lo, hi)
    custom_fn = evaluate_fn is not None
    if custom_fn and n_jobs != 1:
        raise ValueError("parallel evaluation requires the default "
                         "evaluate() (custom evaluate_fn is not picklable)")

    def _eval_one(th, ev_seed):
        if custom_fn:
            return float(evaluate_fn(np.asarray(th, float), n_eval, ev_seed))
        return evaluate(th, n=n_eval, seed=ev_seed,
                        dispersion=dispersion, dt=dt, t_end=t_end)

    rng = np.random.default_rng(seed)
    hist = {k: [] for k in (
        "gen", "eval_seed", "best_reward", "mean_reward",
        "elite_mean_reward", "elite_std", "baseline_reward", "theta_best",
        "mu", "sigma", "eval_time_s", "gen_time_s")}
    best_theta, best_reward = mu.copy(), -np.inf
    gen_times: list[float] = []
    eval_index = 0          # generation counter -> CRN seed = seed + index
    target_gens = int(gens)
    t_start = time.perf_counter()
    pool = (ProcessPoolExecutor(max_workers=n_jobs) if n_jobs > 1 else None)

    g = 0
    try:
        while g < target_gens:
            eval_seed = seed + eval_index
            eval_index += 1
            samples = rng.normal(mu[None, :], sigma[None, :],
                                 size=(pop, d))
            samples = np.clip(samples, lo[None, :], hi[None, :])
            # Paired baseline on the same seed/draws: honest comparison on
            # the same eval budget at zero extra cost.
            samples[0] = theta0

            t_gen = time.perf_counter()
            if pool is not None and not custom_fn:
                args = [(samples[j], n_eval, eval_seed, dispersion, dt,
                         t_end) for j in range(pop)]
                rewards = np.array(list(pool.map(_eval_star, args)))
            else:
                rewards = np.array([_eval_one(samples[j], eval_seed)
                                    for j in range(pop)])
            gen_wall = time.perf_counter() - t_gen
            gen_times.append(gen_wall)
            per_eval = gen_wall / pop

            order = np.argsort(rewards)[::-1]
            ei = order[:elite]
            e_mu = samples[ei].mean(axis=0)
            e_sig = samples[ei].std(axis=0)
            mu = e_mu
            sigma = sigma_blend * sigma + (1.0 - sigma_blend) * e_sig
            sigma = np.maximum(sigma, 1e-9)

            if rewards[order[0]] > best_reward:
                best_reward = float(rewards[order[0]])
                best_theta = samples[order[0]].copy()

            hist["gen"].append(g)
            hist["eval_seed"].append(eval_seed)
            hist["best_reward"].append(float(rewards[order[0]]))
            hist["mean_reward"].append(float(rewards.mean()))
            hist["elite_mean_reward"].append(float(rewards[ei].mean()))
            hist["elite_std"].append(float(rewards[ei].std()))
            hist["baseline_reward"].append(float(rewards[0]))
            hist["theta_best"].append(samples[order[0]].tolist())
            hist["mu"].append(mu.tolist())
            hist["sigma"].append(sigma.tolist())
            hist["eval_time_s"].append(float(per_eval))
            hist["gen_time_s"].append(float(gen_wall))

            elapsed = time.perf_counter() - t_start
            if verbose:
                th = np.array2string(mu, precision=3, suppress_small=False,
                                     max_line_width=200)
                print(f"[cem] gen {g + 1}/{target_gens} seed={eval_seed} "
                      f"best={rewards[order[0]]:.4f} "
                      f"mean={rewards.mean():.4f} "
                      f"elite={rewards[ei].mean():.4f} "
                      f"base={rewards[0]:.4f} "
                      f"eval={per_eval:.1f}s gen={gen_wall:.1f}s "
                      f"elapsed={elapsed / 60:.1f}m mu={th}", flush=True)

            # ---- adaptive budget -------------------------------------
            if g == 0 and bonus_gens is not None and per_eval < fast_eval_s:
                if bonus_gens > target_gens:
                    print(f"[cem] fast evals ({per_eval:.1f}s < "
                          f"{fast_eval_s:.0f}s): gens {target_gens} -> "
                          f"{bonus_gens}", flush=True)
                    target_gens = bonus_gens
            if budget_s is not None:
                est_gen = float(np.median(gen_times))
                affordable = int((budget_s - elapsed) // max(est_gen, 1e-9))
                new_target = min(target_gens, g + 1 + max(affordable, 0))
                if new_target < target_gens:
                    print(f"[cem] budget cap {budget_s / 3600:.2f} h: "
                          f"gens {target_gens} -> {new_target}", flush=True)
                    target_gens = new_target
            g += 1
    finally:
        if pool is not None:
            pool.shutdown()

    wall = time.perf_counter() - t_start
    return {
        "theta_best": best_theta,
        "best_reward": float(best_reward),
        "mu": mu,
        "sigma": sigma,
        "history": hist,
        "gens_run": g,
        "wall_time_s": wall,
    }
