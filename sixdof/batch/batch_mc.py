"""Vectorized full-6-DOF Monte Carlo landing engine (Phase 4).

Simulates ``n`` powered-descent landings simultaneously with numpy arrays.
State per run is the same 14-vector as the single-run sim
(``r_I, v_I, q, omega_B, m``), stored as ``(n, *)`` arrays plus an
``active`` mask for runs that already touched down or timed out.

Integration scheme
------------------
Explicit Euler at ``dt = 0.02 s`` for all state components (quaternion
renormalized each step).  Euler is used instead of the single-run RK4
because the batch cannot afford 4 derivative evaluations per step; the
dynamics timescales (attitude bandwidth ~ a few rad/s, servo wn = 25 rad/s,
flight ~ 30 s) are all >> dt, and ``tests/test_batch_mc.py`` verifies the
nominal batch touchdown box against the single-run RK4 sim.  Actuator /
throttle / RCS lags and all noise processes use *exact* discrete updates
(exponential / OU), so they are not Euler-limited.

Structural fidelity to the single-run pipeline
----------------------------------------------
* Same vehicle constants (``small_landing_vehicle``), same mass/CG/J
  cylinder model (vectorized), same engine model including back-pressure
  loss ``T = throttle * T_vac * thrust_scale - p * A_e`` and the 0.35
  throttle floor (thrust commands below the floor are clamped *up* to it
  while the engine is on -- this is what creates the hover-slam
  overshoot).
* Same ZEM/ZEV guidance (vectorized port of
  ``sixdof.guidance.descent.ZEMZEVGuidance`` / ``PoweredDescentBase``):
  hoverslam ignition estimate, retrograde coast attitude blended to
  <= 25 deg tilt, terminal velocity target switch at 30 m with the aim
  point 0.5 m below the surface, the kinematic t_go estimate
  ``t_go = -2 dz / (v_z + v_fz)`` clamped to [0.5, 120] s and scaled by
  1.15 for brake margin, the bounded-approach-speed lateral servo
  (``v_des = clip((r_f - r)/tau_app, +-25 m/s) * fade`` with
  ``tau_app = clip(0.35 t_go, 4, 12)`` s and ``fade`` ramping 0->1 over
  0-100 m, gain ``kv_lat = 0.45`` 1/s), the altitude-tapered accel tilt
  cap ``|a_lat| <= a_z tan(cap)`` with cap 35 deg -> 3 deg below 150 m
  plus the 0.95 T_max/m magnitude clamp, and the ``_filter_lateral``
  first-order low-pass (tau = 1.5 s) on the lateral command channels
  (each run's filter initializes to the unfiltered command on its first
  powered tick, matching the scalar ``_a_filt is None`` init).
* Same geometric attitude law: ``M = +K_R e - K_w w + w x (J w)`` with
  ``K_R = 4 J``, ``K_w = 2.5 J``, the ``M_i = -T l sin(d_i)`` gimbal
  inversion with anti-saturation scaling, and pseudo-RCS PD
  (k_r = 3, k_w = 6) clipped to (5e3, 4e4, 4e4) N m during coast.
  ``powered`` mirrors ``gcmd.engine_on``: it latches at ignition and is
  *not* cleared by a dead engine (fuel exhaustion / failure), so dead
  runs keep tracking the guidance attitude with the gimbal (harmless at
  zero thrust) and get no RCS torque -- identical to the scalar stack.
* Same actuator models: first-order throttle lag (tau = 0.1 s),
  second-order gimbal servo (wn = 8 pi, zeta = 0.7, rate limit),
  first-order RCS lag (tau = 0.02 s).

Simplifications (documented; all second-order effects)
------------------------------------------------------
* The 50 ms throttle transport delay is folded into the lag.
* Measured attitude/omega are taken as truth (EKF attitude errors are
  ~ mrad, negligible next to the nav position/velocity error model); the
  controller consumes *measured* r, v and *true* q, omega, m.
* Ambient pressure for the guidance thrust limits is evaluated at the
  true (not measured) altitude -- a ~2 m error changes p by ~0.02 %.
* ``thrust_scale`` scales thrust and propellant flow directly
  (``T = thr * scale * T_vac - p A_e``, ``mdot = thr * scale * T_vac /
  (Isp_vac g0)``) and guidance sees the scaled authority.  The single-run
  pipeline instead scales the throttle signal upstream of a [floor, 1]
  clamp, which makes scale > 1 a no-op at full throttle; the batch
  convention is the physically intended one.
* An engine that is dead (failure dispersion or fuel exhaustion) produces
  exactly zero thrust rather than decaying through the throttle floor.

Navigation-error approximation
------------------------------
Rather than running n EKFs, the measured translational state is
``meas = truth + OU error`` (sigma_pos = 2 m, sigma_vel = 0.15 m/s,
tau = 2 s steady state) passed through a per-run first-order lag
(tau ~ U(0.1, meas_delay) s) approximating the measurement delay.
During a per-run GPS-outage window the position OU mean-reversion is
replaced by an integrating random walk (drift ~ sqrt(t), rate
``gps_drift_rate`` m/s^0.5) -- IMU dead-reckoning drift -- and the
velocity-error amplitude doubles.  This approximates the real
sensor/EKF pipeline at a fraction of the cost.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..environment.atmosphere import USStandardAtmosphere1976
from ..environment.gravity import gravity_inertial
from ..vehicle import G0, small_landing_vehicle

# ---------------------------------------------------------------------------
# Constants / result codes
# ---------------------------------------------------------------------------

GATE_ALTITUDES = np.array([2000.0, 1500.0, 1000.0, 500.0, 200.0, 50.0, 0.0])

FAILURE_MODES = [
    "success",
    "hard_landing",         # |v_z| > 3 m/s
    "lateral_velocity",     # lateral speed > 1.5 m/s
    "tipover",              # tilt > 5 deg
    "miss_pad",             # lateral offset > 10 m
    "fuel_exhausted",       # propellant hit zero before touchdown
    "engine_failure",       # engine-fail dispersion fired before touchdown
    "gps_outage_diverged",  # had a GPS outage and failed (nav-attributed)
    "timeout",              # still airborne at t_end
]
_MODE = {m: i for i, m in enumerate(FAILURE_MODES)}

_XB = np.array([1.0, 0.0, 0.0])
_ZH = np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Fast small-array helpers (np.cross / np.linalg.norm have heavy per-call
# Python overhead that dominates the inner loop at batch sizes ~1e4-1e5)
# ---------------------------------------------------------------------------

def _cross3(a, b):
    """a x b for (n,3) x (n,3) -- ~10x faster than np.cross."""
    return np.stack(
        [a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
         a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
         a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]],
        axis=1)


def _norm(a):
    """Row-wise 2-norm of an (n,k) array."""
    return np.sqrt(np.einsum("ij,ij->i", a, a))


# ---------------------------------------------------------------------------
# Vectorized quaternion helpers (scalar-first, q: body -> inertial)
# ---------------------------------------------------------------------------

def _qnorm(q):
    return q / np.maximum(_norm(q)[:, None], 1e-12)


def _qconj(q):
    out = q.copy()
    out[:, 1:] *= -1.0
    return out


def _qmul(p, q):
    w1, x1, y1, z1 = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    w2, x2, y2, z2 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=1,
    )


def _qrot(q, v):
    """v_I = R(q) v_B, vectorized over (n,4) x (n,3)."""
    qv = q[:, 1:]
    t = 2.0 * _cross3(qv, v)
    return v + q[:, 0:1] * t + _cross3(qv, t)


def _qrot_inv(q, v):
    """v_B = R(q)^T v_I."""
    return _qrot(_qconj(q), v)


def _qerr(q, qd):
    """2 * vec(q^-1 (x) q_des), shortest-path sign fix; body frame."""
    e = _qmul(_qconj(_qnorm(q)), _qnorm(qd))
    s = np.where(e[:, 0] < 0.0, -1.0, 1.0)
    return 2.0 * e[:, 1:] * s[:, None]


def _qdot(q, w):
    """q_dot = 0.5 q (x) [0, w_B], vectorized."""
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]
    return 0.5 * np.stack(
        [
            -qx * wx - qy * wy - qz * wz,
            qw * wx + qy * wz - qz * wy,
            qw * wy - qx * wz + qz * wx,
            qw * wz + qx * wy - qy * wx,
        ],
        axis=1,
    )


def _q_from_two(a, b):
    """Shortest-arc quaternion rotating vectors ``a`` onto ``b`` (both (n,3))."""
    an = a / np.maximum(_norm(a)[:, None], 1e-12)
    bn = b / np.maximum(_norm(b)[:, None], 1e-12)
    c = _cross3(an, bn)
    d = np.einsum("ij,ij->i", an, bn)
    q = np.empty((a.shape[0], 4))
    q[:, 0] = 1.0 + d
    q[:, 1:] = c
    anti = d < -1.0 + 1e-9
    if anti.any():
        k = int(anti.sum())
        axis = _cross3(an[anti], np.broadcast_to(_XB, (k, 3)))
        small = _norm(axis) < 1e-6
        if small.any():
            # Upstream fallback axis is +y ([0,1,0]), not +z.
            axis[small] = _cross3(
                an[anti][small],
                np.broadcast_to(np.array([0.0, 1.0, 0.0]),
                                (int(small.sum()), 3)))
        axis = axis / np.maximum(_norm(axis)[:, None], 1e-12)
        q[anti, 0] = 0.0
        q[anti, 1:] = axis
    return _qnorm(q)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DispersionConfig:
    """Monte Carlo dispersion parameters (all per-run draws at init).

    ``nominal()`` returns a config with every dispersion disabled.
    """

    wind_max: float = 12.0          # wind_speed ~ U(0, wind_max) at 10 m ref [m/s]
    wind_shear_exp: float = 0.14    # power-law shear exponent
    gust_sigma: float = 2.0         # per-run OU gust sigma ~ U(0, gust_sigma) [m/s]
    gust_L: float = 175.0           # turbulence length scale [m]
    density_sigma: float = 0.10     # density_scale = exp(N(0, sigma)) (lognormal)
    mass_sigma: float = 0.03        # mass_scale ~ N(1, sigma), multiplicative
    thrust_sigma: float = 0.03      # thrust_scale ~ N(1, sigma)
    ic_sigma_pos: float = 5.0       # IC position offset sigma per axis [m]
    ic_sigma_vel: float = 1.0       # IC velocity offset sigma per axis [m/s]
    tilt0_deg: float = 10.0         # initial tilt of +x_B off vertical [deg]
    # navigation-error approximation (see module docstring)
    nav_sigma_pos: float = 2.0      # OU position-error steady-state sigma [m]
    nav_sigma_vel: float = 0.15     # OU velocity-error sigma [m/s]
    nav_tau: float = 2.0            # OU time constant [s]
    meas_delay: float = 0.3         # meas lag tau ~ U(min(0.1, x), x) s; 0 disables
    # GPS outage (dead-reckoning) windows
    gps_outage_prob: float = 0.2
    gps_outage_start: tuple = (8.0, 30.0)   # U(start) seconds after t0
    gps_outage_dur: tuple = (3.0, 12.0)     # U(duration) seconds
    gps_drift_rate: float = 1.5             # m/sqrt(s) position random walk
    gps_vel_scale: float = 2.0              # velocity-error multiplier in outage
    # engine reliability
    engine_fail_prob: float = 0.02
    engine_fail_t: tuple = (5.0, 40.0)      # U(failure time) seconds
    # control
    control_noise: float = 0.0              # sigma [rad] on gimbal commands

    @classmethod
    def nominal(cls) -> "DispersionConfig":
        """All dispersions disabled (nominal scenario)."""
        return cls(
            wind_max=0.0, gust_sigma=0.0, density_sigma=0.0, mass_sigma=0.0,
            thrust_sigma=0.0, ic_sigma_pos=0.0, ic_sigma_vel=0.0,
            nav_sigma_pos=0.0, nav_sigma_vel=0.0, meas_delay=0.0,
            gps_outage_prob=0.0, engine_fail_prob=0.0, control_noise=0.0,
        )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MCResult:
    """Aggregated Monte Carlo output (all leading-dim-n arrays)."""

    n: int
    runtime_s: float
    seed: int
    dt: float
    # touchdown metrics (touchdown_time NaN for timeouts; other metrics are
    # still recorded from the final state so distributions stay populated)
    touchdown_time: np.ndarray        # (n,)
    vertical_speed: np.ndarray        # (n,)
    lateral_speed: np.ndarray         # (n,)
    tilt_deg: np.ndarray              # (n,)
    lateral_offset: np.ndarray        # (n,)
    prop_remaining: np.ndarray        # (n,)
    fuel_remaining_frac: np.ndarray   # (n,)
    success: np.ndarray               # (n,) bool
    failure_mode: np.ndarray          # (n,) int codes into FAILURE_MODES
    final_state: np.ndarray           # (n, 14)
    gate_positions: np.ndarray        # (n, 7, 3) interpolated at GATE_ALTITUDES
    gate_states: np.ndarray           # (n, 7, 7) [r_I(3), v_I(3), m] at crossing
    gate_times: np.ndarray            # (n, 7) gate-crossing time since t0 [s]
    gate_altitudes: np.ndarray        # (7,)
    # dispersion draws (factor analysis / reproducibility)
    wind_speed: np.ndarray
    wind_dir: np.ndarray
    gust_sigma: np.ndarray
    mass_scale: np.ndarray
    thrust_scale: np.ndarray
    density_scale: np.ndarray
    meas_delay_tau: np.ndarray
    gps_outage: np.ndarray            # bool
    engine_fail: np.ndarray           # bool
    engine_fail_t: np.ndarray         # -1 when no failure

    def save_npz(self, path) -> None:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__
             if k not in ("n", "runtime_s", "seed", "dt")}
        np.savez(path, n=self.n, runtime_s=self.runtime_s, seed=self.seed,
                 dt=self.dt, failure_modes=np.array(FAILURE_MODES), **d)

    @classmethod
    def concatenate(cls, parts: list["MCResult"]) -> "MCResult":
        """Concatenate per-chunk results along the run axis."""
        if len(parts) == 1:
            return parts[0]
        out = cls.__new__(cls)
        out.n = sum(p.n for p in parts)
        out.runtime_s = sum(p.runtime_s for p in parts)
        out.seed = parts[0].seed
        out.dt = parts[0].dt
        out.gate_altitudes = parts[0].gate_altitudes
        for f in cls.__dataclass_fields__:
            if f in ("n", "runtime_s", "seed", "dt", "gate_altitudes"):
                continue
            setattr(out, f, np.concatenate([getattr(p, f) for p in parts], axis=0))
        return out


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BatchMonteCarlo:
    """Vectorized Monte Carlo powered-descent simulator.

    Parameters
    ----------
    n : int
        Number of parallel runs.
    dt : float
        Physics step [s] (Euler; see module docstring).
    control_decimation : int
        Guidance/control update every this many physics steps
        (control rate = ``1 / (dt * control_decimation)``; default 50 Hz).
    seed : int
        RNG seed (``np.random.default_rng``).
    dispersion : DispersionConfig
    guidance : str
        Only ``"zemzev"`` is implemented.
    t_end : float
        Timeout cap [s]; runs still airborne are recorded as 'timeout'.
    """

    def __init__(self, n: int = 100_000, dt: float = 0.02,
                 control_decimation: int = 1, seed: int = 0,
                 dispersion: DispersionConfig | None = None,
                 guidance: str = "zemzev", t_end: float = 120.0) -> None:
        if guidance != "zemzev":
            raise NotImplementedError("batch engine implements 'zemzev' only")
        self.n = int(n)
        self.dt = float(dt)
        self.dec = int(control_decimation)
        self.seed = int(seed)
        self.disp = dispersion or DispersionConfig()
        self.t_end = float(t_end)

        # Vehicle constants (from small_landing_vehicle, single stage).
        stage = small_landing_vehicle().stages[0]
        eng, aero = stage.engine, stage.aero
        self.dry_nom = stage.dry_mass
        self.prop0_nom = 10_000.0            # prop load at entry (scenarios.py)
        self.prop_cap_nom = stage.prop_mass
        self.length = stage.length
        self.radius = stage.diameter / 2.0
        self.cg_dry = stage.cg_offset_from_base_dry
        self.cg_prop_full = stage.cg_offset_prop
        self.gimbal_x = stage.gimbal_point
        self.cp_x = stage.cp_offset
        self.T_vac = eng.thrust_max_vac
        self.min_frac = eng.thrust_min_frac
        self.isp_vac = eng.isp_vac
        self.Ae = eng.nozzle_exit_area * eng.n_engines
        self.gimbal_max = eng.gimbal_max
        self.gimbal_rate = eng.gimbal_rate_max
        self.throttle_tau = eng.throttle_tau
        self.S = aero.ref_area
        self.cn_alpha = aero.cn_alpha
        self.cm_damp = aero.cm_damping
        self.cl_roll = aero.cl_roll_damping
        self.cd_mach, self.cd_vals = aero.cd_table
        # Servo (mirrors GimbalActuator) and RCS (mirrors RCSActuator).
        self.servo_wn = 2.0 * np.pi * 4.0
        self.servo_zeta = 0.7
        self.rcs_tau = 0.02
        self.rcs_max = np.array([5e3, 4e4, 4e4])
        # Attitude gains (mirror GeometricController / ControllerBase._rcs_cmd).
        self.k_r, self.k_w = 4.0, 2.5
        self.rcs_k_r, self.rcs_k_w = 3.0, 6.0
        # Guidance constants (mirror PoweredDescentBase / ZEMZEVGuidance).
        self.h_terminal = 30.0
        self.tilt_max = np.radians(25.0)
        self.tilt_cap = np.radians(35.0)   # ZEMZEVGuidance(tilt_cap_deg=35)
        self.tilt_cap_min = np.radians(3.0)
        self.h_cap_ref = 150.0             # altitude taper for the tilt cap
        self.t_go_scale = 1.15             # brake margin (longer plan t_go)
        # Bounded-approach-speed lateral servo (PoweredDescentBase._lateral_servo).
        self.v_lat_max = 25.0              # bounded approach speed [m/s]
        self.kv_lat = 0.45                 # lateral velocity-servo gain [1/s]
        self.lat_filt_tau = 1.5            # _filter_lateral low-pass tau [s]
        self.ignite_margin = 150.0
        self.ignite_decel_frac = 0.85

        self.atm = USStandardAtmosphere1976(1.0)  # per-run scale applied manually
        self.aero_cfg = aero
        self.rng: np.random.Generator | None = None
        self.trace: list[dict] = []   # populated when trace_on=True (debugging)
        self.trace_on = False
        # Probe the scalar aero model for its effective normal-force /
        # CP-moment sign conventions so the vectorized port stays consistent
        # with upstream adjustments (see _probe_aero_signs).
        self.fn_sign, self.mn_sign = self._probe_aero_signs()

    # ------------------------------------------------------------------
    def _probe_aero_signs(self):
        """Effective sign conventions of ``aero.aero_forces_moments``.

        Probe state: q = identity (x_B = +x_I), v_rel_B = (-100, 0, +5)
        -> alpha ~ 177 deg, e_n_plane = +z_B.  ``fn_sign`` is the sign of
        the normal force along +e_n_plane; ``mn_sign`` is the sign of the
        CP moment relative to ``cross(r_cp, F_n)`` (r_cp = +x_B when the
        CP sits above the CG).  Probing (rather than hard-coding) keeps
        the batch port consistent if the upstream aero conventions are
        adjusted -- e.g. a normal-force sign or CP-offset fix.
        """
        from ..aero import aero_forces_moments
        cg_probe = self.cg_dry                    # 8.0 m
        r_I = np.array([0.0, 0.0, 1000.0])
        v_I = np.array([-100.0, 0.0, 5.0])        # = v_rel (no wind)
        q_id = np.array([1.0, 0.0, 0.0, 0.0])
        F_B, M_B, _ = aero_forces_moments(
            r_I, v_I, q_id, np.zeros(3), self.aero_cfg, self.cp_x,
            cg_probe, self.length, self.atm, np.zeros(3))
        # With e_n = +z_B, the signed normal force along +e_n is just F_z
        # (drag contribution ~ -0.015 qS << |F_n| ~ 6 qS).
        fn_sign = float(np.sign(F_B[2])) or 1.0
        a_arm = self.cp_x - cg_probe              # r_cp x-offset
        # cross(a x^, Fz z^)_y = -a Fz ; compare against actual M_y.
        mn_sign = float(np.sign(M_B[1] / (-a_arm * F_B[2]))) or 1.0
        return fn_sign, mn_sign

    # ------------------------------------------------------------------
    def _mass_props(self, prop, dry):
        """Vectorized cylinder mass model -> (cg_x, J diag (n,3)).

        ``prop`` and ``dry`` are already per-run (mass-scaled) values; the
        scale cancels in cg/fill and scales J correctly.
        """
        prop_cap = self.prop_cap_nom * (dry / self.dry_nom)
        fill = np.clip(prop / np.maximum(prop_cap, 1.0), 0.0, 1.0)
        h_prop = fill * 2.0 * self.cg_prop_full
        cg_prop = 0.5 * h_prop
        m = dry + prop
        cg = (dry * self.cg_dry + prop * cg_prop) / np.maximum(m, 1.0)
        r2 = self.radius**2
        Jx = 0.5 * m * r2
        Jy = (dry * (3.0 * r2 + self.length**2) / 12.0
              + dry * (self.cg_dry - cg) ** 2
              + prop * (3.0 * r2 + h_prop**2) / 12.0
              + prop * (cg_prop - cg) ** 2)
        Jd = np.stack([Jx, Jy, Jy], axis=1)
        return cg, Jd

    # ------------------------------------------------------------------
    def _draw_dispersions(self, rng):
        n, d = self.n, self.disp
        D = {}
        D["wind_speed"] = (rng.uniform(0.0, d.wind_max, n)
                           if d.wind_max > 0 else np.zeros(n))
        D["wind_dir"] = rng.uniform(0.0, 2.0 * np.pi, n)
        D["gust_sigma"] = (rng.uniform(0.0, d.gust_sigma, n)
                           if d.gust_sigma > 0 else np.zeros(n))
        D["mass_scale"] = np.clip(1.0 + d.mass_sigma * rng.standard_normal(n),
                                  0.7, 1.3)
        D["thrust_scale"] = np.clip(1.0 + d.thrust_sigma * rng.standard_normal(n),
                                    0.5, 1.5)
        D["density_scale"] = (np.exp(d.density_sigma * rng.standard_normal(n))
                              if d.density_sigma > 0 else np.ones(n))
        D["meas_delay_tau"] = (rng.uniform(min(0.1, d.meas_delay), d.meas_delay, n)
                               if d.meas_delay > 0 else np.zeros(n))
        D["gps_outage"] = rng.random(n) < d.gps_outage_prob
        D["gps_start"] = rng.uniform(*d.gps_outage_start, n)
        D["gps_dur"] = rng.uniform(*d.gps_outage_dur, n)
        D["engine_fail"] = rng.random(n) < d.engine_fail_prob
        D["engine_fail_t"] = np.where(D["engine_fail"],
                                      rng.uniform(*d.engine_fail_t, n), np.inf)
        return D

    # ------------------------------------------------------------------
    def run(self) -> MCResult:
        n, dt, d = self.n, self.dt, self.disp
        rng = np.random.default_rng(self.seed)
        self.rng = rng
        D = self._draw_dispersions(rng)

        # --- state ---
        r = np.tile(np.array([-400.0, 200.0, 3000.0]), (n, 1))
        v = np.tile(np.array([30.0, -15.0, -180.0]), (n, 1))
        if d.ic_sigma_pos > 0:
            r += d.ic_sigma_pos * rng.standard_normal((n, 3))
        if d.ic_sigma_vel > 0:
            v += d.ic_sigma_vel * rng.standard_normal((n, 3))
        # q0: +x_B tilted tilt0_deg from +z toward -horizontal velocity.
        v_h = v.copy()
        v_h[:, 2] = 0.0
        vhn = _norm(v_h)[:, None]
        opp = np.where(vhn > 1e-9, -v_h / np.maximum(vhn, 1e-9),
                       np.broadcast_to(np.array([-1.0, 0.0, 0.0]), (n, 3)))
        t0 = np.radians(d.tilt0_deg)
        x_b_dir = _ZH * np.cos(t0) + opp * np.sin(t0)
        q = _q_from_two(np.broadcast_to(_XB, (n, 3)), x_b_dir)
        w = np.zeros((n, 3))
        mass_scale = D["mass_scale"]
        dry = self.dry_nom * mass_scale
        prop0 = self.prop0_nom * mass_scale
        m = dry + prop0

        # --- environment state ---
        steady = np.stack([D["wind_speed"] * np.cos(D["wind_dir"]),
                           D["wind_speed"] * np.sin(D["wind_dir"]),
                           np.zeros(n)], axis=1)
        gust = np.zeros((n, 3))

        # --- nav-error state ---
        err_r = d.nav_sigma_pos * rng.standard_normal((n, 3))
        err_v = d.nav_sigma_vel * rng.standard_normal((n, 3))
        tau_d = D["meas_delay_tau"]
        a_d = np.where(tau_d > 1e-9,
                       1.0 - np.exp(-dt / np.maximum(tau_d, 1e-9)), 1.0)
        r_lag = r + err_r
        v_lag = v + err_v

        # --- status flags / ZOH commands ---
        active = np.ones(n, dtype=bool)
        ignited = np.zeros(n, dtype=bool)
        starved = np.zeros(n, dtype=bool)
        engine_failed = np.zeros(n, dtype=bool)
        powered = np.zeros(n, dtype=bool)      # guidance says burn (ZOH)
        thr_cmd = np.zeros(n)
        thr_val = np.zeros(n)
        gim_cmd = np.zeros((n, 2))
        gim_d = np.zeros((n, 2))
        gim_dd = np.zeros((n, 2))
        rcs_cmd = np.zeros((n, 3))
        rcs_val = np.zeros((n, 3))
        a_filt = np.zeros((n, 2))      # _filter_lateral state (per-run)

        # --- outputs ---
        td_time = np.full(n, np.nan)
        td_vs = np.full(n, np.nan)
        td_ls = np.full(n, np.nan)
        td_tilt = np.full(n, np.nan)
        td_off = np.full(n, np.nan)
        gates = np.full((n, len(GATE_ALTITUDES), 3), np.nan)
        gate_states = np.full((n, len(GATE_ALTITUDES), 7), np.nan)
        gate_times = np.full((n, len(GATE_ALTITUDES)), np.nan)
        prop_at_end = np.zeros(n)

        # precomputed update coefficients
        phi_nav = np.exp(-dt / d.nav_tau)
        q_nav = np.sqrt(max(1.0 - phi_nav**2, 0.0))
        a_thr = 1.0 - np.exp(-dt / self.throttle_tau)
        a_rcs = 1.0 - np.exp(-dt / self.rcs_tau)
        drift = d.gps_drift_rate * np.sqrt(dt)
        shear_top = (300.0 / 10.0) ** d.wind_shear_exp
        nav_active = (d.nav_sigma_pos > 0 or d.nav_sigma_vel > 0
                      or D["gps_outage"].any())

        n_steps = int(np.ceil(self.t_end / dt))
        t = 0.0
        t_start = time.perf_counter()

        for k in range(n_steps):
            if not active.any():
                break
            r_prev = r.copy()
            v_prev = v.copy()
            m_prev = m.copy()

            # ---------------- environment processes ----------------
            h_w = np.clip(r[:, 2], 0.0, None)
            shear = np.clip((np.minimum(h_w, 300.0) / 10.0) ** d.wind_shear_exp,
                            0.0, shear_top)
            wind = steady * shear[:, None] + gust
            v_rel_I = v - wind
            V = _norm(v_rel_I)
            if d.gust_sigma > 0:
                phi_g = np.exp(-dt * np.maximum(V, 1.0) / d.gust_L)
                gust = (phi_g[:, None] * gust
                        + D["gust_sigma"][:, None]
                        * np.sqrt(1.0 - phi_g**2)[:, None]
                        * rng.standard_normal((n, 3)))
                gust[:, 2] = 0.0

            # ---------------- navigation-error process ----------------
            if nav_active:
                in_out = (D["gps_outage"] & (t >= D["gps_start"])
                          & (t < D["gps_start"] + D["gps_dur"]))
                rw = drift * rng.standard_normal((n, 3))
                ou = d.nav_sigma_pos * q_nav * rng.standard_normal((n, 3))
                err_r = np.where(in_out[:, None], err_r + rw,
                                 phi_nav * err_r + ou)
                sv = d.nav_sigma_vel * np.where(in_out, d.gps_vel_scale, 1.0)
                err_v = (phi_nav * err_v
                         + sv[:, None] * q_nav * rng.standard_normal((n, 3)))
            r_lag += a_d[:, None] * (r + err_r - r_lag)
            v_lag += a_d[:, None] * (v + err_v - v_lag)

            # ---------------- environment at true state ----------------
            _, p, rho, a_snd = self.atm.properties(r[:, 2])
            rho = rho * D["density_scale"]
            p = p * D["density_scale"]
            g_vec = gravity_inertial(r)

            prop = np.clip(m - dry, 0.0, None)
            starved |= (prop <= 0.0) & active
            cg, Jd = self._mass_props(prop, dry)

            dead = starved | (t >= D["engine_fail_t"])
            engine_failed |= (t >= D["engine_fail_t"]) & active

            # ---------------- control update (ZOH) ----------------
            if k % self.dec == 0:
                powered = self._control_update(
                    r_lag, v_lag, q, w, m, cg, Jd, p, g_vec, D,
                    active, ignited, dead,
                    thr_cmd, gim_cmd, rcs_cmd, a_filt)
            burning = powered & ~dead & active

            # ---------------- actuators ----------------
            thr_val += a_thr * (thr_cmd - thr_val)
            # Nonzero throttle clamps up to the floor (hover-slam physics).
            thr_eff = np.where(burning & (thr_val > 0.0),
                               np.clip(thr_val, self.min_frac, 1.0), 0.0)
            # Gimbal servo: 2nd-order, semi-implicit Euler (GimbalActuator).
            acc = (self.servo_wn**2 * (gim_cmd - gim_d)
                   - 2.0 * self.servo_zeta * self.servo_wn * gim_dd)
            gim_dd = np.clip(gim_dd + acc * dt,
                             -self.gimbal_rate, self.gimbal_rate)
            gim_d = np.clip(gim_d + gim_dd * dt,
                            -self.gimbal_max, self.gimbal_max)
            # RCS lag.
            rcs_val += a_rcs * (rcs_cmd - rcs_val)

            # ---------------- forces & moments ----------------
            thrust = np.clip(thr_eff * D["thrust_scale"] * self.T_vac
                             - p * self.Ae, 0.0, None)
            dy, dz = gim_d[:, 0], gim_d[:, 1]
            t_B = np.stack([np.cos(dy) * np.cos(dz), np.sin(dz),
                            -np.sin(dy) * np.cos(dz)], axis=1)
            F_thr_B = thrust[:, None] * t_B

            # Aero in body frame (mirrors aero_forces_moments).
            v_rel_B = _qrot_inv(q, v_rel_I)
            Vb = _norm(v_rel_B)
            moving = Vb > 1e-3
            Vc = np.maximum(Vb, 1e-3)
            alpha = np.arctan2(np.hypot(v_rel_B[:, 1], v_rel_B[:, 2]),
                               v_rel_B[:, 0])
            mach = Vb / np.maximum(a_snd, 1.0)
            Cd = np.interp(mach, self.cd_mach, self.cd_vals)
            qdyn = 0.5 * rho * Vb * Vb
            e_rel = v_rel_B / Vc[:, None]
            F_drag = -Cd[:, None] * qdyn[:, None] * self.S * e_rel
            nvec = np.stack([np.zeros(n), v_rel_B[:, 1], v_rel_B[:, 2]], axis=1)
            nvec /= np.maximum(_norm(nvec)[:, None], 1e-9)
            F_nrm = (self.fn_sign * self.cn_alpha
                     * np.sin(alpha)[:, None]
                     * qdyn[:, None] * self.S * nvec)
            F_aero_B = np.where(moving[:, None], F_drag + F_nrm, 0.0)
            r_cp = np.stack([np.full(n, self.cp_x) - cg,
                             np.zeros(n), np.zeros(n)], axis=1)
            M_aero = self.mn_sign * _cross3(r_cp, F_nrm)
            w_hat = w * self.length / (2.0 * Vc[:, None])
            qsl = (qdyn * self.S * self.length)[:, None]
            M_aero += qsl * np.stack([self.cl_roll * w_hat[:, 0],
                                      self.cm_damp * w_hat[:, 1],
                                      self.cm_damp * w_hat[:, 2]], axis=1)
            M_aero = np.where(moving[:, None], M_aero, 0.0)

            F_I = _qrot(q, F_thr_B + F_aero_B) + m[:, None] * g_vec
            r_g = np.stack([-cg, np.zeros(n), np.zeros(n)], axis=1)
            Jw = Jd * w
            M_B = (_cross3(r_g, F_thr_B) + M_aero + rcs_val
                   - _cross3(w, Jw))
            w_dot = M_B / np.maximum(Jd, 1.0)

            mdot = np.where(burning,
                            thr_eff * D["thrust_scale"] * self.T_vac
                            / (self.isp_vac * G0), 0.0)

            if self.trace_on:
                self.trace.append(dict(
                    t=t, r=r.copy(), v=v.copy(), q=q.copy(), w=w.copy(),
                    m=m.copy(), thrust=thrust.copy(), thr_eff=thr_eff.copy(),
                    ignited=ignited.copy(), powered=powered.copy(),
                    active=active.copy(), gim_d=gim_d.copy(),
                    thr_cmd=thr_cmd.copy(), a_cmd=getattr(self, "_last_acmd", None)))

            # ---------------- Euler step ----------------
            am = active[:, None]
            r += dt * v * am
            v += dt * (F_I / np.maximum(m[:, None], 1.0)) * am
            q += dt * _qdot(q, w) * am
            q = _qnorm(q)
            w += dt * w_dot * am
            m = np.maximum(m - dt * mdot, dry)

            # ---------------- touchdown & gates ----------------
            t += dt
            z_now = r[:, 2]
            td = active & (z_now <= 0.0)
            if td.any():
                f = np.clip(r_prev[td, 2]
                            / np.maximum(r_prev[td, 2] - z_now[td], 1e-9),
                            0.0, 1.0)
                r_td = r_prev[td] + f[:, None] * (r[td] - r_prev[td])
                x_ax = _qrot(q[td], np.broadcast_to(_XB, (int(td.sum()), 3)))
                td_vs[td] = np.abs(v[td, 2])
                td_ls[td] = np.hypot(v[td, 0], v[td, 1])
                td_tilt[td] = np.degrees(
                    np.arccos(np.clip(x_ax[:, 2], -1.0, 1.0)))
                td_off[td] = np.hypot(r_td[:, 0], r_td[:, 1])
                td_time[td] = t
                prop_at_end[td] = np.clip(m[td] - dry[td], 0.0, None)
                gates[td, -1, :] = r_td          # z = 0 gate
                gate_states[td, -1, :3] = r_td
                gate_states[td, -1, 3:6] = (v_prev[td]
                                            + f[:, None]
                                            * (v[td] - v_prev[td]))
                gate_states[td, -1, 6] = (m_prev[td]
                                          + f * (m[td] - m_prev[td]))
                gate_times[td, -1] = t - dt + f * dt
            for gi, ga in enumerate(GATE_ALTITUDES[:-1]):
                cross = active & (r_prev[:, 2] > ga) & (z_now <= ga)
                if cross.any():
                    f = ((r_prev[cross, 2] - ga)
                         / np.maximum(r_prev[cross, 2] - z_now[cross], 1e-9))
                    gates[cross, gi, :] = (r_prev[cross]
                                           + f[:, None]
                                           * (r[cross] - r_prev[cross]))
                    gate_states[cross, gi, :3] = gates[cross, gi, :]
                    gate_states[cross, gi, 3:6] = (
                        v_prev[cross]
                        + f[:, None] * (v[cross] - v_prev[cross]))
                    gate_states[cross, gi, 6] = (
                        m_prev[cross] + f * (m[cross] - m_prev[cross]))
                    gate_times[cross, gi] = t - dt + f * dt
            active &= ~td
            prop_at_end[active] = np.clip(m[active] - dry[active], 0.0, None)

        runtime = time.perf_counter() - t_start

        # ---------------- metrics & failure modes ----------------
        touched = ~np.isnan(td_time)
        # Timeouts: record final-state metrics too (excluded by 'timeout').
        x_ax = _qrot(q, np.broadcast_to(_XB, (n, 3)))
        tilt_fin = np.degrees(np.arccos(np.clip(x_ax[:, 2], -1.0, 1.0)))
        td_vs = np.where(np.isnan(td_vs), np.abs(v[:, 2]), td_vs)
        td_ls = np.where(np.isnan(td_ls), np.hypot(v[:, 0], v[:, 1]), td_ls)
        td_tilt = np.where(np.isnan(td_tilt), tilt_fin, td_tilt)
        td_off = np.where(np.isnan(td_off),
                          np.hypot(r[:, 0], r[:, 1]), td_off)

        base = np.full(n, _MODE["success"], dtype=np.int8)
        checks = [
            ("hard_landing", td_vs <= 3.0),
            ("lateral_velocity", td_ls <= 1.5),
            ("tipover", td_tilt <= 5.0),
            ("miss_pad", td_off <= 10.0),
            ("fuel_exhausted", prop_at_end > 0.0),
        ]
        for name, ok in checks:
            viol = touched & ~ok & (base == _MODE["success"])
            base[viol] = _MODE[name]
        # Root-cause overrides for failing touched-down runs.
        fail = touched & (base != _MODE["success"])
        over = fail & starved
        base[over] = _MODE["fuel_exhausted"]
        over = fail & ~starved & engine_failed
        base[over] = _MODE["engine_failure"]
        over = fail & ~starved & ~engine_failed & D["gps_outage"]
        base[over] = _MODE["gps_outage_diverged"]
        base[~touched] = _MODE["timeout"]
        success = base == _MODE["success"]

        final_state = np.concatenate([r, v, q, w, m[:, None]], axis=1)
        return MCResult(
            n=n, runtime_s=runtime, seed=self.seed, dt=dt,
            touchdown_time=td_time, vertical_speed=td_vs,
            lateral_speed=td_ls, tilt_deg=td_tilt, lateral_offset=td_off,
            prop_remaining=prop_at_end,
            fuel_remaining_frac=prop_at_end / np.maximum(prop0, 1.0),
            success=success, failure_mode=base,
            final_state=final_state, gate_positions=gates,
            gate_states=gate_states, gate_times=gate_times,
            gate_altitudes=GATE_ALTITUDES.copy(),
            wind_speed=D["wind_speed"], wind_dir=D["wind_dir"],
            gust_sigma=D["gust_sigma"], mass_scale=D["mass_scale"],
            thrust_scale=D["thrust_scale"], density_scale=D["density_scale"],
            meas_delay_tau=D["meas_delay_tau"],
            gps_outage=D["gps_outage"], engine_fail=D["engine_fail"],
            engine_fail_t=np.where(np.isfinite(D["engine_fail_t"]),
                                   D["engine_fail_t"], -1.0),
        )

    # ------------------------------------------------------------------
    def _control_update(self, r_m, v_m, q, w, m, cg, Jd, p, g_vec, D,
                        active, ignited, dead,
                        thr_cmd, gim_cmd, rcs_cmd, a_filt):
        """One vectorized guidance+control tick; returns `powered` mask.

        ``r_m``/``v_m`` are the lagged measured position/velocity;
        q, w, m are truth (nav approximation, see module docstring).
        """
        n = self.n
        rng = self.rng
        z_m, vz_m = r_m[:, 2], v_m[:, 2]
        g_mag = _norm(g_vec)
        T_max = np.maximum(D["thrust_scale"] * self.T_vac - p * self.Ae, 0.0)
        T_min = np.maximum(self.min_frac * D["thrust_scale"] * self.T_vac
                           - p * self.Ae, 0.0)
        a_lim = np.maximum(T_max / np.maximum(m, 1.0), 1e-3)

        # --- ignition latch (hoverslam estimate, PoweredDescentBase) ---
        a_net = np.maximum(self.ignite_decel_frac * (a_lim - g_mag), 1e-3)
        h_ig = vz_m * vz_m / (2.0 * a_net) + self.ignite_margin
        newly = active & ~ignited & (vz_m < 0.0) & (z_m <= h_ig)
        ignited |= newly
        # powered mirrors upstream gcmd.engine_on: latched at ignition and
        # NOT cleared by a dead engine -- the scalar guidance keeps
        # requesting the burn (thrust is zeroed by `burning` downstream),
        # the controller keeps tracking the guidance attitude, and no RCS
        # torque is applied (GeometricController only fires RCS when
        # engine_on is False).
        powered = active & ignited

        # --- ZEM/ZEV on the powered subset ---
        # Terminal aim point sits 0.5 m below the surface (mirrors
        # PoweredDescentBase._targets) so the descent flies through z=0
        # instead of hovering at the t_go-floor stand-off.
        a_cmd = np.zeros((n, 3))
        ip = np.flatnonzero(powered)
        if ip.size:
            rs, vs = r_m[ip], v_m[ip]
            term = rs[:, 2] < self.h_terminal
            rf = np.zeros((ip.size, 3))
            rf[:, 2] = np.where(term, -0.5, 0.0)
            vf = np.zeros((ip.size, 3))
            vf[:, 2] = np.where(term, -1.5, -3.0)
            gs = g_vec[ip]
            t_go = self._t_go(rs, vs, rf, vf) * self.t_go_scale
            T_z = np.maximum(t_go, 0.5)
            g_z = gs[:, 2]
            zem_z = rf[:, 2] - rs[:, 2] - vs[:, 2] * T_z - 0.5 * g_z * T_z**2
            zev_z = vf[:, 2] - vs[:, 2] - g_z * T_z
            ac = a_cmd[ip]
            ac[:] = 0.0
            ac[:, 2] = 6.0 * zem_z / T_z**2 - 2.0 * zev_z / T_z
            # Bounded-approach-speed lateral servo
            # (PoweredDescentBase._lateral_servo): approach-velocity demand
            # (rf - r)/tau_app capped at v_lat_max and faded to zero below
            # ~100 m, tracked with gain kv_lat.  Keeps the thrust azimuth
            # steady toward the pad -- the classic 6*ZEM/T^2 - 2*ZEV/T
            # lateral law demands ~180 deg azimuth slews the slow TVC
            # attitude loop cannot track.
            tau_app = np.clip(0.35 * t_go, 4.0, 12.0)
            fade = np.clip(rs[:, 2] / 100.0, 0.0, 1.0)
            for k in (0, 1):
                v_des = (np.clip((rf[:, k] - rs[:, k]) / tau_app,
                                 -self.v_lat_max, self.v_lat_max) * fade)
                ac[:, k] = self.kv_lat * (v_des - vs[:, k])
            # _shape_accel: altitude-tapered lateral tilt cap
            # (|a_lat| <= a_z tan(cap_eff), cap_eff: 35 deg high -> ~3 deg
            # below 150 m), then clamp |a_cmd| at 0.95 T_max/m.
            cap_eff = np.maximum(
                self.tilt_cap_min,
                self.tilt_cap * np.clip(rs[:, 2] / self.h_cap_ref, 0.0, 1.0))
            a_lat = np.hypot(ac[:, 0], ac[:, 1])
            cap = np.tan(cap_eff) * np.maximum(ac[:, 2], 0.0)
            s_lat = np.where((a_lat > cap) & (cap > 0.0),
                             cap / np.maximum(a_lat, 1e-9), 1.0)
            ac[:, 0] *= s_lat
            ac[:, 1] *= s_lat
            amax = (0.95 * T_max[ip]
                    / np.maximum(m[ip], 1e-9))
            am = _norm(ac)
            s_am = np.where(am > amax, amax / np.maximum(am, 1e-9), 1.0)
            ac *= s_am[:, None]
            # _filter_lateral (PoweredDescentBase): first-order low-pass on
            # the lateral channels only, tau = 1.5 s.  Each run's filter
            # state initializes to the *unfiltered* command on its first
            # powered tick (upstream `_a_filt is None` init -> passthrough).
            alpha_f = min(self.dec * self.dt / self.lat_filt_tau, 1.0)
            af = a_filt[ip]
            is_new = newly[ip]
            af[is_new] = ac[is_new, :2]
            cont = ~is_new
            af[cont] += alpha_f * (ac[cont, :2] - af[cont])
            a_filt[ip] = af
            ac[:, 0] = af[:, 0]
            ac[:, 1] = af[:, 1]
            a_cmd[ip] = ac
        self._last_acmd = a_cmd

        # --- allocation: thrust magnitude + desired attitude ---
        a_mag = _norm(a_cmd)
        T_req = m * a_mag
        T_cmd = np.clip(T_req, T_min, np.maximum(T_max, 1.0))
        thr_cmd[:] = np.where(
            powered,
            np.clip(T_cmd / np.maximum(T_max, 1.0), 0.0, 1.0), 0.0)
        # Desired thrust direction: body +x onto a_cmd, roll-preserving.
        dir_des = np.where(a_mag[:, None] > 1e-3,
                           a_cmd / np.maximum(a_mag[:, None], 1e-9),
                           np.broadcast_to(_ZH, (n, 3)))
        x_b = _qrot(q, np.broadcast_to(_XB, (n, 3)))
        q_des_pow = _qmul(_q_from_two(x_b, dir_des), q)
        # Coast: retrograde blended to <= tilt_max off +z (PoweredDescentBase).
        vmag = _norm(v_m)[:, None]
        des = np.where(vmag > 1.0, -v_m / np.maximum(vmag, 1e-9),
                       np.broadcast_to(_ZH, (n, 3)))
        cos_t = np.clip(np.sum(des * _ZH, axis=1), -1.0, 1.0)
        over = cos_t < np.cos(self.tilt_max)
        if over.any():
            io = np.flatnonzero(over)
            horiz = des[io] - cos_t[io, None] * _ZH
            hn = _norm(horiz)
            okh = hn > 1e-9
            des_over = (_ZH * np.cos(self.tilt_max)
                        + (horiz / np.maximum(hn[:, None], 1e-9))
                        * np.sin(self.tilt_max))
            # Degenerate case (desired ~ -z_B): upstream falls back to
            # upright, not a 25 deg tilt toward an arbitrary azimuth.
            des_over[~okh] = _ZH
            des[io] = des_over
        q_des_coast = _q_from_two(np.broadcast_to(_XB, (n, 3)), des)
        q_des = np.where(powered[:, None], q_des_pow, q_des_coast)

        # --- attitude law (GeometricController) ---
        e = _qerr(q, q_des)
        wxJw = _cross3(w, Jd * w)
        # e is the rotation vector taking current -> desired attitude, so
        # the proportional term drives omega *along* +e (GeometricController).
        M_des = self.k_r * Jd * e - self.k_w * Jd * w + wxJw
        T_auth = np.maximum(thr_cmd * T_max, T_min)
        l_arm = np.maximum(cg - self.gimbal_x, 0.5)
        den = np.maximum(T_auth * l_arm, 1e-6)
        M_lim = np.maximum(T_auth * l_arm * np.sin(self.gimbal_max), 1e-9)
        scale = np.maximum(
            1.0, np.maximum(np.abs(M_des[:, 1]) / M_lim,
                            np.abs(M_des[:, 2]) / M_lim))
        M2 = M_des / scale[:, None]
        d_y = -np.arcsin(np.clip(M2[:, 1] / den, -1.0, 1.0))
        d_z = -np.arcsin(np.clip(M2[:, 2] / den, -1.0, 1.0))
        cmd = np.stack([d_y, d_z], axis=1)
        if self.disp.control_noise > 0.0:
            cmd = cmd + self.disp.control_noise * rng.standard_normal(cmd.shape)
        gim_cmd[:] = np.where(
            powered[:, None],
            np.clip(cmd, -self.gimbal_max, self.gimbal_max), 0.0)
        # Coast: pseudo-RCS PD (ControllerBase._rcs_cmd).
        M_rcs = self.rcs_k_r * Jd * e - self.rcs_k_w * Jd * w
        M_rcs = np.clip(M_rcs, -self.rcs_max, self.rcs_max)
        rcs_cmd[:] = np.where((active & ~powered)[:, None], M_rcs, 0.0)
        # ActuatorSuite.step also jitters the throttle command by
        # sigma = control_noise each step; applied to the (ZOH) powered
        # command only -- non-powered runs keep thr_cmd = 0 so the batch
        # does not reproduce the scalar sim's coast sputter quirk (a
        # positive lagged throttle value would clamp up to the floor).
        if self.disp.control_noise > 0.0 and powered.any():
            thr_cmd[powered] += (self.disp.control_noise
                                 * rng.standard_normal(int(powered.sum())))
        return powered

    # ------------------------------------------------------------------
    @staticmethod
    def _t_go(r, v, rf, vf):
        """Vectorized kinematic t_go estimate (port of ``_t_go_estimate``).

        Solves ``dz + 0.5 (v_z + v_fz) t = 0`` -- descent time under
        constant net acceleration toward a possibly below-ground target;
        falls back to ``dz / |v_fz|`` when not descending.  Clamped to
        [0.5, 120] s.
        """
        dz = r[:, 2] - rf[:, 2]
        vs = v[:, 2] + vf[:, 2]
        t_go = np.where(
            (dz > 0.0) & (vs < -1e-3),
            -2.0 * dz / np.where(vs < -1e-3, vs, -1e-3),
            np.where(dz > 0.0,
                     dz / np.maximum(-vf[:, 2], 0.5), 1.0))
        return np.clip(t_go, 0.5, 120.0)
