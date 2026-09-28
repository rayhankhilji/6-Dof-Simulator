"""Powered-descent guidance laws: ZEM/ZEV and Apollo-style polynomial.

For point-mass dynamics ``r_ddot = u + g`` the ZEM/ZEV law computes the
commanded thrust acceleration

    ZEM = r_f - r - v t_go - 0.5 g t_go^2
    ZEV = v_f - v - g t_go
    u   = 6 ZEM / t_go^2 - 2 ZEV / t_go

t_go is the kinematic landing time for the vertical channel: the time to
reach ``r_fz`` at ``v_fz`` under constant net acceleration,
``t_go = -2 (z - z_f) / (v_z + v_fz)``. This estimate is self-consistent
under continuous replanning -- the ZEM/ZEV command then stays inside the
thrust bounds and counts down with the actual descent (unlike a
smallest-feasible-|u| search, which returns hover-like commands that never
brake, or stale huge values once the window is missed).

Polynomial guidance fits ``u(t) = c0 + c1 t + c2 t^2`` per axis through the
boundary conditions (r, v now -> r_f, v_f, a_f at t_go) and commands u(0).
"""

from __future__ import annotations

import numpy as np

from ..control_types import GuidanceCommand
from ..navigation.base import NavState
from .base import PoweredDescentBase


def _t_go_estimate(r: np.ndarray, v: np.ndarray, rf, vf, g_vec=None,
                   a_lim=None, info=None) -> float:
    """Kinematic landing time-to-go for ZEM/ZEV-style replanning.

    Solves ``(z - z_f) + 0.5 (v_z + v_fz) t_go = 0`` -- the descent time
    under constant net acceleration to the (possibly below-ground) target.
    Falling back to ``dz / |v_fz|`` when not descending keeps the command
    pointed upward and descending. Clamped to [0.5, 120] s.
    """
    dz = float(r[2] - rf[2])
    vs = float(v[2] + vf[2])
    if dz > 0.0 and vs < -1e-3:
        t_go = -2.0 * dz / vs
    elif dz > 0.0:
        t_go = dz / max(-vf[2], 0.5)
    else:
        t_go = 1.0
    return float(np.clip(t_go, 0.5, 120.0))


# Backward-compatible alias (previous name used a fragile bisection search).
_t_go_bisect = _t_go_estimate


def _alt_tilt_cap(h: float, cap_rad: float, h_ref: float = 150.0,
                  min_rad: float = np.radians(4.0)) -> float:
    """Altitude-tapered tilt cap: full cap high up, ~vertical low down."""
    f = float(np.clip(h / max(h_ref, 1e-9), 0.0, 1.0))
    return max(min_rad, cap_rad * f)


def _shape_accel(a_cmd: np.ndarray, info: dict, tilt_cap_rad: float) -> np.ndarray:
    """Keep the commanded thrust accel inside what a tail-sitter can do.

    1. Tilt cap: ``|a_lat| <= a_z tan(tilt_cap)`` -- bounds the attitude
       excursion the controller must chase (vertical braking keeps
       priority; without this, late-divert demands can require ~90 deg
       of tilt which the gimbal cannot track).
    2. Magnitude clamp at 95% of ``T_max/m``.
    """
    a_cmd = np.asarray(a_cmd, dtype=float).copy()
    a_lat = float(np.hypot(a_cmd[0], a_cmd[1]))
    cap = np.tan(tilt_cap_rad) * max(a_cmd[2], 0.0)
    if a_lat > cap > 0.0:
        s = cap / a_lat
        a_cmd[0] *= s
        a_cmd[1] *= s
    amax = 0.95 * info["thrust_max"] / max(info["mass"], 1e-9)
    am = np.linalg.norm(a_cmd)
    if am > amax:
        a_cmd *= amax / am
    return a_cmd


class ZEMZEVGuidance(PoweredDescentBase):
    """Trajectory-shaping powered-descent guidance (ZEM/ZEV)."""

    def __init__(self, h_terminal: float = 30.0, tilt_cap_deg: float = 35.0,
                 h_cap_ref: float = 150.0, tilt_cap_min_deg: float = 3.0,
                 t_go_scale: float = 1.15) -> None:
        super().__init__(h_terminal)
        self.tilt_cap = np.radians(tilt_cap_deg)
        self.tilt_cap_min = np.radians(tilt_cap_min_deg)
        self.h_cap_ref = float(h_cap_ref)
        # Brake-margin factor: a slightly *longer* t_go makes the plan brake
        # harder early (more of the descent near hover), leaving margin for
        # actuator/attitude lag so touchdown lands near vf, not hot.
        self.t_go_scale = float(t_go_scale)
        self.v_lat_max = 25.0   # bounded approach speed [m/s]
        self.kv_lat = 0.45      # lateral velocity-servo gain [1/s]

    def _eff_tilt_cap(self, h: float) -> float:
        """Lateral aggressiveness tapers to ~vertical near the ground."""
        f = float(np.clip(h / max(self.h_cap_ref, 1e-9), 0.0, 1.0))
        return max(self.tilt_cap_min, self.tilt_cap * f)

    def compute(self, t, nav: NavState, info: dict) -> GuidanceCommand:
        if not self._maybe_ignite(nav, info):
            return self._coast_cmd(nav, info)

        g_vec = np.array([0.0, 0.0, -info["g"]])
        rf, vf, phase = self._targets(nav.r_I)
        t_go = _t_go_estimate(nav.r_I, nav.v_I, rf, vf, g_vec) * self.t_go_scale
        T_z = max(t_go, 0.5)

        a_cmd = np.zeros(3)
        zem_z = rf[2] - nav.r_I[2] - nav.v_I[2] * T_z - 0.5 * g_vec[2] * T_z**2
        zev_z = vf[2] - nav.v_I[2] - g_vec[2] * T_z
        a_cmd[2] = 6.0 * zem_z / T_z**2 - 2.0 * zev_z / T_z
        # Bounded-approach-speed lateral servo (see PoweredDescentBase).
        a_cmd[:2] = self._lateral_servo(nav, rf, t_go,
                                        self.v_lat_max, self.kv_lat)
        a_cmd = _shape_accel(a_cmd, info, self._eff_tilt_cap(nav.r_I[2]))
        a_cmd = self._filter_lateral(a_cmd, t)

        return GuidanceCommand(
            accel_cmd_I=a_cmd, throttle=self._throttle_hint(a_cmd, info),
            engine_on=True, q_des=None, phase=phase,
            r_ref=rf.copy(), v_ref=vf.copy(), t_go=t_go,
        )


class PolynomialGuidance(PoweredDescentBase):
    """Apollo/E-guidance: quadratic thrust-acceleration profile to t_go.

    At each call it solves the boundary-value problem for
    ``u(tau) = c0 + c1 tau + c2 tau^2`` (per axis) hitting ``r_f, v_f`` at
    ``t_go`` with terminal acceleration ``a_f``; the issued command is u(0).
    """

    def __init__(self, h_terminal: float = 30.0, t_go_init: float | None = None,
                 a_f: np.ndarray | None = None, tilt_cap_deg: float = 35.0,
                 t_go_scale: float = 1.15) -> None:
        super().__init__(h_terminal)
        self.t_go = t_go_init
        self.a_f = np.asarray(a_f if a_f is not None else np.array([0, 0, 0.98]), dtype=float)
        self.tilt_cap = np.radians(tilt_cap_deg)
        # Brake-margin factor (same convention as ZEMZEVGuidance).
        self.t_go_scale = float(t_go_scale)
        self._t_last: float | None = None
        self._C: np.ndarray | None = None

    def solve_coeffs(self, r, v, rf, vf, t_go, g_vec):
        """Quadratic coefficients (c0,c1,c2) per axis; returns (3,3) array.

        Boundary conditions (u is thrust acceleration, dynamics r'' = u + g):
            v(T) = v + g T + c0 T + c1 T^2/2 + c2 T^3/3 = vf
            r(T) = r + v T + g T^2/2 + c0 T^2/2 + c1 T^3/6 + c2 T^4/12 = rf
            u(T) = c0 + c1 T + c2 T^2 = a_f
        """
        T = t_go
        A = np.array([
            [T, T**2 / 2.0, T**3 / 3.0],
            [T**2 / 2.0, T**3 / 6.0, T**4 / 12.0],
            [1.0, T, T**2],
        ])
        b = np.stack([
            vf - v - g_vec * T,
            rf - r - v * T - 0.5 * g_vec * T**2,
            self.a_f,
        ], axis=0)  # rows: equations (v, r, a); cols: axes
        return np.linalg.solve(A, b)  # (3 coeffs, 3 axes)

    def _linear_coeffs(self, r, v, rf, vf, t_go, g_vec):
        """Linear thrust-accel profile ``u = c0 + c1 tau`` per axis.

        Fits the two boundary conditions ``v(T)=vf``, ``r(T)=rf``; the
        issued coefficient c0 equals the ZEM/ZEV command
        ``6 ZEM/T^2 - 2 ZEV/T`` -- which is exactly the *self-consistent*
        replanning command: a quadratic profile pinned additionally at
        ``u(T)=a_f`` leaves u(0) under-constrained and degenerates to a
        "coast then spike" plan whose commanded accel stays near zero
        while the vehicle falls (verified in closed loop). ``solve_coeffs``
        retains the quadratic variant for boundary-value verification and
        plan projection.
        """
        T = t_go
        A = np.array([[T, 0.5 * T * T],
                      [0.5 * T * T, T**3 / 6.0]])
        b = np.stack([vf - v - g_vec * T,
                      rf - r - v * T - 0.5 * g_vec * T * T], axis=0)
        return np.linalg.solve(A, b)  # (2 coeffs, 3 axes)

    def plan_trajectory(self, r, v, g_vec, n: int = 60):
        """Predicted (r, v) under the fitted profile, for plan logging."""
        C = self._C
        tau = np.linspace(0.0, self.t_go, n)[:, None]
        u = C[0] + C[1] * tau
        if C.shape[0] > 2:
            u = u + C[2] * tau**2
        v_p = v + g_vec * tau + C[0] * tau + C[1] * tau**2 / 2.0
        r_p = (r + v * tau + 0.5 * g_vec * tau**2 + C[0] * tau**2 / 2.0
               + C[1] * tau**3 / 6.0)
        if C.shape[0] > 2:
            v_p = v_p + C[2] * tau**3 / 3.0
            r_p = r_p + C[2] * tau**4 / 12.0
        return {"tau": tau[:, 0], "r": r_p.T, "v": v_p.T}

    def compute(self, t, nav: NavState, info: dict) -> GuidanceCommand:
        if not self._maybe_ignite(nav, info):
            return self._coast_cmd(nav, info)

        g_vec = np.array([0.0, 0.0, -info["g"]])
        rf, vf, phase = self._targets(nav.r_I)
        # Recompute the consistent descent time each call (same estimate as
        # ZEM/ZEV) so mid-flight target changes stay feasible; the brake
        # margin factor lengthens t_go slightly so the plan brakes early.
        self.t_go = _t_go_estimate(nav.r_I, nav.v_I, rf, vf, g_vec) * self.t_go_scale
        self._t_last = t
        T = max(self.t_go, 0.5)

        self._C = self._linear_coeffs(nav.r_I, nav.v_I, rf, vf, T, g_vec)
        a_cmd = self._C[0].copy()  # u(tau=0)
        # Bounded-approach-speed lateral servo (see PoweredDescentBase).
        a_cmd[:2] = self._lateral_servo(nav, rf, T)
        a_cmd = _shape_accel(a_cmd, info,
                             _alt_tilt_cap(nav.r_I[2], self.tilt_cap))
        a_cmd = self._filter_lateral(a_cmd, t)

        return GuidanceCommand(
            accel_cmd_I=a_cmd, throttle=self._throttle_hint(a_cmd, info),
            engine_on=True, q_des=None, phase=phase,
            r_ref=rf.copy(), v_ref=vf.copy(), t_go=self.t_go,
        )
