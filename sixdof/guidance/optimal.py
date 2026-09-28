"""Fuel-optimal powered-descent guidance (receding-horizon SLSQP).

3-DOF point-mass direct multiple shooting over N=25 nodes. Decision
variables ``z = [u_0..u_{N-1} (3N), t_f]``; forward-Euler dynamics
``r' = v``, ``v' = u + g`` with mass from the rocket equation
``m' = -m |u| / (Isp g0)``.

Constraints: terminal ``r(t_f)=r_f``, ``v(t_f)=v_f``; per-node thrust bounds
``T_min/m <= |u| <= T_max/m`` (smooth |u| = sqrt(u^2 + eps)); glide-slope cone
``sqrt(x^2+y^2) <= z tan(75 deg)``; thrust-tilt cone ``u_z >= |u| cos(th)``,
th = 30 deg tightening to 10 deg below 50 m. Cost ``sum |u_k| dt`` (~ fuel).

Replans every ``replan_period`` warm-started from the shifted previous plan;
on failure it falls back to the ZEM/ZEV command (``fallback_count`` tracks
how often). Between replans the issued acceleration is the planned u(t)
plus a PD correction toward the planned state.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from ..control_types import GuidanceCommand
from ..navigation.base import NavState
from .base import PoweredDescentBase
from .descent import ZEMZEVGuidance, _alt_tilt_cap, _shape_accel, _t_go_estimate

_EPS = 1e-6


class OptimalGuidance(PoweredDescentBase):
    """Receding-horizon fuel-optimal guidance."""

    def __init__(
        self,
        n_nodes: int = 20,
        replan_period: float = 1.5,
        maxiter: int = 40,
        gamma_gs_deg: float = 75.0,
        tilt_max_deg: float = 20.0,
        tilt_terminal_deg: float = 8.0,
        h_terminal: float = 30.0,
        kp: float = 0.25,
        kd: float = 0.8,
        isp: float = 311.0,
        g0: float = 9.80665,
    ) -> None:
        super().__init__(h_terminal)
        self.N = int(n_nodes)
        self.replan_period = float(replan_period)
        self.maxiter = int(maxiter)
        self.gamma_gs = np.radians(gamma_gs_deg)
        self.tilt_hi = np.radians(tilt_max_deg)
        self.tilt_lo = np.radians(tilt_terminal_deg)
        self.kp, self.kd = kp, kd
        self.isp = isp
        self.g0 = g0

        self.fallback = ZEMZEVGuidance(h_terminal=h_terminal)
        self.fallback_count = 0
        self.solve_times: list[float] = []
        self.fuel_predicted: float | None = None

        self._last_replan_t: float | None = None
        self._plan: dict | None = None      # t, r(3,N+1), v(3,N+1), u(3,N), m(N+1)
        self._z_prev: np.ndarray | None = None
        self._prop_avail: float = 1e9       # set per-solve from info

    # ------------------------------------------------------------------
    # Optimal-control solve
    # ------------------------------------------------------------------
    def _simulate(self, z, r0, v0, m0, g_vec, sens: bool = False):
        """Rollout; optionally also propagate sensitivities ``dX/dz``.

        X = (r, v, m). P_k = dX_k/dz is (7, 3N+1). Returns
        ``(r, v, m, u, dt[, P])`` where P is the stacked (N+1, 7, 3N+1).
        """
        N = self.N
        u = z[: 3 * N].reshape(N, 3)
        tf = np.clip(z[3 * N], 1e-3, 1e4)
        dt = tf / N
        c = self.isp * self.g0
        # Mass floor: never let the rocket equation drive m below dry mass
        # (also guards the T/m constraint denominators at extreme iterates).
        m_floor = max(m0 - self._prop_avail, 1.0)
        r = np.empty((N + 1, 3)); v = np.empty((N + 1, 3)); m = np.empty(N + 1)
        r[0], v[0], m[0] = r0, v0, m0
        umag = np.sqrt(np.sum(u * u, axis=1) + _EPS)
        P = np.zeros((N + 1, 7, 3 * N + 1)) if sens else None
        for k in range(N):
            decay = np.exp(-min(umag[k] * dt / c, 50.0))
            r[k + 1] = r[k] + v[k] * dt
            v[k + 1] = v[k] + (u[k] + g_vec) * dt
            m[k + 1] = max(m[k] * decay, m_floor)
            if sens:
                # A_k = dX_{k+1}/dX_k (7x7), B_u = dX_{k+1}/du_k (7x3),
                # b_tf = dX_{k+1}/dtf (7,).
                A = np.eye(7)
                A[0:3, 3:6] = np.eye(3) * dt
                A[6, 6] = decay
                B = np.zeros((7, 3))
                B[3:6] = np.eye(3) * dt
                B[6] = -m[k] * decay * dt / c * u[k] / umag[k]
                btf = np.zeros(7)
                btf[0:3] = v[k] / N
                btf[3:6] = (u[k] + g_vec) / N
                btf[6] = -m[k] * decay * umag[k] / (c * N)
                P[k + 1] = A @ P[k]
                P[k + 1, :, 3 * k: 3 * k + 3] += B
                P[k + 1, :, -1] += btf
        if sens:
            return r, v, m, u, dt, P
        return r, v, m, u, dt

    def _solve(self, nav: NavState, info: dict, rf, vf, t_now: float = 0.0):
        N = self.N
        r0, v0, m0 = nav.r_I.copy(), nav.v_I.copy(), info["mass"]
        g_vec = np.array([0.0, 0.0, -info["g"]])
        T_max, T_min = info["thrust_max"], info["thrust_min"]

        self._prop_avail = float(info.get("prop_remaining", m0))
        # Initial guess: kinematic landing t_go + constant-acceleration
        # descent profile (per-axis accel that meets the position BC).
        t_go0 = _t_go_estimate(r0, v0, rf, vf, g_vec)
        if self._z_prev is not None:
            # Warm start: shift the previous control history forward by the
            # elapsed time (in nodes) so the guess stays feasible.
            z_prev = self._z_prev
            dt_prev = z_prev[3 * N] / N
            elapsed = (t_now - self._last_replan_t) if self._last_replan_t is not None else 0.0
            shift = int(round(elapsed / max(dt_prev, 1e-9)))
            u_prev = z_prev[: 3 * N].reshape(N, 3)
            if 0 < shift < N:
                u_shift = np.vstack([u_prev[shift:], np.tile(u_prev[-1], (shift, 1))])
            elif shift >= N:
                u_shift = np.tile(u_prev[-1], (N, 1))
            else:
                u_shift = u_prev
            tf_guess = float(np.clip(z_prev[3 * N] - elapsed, 2.0, 120.0))
            z0 = np.concatenate([u_shift.ravel(), [tf_guess]])
        else:
            u0 = 2.0 * (rf - r0 - v0 * t_go0) / t_go0**2 - g_vec
            u0[2] = np.clip(u0[2], 0.0, info["thrust_max"] / m0)
            z0 = np.concatenate([np.tile(u0, N), [t_go0]])

        def cost(z):
            u = z[: 3 * N].reshape(N, 3)
            dt = z[3 * N] / N
            umag = np.sqrt(np.sum(u * u, axis=1) + _EPS)
            return np.sum(umag) * dt

        def cost_jac(z):
            u = z[: 3 * N].reshape(N, 3)
            dt = z[3 * N] / N
            umag = np.sqrt(np.sum(u * u, axis=1) + _EPS)
            j = np.zeros(3 * N + 1)
            j[: 3 * N] = (u / umag[:, None]).ravel() * dt
            j[3 * N] = np.sum(umag) / N
            return j

        # Force a marginally *longer* flight time than the kinematic
        # constant-deceleration estimate: pure fuel-optimal descents brake
        # as late as possible, which leaves the 6-DOF loop no margin for
        # attitude/actuator lag.
        tf_lo = max(2.0, 1.15 * t_go0)

        def eq_cons(z):
            r, v, m, u, dt = self._simulate(z, r0, v0, m0, g_vec)
            return np.concatenate([r[-1] - rf, v[-1] - vf])

        def eq_jac(z):
            _, _, _, _, _, P = self._simulate(z, r0, v0, m0, g_vec, sens=True)
            return P[N, 0:6, :]

        m_floor = max(m0 - self._prop_avail, 1.0)

        def ineq_cons(z):
            r, v, m, u, dt = self._simulate(z, r0, v0, m0, g_vec)
            umag = np.sqrt(np.sum(u * u, axis=1) + _EPS)
            mk = np.maximum(m[:-1], m_floor)  # mass at each node's start
            cons = [umag - T_min / mk,                   # >=0 min thrust
                    T_max / mk - umag,                   # >=0 max thrust
                    np.array([z[3 * N] - tf_lo]),        # tf >= gentler floor
                    np.array([120.0 - z[3 * N]])]        # tf <= 120 s
            # Glide slope on interior nodes (k>=1).
            lat = np.hypot(r[1:, 0], r[1:, 1])
            cons.append(r[1:, 2] * np.tan(self.gamma_gs) - lat)
            # Thrust tilt: u_z >= |u| cos(theta_max(z)).
            theta = np.where(r[:-1, 2] < 50.0, self.tilt_lo, self.tilt_hi)
            cons.append(u[:, 2] - umag * np.cos(theta))
            return np.concatenate(cons)

        def ineq_jac(z):
            r, v, m, u, dt, P = self._simulate(z, r0, v0, m0, g_vec, sens=True)
            N = self.N
            nv = 3 * N + 1
            umag = np.sqrt(np.sum(u * u, axis=1) + _EPS)
            udir = u / umag[:, None]
            mk = np.maximum(m[:-1], m_floor)
            tan_g = np.tan(self.gamma_gs)
            theta = np.where(r[:-1, 2] < 50.0, self.tilt_lo, self.tilt_hi)
            rows = []
            # min thrust: |u_k| - T_min/m_k
            Jmin = np.zeros((N, nv))
            # max thrust: T_max/m_k - |u_k|
            Jmax = np.zeros((N, nv))
            for k in range(N):
                Jmin[k, 3 * k: 3 * k + 3] = udir[k]
                Jmin[k] += (T_min / mk[k] ** 2) * P[k, 6]
                Jmax[k, 3 * k: 3 * k + 3] = -udir[k]
                Jmax[k] += (-T_max / mk[k] ** 2) * P[k, 6]
            rows += [Jmin, Jmax]
            Jb = np.zeros((2, nv)); Jb[0, -1] = 1.0; Jb[1, -1] = -1.0
            rows.append(Jb)
            # glide slope at nodes k=1..N
            Jgs = np.zeros((N, nv))
            for k in range(1, N + 1):
                lat = np.hypot(r[k, 0], r[k, 1])
                Jgs[k - 1] = tan_g * P[k, 2]
                if lat > 1e-9:
                    Jgs[k - 1] -= (r[k, 0] * P[k, 0] + r[k, 1] * P[k, 1]) / lat
            rows.append(Jgs)
            # thrust tilt: u_z - |u| cos(theta)
            Jt = np.zeros((N, nv))
            for k in range(N):
                Jt[k, 3 * k + 2] = 1.0
                Jt[k, 3 * k: 3 * k + 3] -= np.cos(theta[k]) * udir[k]
            rows.append(Jt)
            return np.vstack(rows)

        t0 = time.perf_counter()
        res = minimize(
            cost, z0, method="SLSQP", jac=cost_jac,
            constraints=[
                {"type": "eq", "fun": eq_cons, "jac": eq_jac},
                {"type": "ineq", "fun": ineq_cons, "jac": ineq_jac},
            ],
            bounds=[(None, None)] * (3 * N) + [(tf_lo, 120.0)],
            options={"maxiter": self.maxiter, "ftol": 1e-6},
        )
        self.solve_times.append(time.perf_counter() - t0)

        # Accept on constraint violation, not res.success alone: SLSQP often
        # reports "iteration limit reached" while sitting on a feasible,
        # near-optimal point.
        eq_viol = np.max(np.abs(eq_cons(res.x)))
        iq_viol = -np.min(ineq_cons(res.x))  # >0 means violated
        ok = (eq_viol < 1e-3 * max(1.0, np.linalg.norm(r0))
              and iq_viol < 0.5)
        if not ok:
            return None
        r, v, m, u, dt = self._simulate(res.x, r0, v0, m0, g_vec)
        self._z_prev = res.x
        ts = np.linspace(0.0, res.x[3 * N], N + 1)
        self._plan = {"t": ts, "r": r.T, "v": v.T, "u": u.T, "m": m, "dt": dt}
        self.fuel_predicted = m0 - m[-1]
        return res

    # ------------------------------------------------------------------
    def plan_trajectory(self) -> dict | None:
        """Most recent solved plan: {t, r, v, u, m, dt} or None."""
        return self._plan

    def _interp_plan(self, tau: float):
        """Interpolated planned (r, v, u) at time-since-plan ``tau``."""
        p = self._plan
        ts = p["t"]
        j = np.searchsorted(ts, tau) - 1
        j = int(np.clip(j, 0, self.N - 1))
        frac = np.clip((tau - ts[j]) / max(ts[j + 1] - ts[j], 1e-9), 0.0, 1.0)
        r = (1 - frac) * p["r"][:, j] + frac * p["r"][:, j + 1]
        v = (1 - frac) * p["v"][:, j] + frac * p["v"][:, j + 1]
        u = p["u"][:, j]
        return r, v, u

    # ------------------------------------------------------------------
    def compute(self, t, nav: NavState, info: dict) -> GuidanceCommand:
        if not self._maybe_ignite(nav, info):
            return self._coast_cmd(nav, info)

        rf, vf, phase = self._targets(nav.r_I)

        # Low altitude / terminal: hand off to the ZEM/ZEV law -- the SLSQP
        # plan becomes ill-conditioned low down (t_f bound pinches) and its
        # late lateral demands exceed what the slow attitude loop can
        # track; the closed-form law lands cleanly.
        if phase == "terminal" or nav.r_I[2] < 800.0:
            self.fallback.ignited = True
            fb = self.fallback.compute(t, nav, info)
            fb.phase = phase if phase == "terminal" else phase + "_handoff"
            return fb

        # Replan on a fixed cadence only -- never retry inside the period
        # (a failed solve must not trigger an SLSQP run every control step).
        need_replan = (
            self._last_replan_t is None
            or t - self._last_replan_t >= self.replan_period
        )
        if need_replan:
            res = self._solve(nav, info, rf, vf, t_now=t)
            self._last_replan_t = t
            if res is None:
                self.fallback_count += 1
                self._plan = None
            else:
                self._plan_t0 = t
        if self._plan is None:
            self.fallback.ignited = True
            fb = self.fallback.compute(t, nav, info)
            fb.phase = phase + "_fallback"
            return fb

        tau = t - getattr(self, "_plan_t0", self._last_replan_t)
        r_p, v_p, u_p = self._interp_plan(tau)
        a_cmd = u_p + self.kp * (r_p - nav.r_I) + self.kd * (v_p - nav.v_I)
        a_cmd = _shape_accel(a_cmd, info,
                             _alt_tilt_cap(nav.r_I[2], np.radians(20.0),
                                           h_ref=250.0, min_rad=np.radians(4.0)))
        a_cmd = self._filter_lateral(a_cmd, t)

        return GuidanceCommand(
            accel_cmd_I=a_cmd, throttle=self._throttle_hint(a_cmd, info),
            engine_on=True, q_des=None, phase=phase,
            r_ref=r_p.copy(), v_ref=v_p.copy(),
            t_go=max(self._plan["t"][-1] - tau, 0.0),
        )
