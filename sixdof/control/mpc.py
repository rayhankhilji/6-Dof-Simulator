"""Linear MPC lateral-channel controller (FISTA-projected box QP).

Per horizontal channel (East via gimbal_y, North via gimbal_z) the 5-state
model is discretized at the control period and regulated over N=40 steps
(0.8 s). Cost:

    sum (x - x_ref)' Q (x - x_ref) + R d^2 + Rd (d_k - d_{k-1})^2

with terminal cost from the DARE and box constraints ``|d_cmd| <=
gimbal_max``. The condensed QP is solved with FISTA (40 iterations,
warm-started by shifting the previous solution). Pure numpy/scipy.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.linalg import solve_discrete_are, expm

from ..control_types import ControlCommand
from ..math.quaternion import quat_rotate
from ..navigation.base import NavState
from .allocation import (
    lateral_linear_model,
    lateral_rates,
    lateral_tilts,
    lateral_tilts_from_dir,
)
from .base import ControllerBase


def _discretize(A, B, dt):
    """Exact ZOH discretization of (A, B) over ``dt``."""
    n = A.shape[0]
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = A
    M[:n, n:] = B
    Md = expm(M * dt)
    return Md[:n, :n], Md[:n, n:]


class _ChannelMPC:
    """Condensed QP + FISTA for one lateral channel."""

    def __init__(self, dt, n_horizon, Q, R, R_d, u_max):
        self.dt = dt
        self.N = n_horizon
        self.Q = Q
        self.R = float(R)
        self.R_d = float(R_d)
        self.u_max = float(u_max)
        self._H = None
        self._Sx = None
        self._key = None
        self._u_prev = np.zeros(n_horizon)
        self.iterations = 0
        self.solve_time = 0.0

    def setup(self, A, B):
        Ad, Bd = _discretize(A, B, self.dt)
        n = Ad.shape[0]
        N = self.N
        # Condensed dynamics: X = Sx x0 + Su U.
        Sx = np.zeros((n * N, n))
        Su = np.zeros((n * N, N))
        A_pow = np.eye(n)
        for k in range(N):
            A_pow = A_pow @ Ad
            Sx[k * n:(k + 1) * n] = A_pow
            for j in range(k + 1):
                Su[k * n:(k + 1) * n, j] = (np.linalg.matrix_power(Ad, k - j) @ Bd)[:, 0]
        # Q bar with DARE terminal cost. With (near-)zero state weights the
        # symplectic pencil can be singular; fall back to a regularized Q
        # and ultimately to Q itself (finite-horizon-only cost).
        for q_try in (self.Q, self.Q + 1e-3 * np.eye(n), self.Q + 1e-2 * np.eye(n)):
            try:
                P_term = solve_discrete_are(Ad, Bd, q_try, np.array([[self.R]]))
                break
            except np.linalg.LinAlgError:
                continue
        else:
            P_term = self.Q
        Qb = np.kron(np.eye(N), self.Q)
        Qb[(N - 1) * n:, (N - 1) * n:] = P_term
        # Delta-u penalty: D u with first row referencing previous input.
        D = np.eye(N)
        D[1:, :-1] -= np.eye(N - 1)
        H = Su.T @ Qb @ Su + self.R * np.eye(N) + self.R_d * (D.T @ D)
        H = 0.5 * (H + H.T)
        self._Su, self._Sx, self._Qb, self._D = Su, Sx, Qb, D
        self._H = H
        self._L = np.linalg.norm(H, 2)
        self.iter_setup = True

    def solve(self, x0, x_ref, u_applied):
        """FISTA box-constrained solve; returns optimal d_cmd sequence."""
        t0 = time.perf_counter()
        N = self.N
        Xref = np.tile(x_ref, N)
        f = self._Su.T @ self._Qb @ (self._Sx @ x0 - Xref)
        f = f + self.R_d * (self._D.T @ np.concatenate([[-u_applied], np.zeros(N - 1)]))
        # Warm start: shift previous solution.
        u = np.concatenate([self._u_prev[1:], [self._u_prev[-1]]])
        v = u.copy()
        tk = 1.0
        iters = 0
        for _ in range(40):
            iters += 1
            grad = self._H @ v + f
            u_new = np.clip(v - grad / self._L, -self.u_max, self.u_max)
            t_new = 0.5 * (1 + np.sqrt(1 + 4 * tk * tk))
            v = u_new + ((tk - 1) / t_new) * (u_new - u)
            u, tk = u_new, t_new
        self._u_prev = u
        self.iterations = iters
        self.solve_time = time.perf_counter() - t0
        return u


class LinearMPCController(ControllerBase):
    """Two-channel linear MPC (East->gimbal_y, North->gimbal_z)."""

    def __init__(self, control_period: float = 0.02, n_horizon: int = 40,
                 q_pos=0.0, q_vel=0.005, q_tilt=300.0, q_rate=60.0, q_gimbal=0.02,
                 r_move=10.0, r_delta=5.0, tau_gimbal: float = 0.05) -> None:
        super().__init__()
        self.dt = float(control_period)
        self.Q = np.diag([q_pos, q_vel, q_tilt, q_rate, q_gimbal])
        self.tau_gimbal = float(tau_gimbal)
        self._mpc = {
            ax: _ChannelMPC(self.dt, n_horizon, self.Q, r_move, r_delta, np.radians(8.0))
            for ax in ("x", "y")
        }
        self._sched: tuple | None = None
        self._delta_est = np.zeros(2)
        self._last_cmd = np.zeros(2)
        self._last_t: float | None = None

    def _setup(self, T, m, l_arm, J_yy, lim):
        if self._sched is not None:
            T0, m0, l0 = self._sched
            if abs(T - T0) < 0.05 * abs(T0) and abs(m - m0) < 0.05 * abs(m0) and abs(l_arm - l0) < 0.1:
                return
        for ax in ("x", "y"):
            A, B = lateral_linear_model(T, m, l_arm, J_yy, self.tau_gimbal, ax)
            ch = self._mpc[ax]
            ch.u_max = lim
            ch.setup(A, B)
        self._sched = (T, m, l_arm)

    def compute(self, t, nav: NavState, gcmd, info: dict) -> ControlCommand:
        t0 = time.perf_counter()
        throttle, q_des, T_cmd = self._allocate(nav, gcmd, info)

        dt = (t - self._last_t) if self._last_t is not None else self.dt
        self._last_t = t
        self._delta_est += (self._last_cmd - self._delta_est) * min(dt / self.tau_gimbal, 1.0)

        if not gcmd.engine_on:
            cmd = ControlCommand(0.0, 0.0, 0.0,
                                 rcs_torque_B=self._rcs_cmd(nav, q_des, info))
            self.compute_time = time.perf_counter() - t0
            return cmd

        T, l_arm, J_yy = self._authority(info, T_cmd)
        lim = self._gimbal_limits(info)
        self._setup(T, info["mass"], l_arm, J_yy, lim)

        th_e, th_n = lateral_tilts(nav.q)
        w_e, w_n = lateral_rates(nav.omega_B)
        d_des = quat_rotate(q_des, np.array([1.0, 0.0, 0.0]))
        th_e_ff, th_n_ff = lateral_tilts_from_dir(d_des)
        r_ref = gcmd.r_ref if gcmd.r_ref is not None else np.zeros(3)
        v_ref = gcmd.v_ref if gcmd.v_ref is not None else np.zeros(3)

        x_x = np.array([nav.r_I[0], nav.v_I[0], th_e, w_e, self._delta_est[0]])
        xr_x = np.array([r_ref[0], v_ref[0], th_e_ff, 0.0, 0.0])
        x_y = np.array([nav.r_I[1], nav.v_I[1], th_n, w_n, self._delta_est[1]])
        xr_y = np.array([r_ref[1], v_ref[1], th_n_ff, 0.0, 0.0])

        u_x = self._mpc["x"].solve(x_x, xr_x, self._last_cmd[0])
        u_y = self._mpc["y"].solve(x_y, xr_y, self._last_cmd[1])
        dy_cmd = float(np.clip(u_x[0], -lim, lim))
        dz_cmd = float(np.clip(u_y[0], -lim, lim))
        self._last_cmd = np.array([dy_cmd, dz_cmd])

        cmd = ControlCommand(throttle=throttle, gimbal_y=dy_cmd,
                             gimbal_z=dz_cmd, rcs_torque_B=np.zeros(3))
        self.compute_time = time.perf_counter() - t0
        return cmd
