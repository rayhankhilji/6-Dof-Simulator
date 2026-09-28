"""LQR lateral-channel controller with gain scheduling.

Per horizontal inertial channel (East via gimbal_y, North via gimbal_z)
the 5-state model ``[p, v, th, w, d]`` from
``allocation.lateral_linear_model`` is regulated with a CARE solution,
recomputed when T or m changes by more than 5% (gain scheduling).
Control: ``d_cmd = -K (x - x_ref)`` with x_ref built from the guidance
reference position/velocity and the feed-forward tilt of ``q_des``. The
actual gimbal deflection is tracked with the internal first-order actuator
model. Small-roll/yaw assumption: inertial East/North errors map to the
body channels per ``allocation.lateral_tilts`` / ``lateral_rates``.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.linalg import solve_continuous_are

from ..control_types import ControlCommand
from ..math.quaternion import quat_error
from ..navigation.base import NavState
from .allocation import (
    lateral_linear_model,
    lateral_rates,
    lateral_tilts,
    lateral_tilts_from_dir,
)
from .base import ControllerBase

# Position tracking is intentionally ~off: the guidance acceleration
# feed-forward (via q_des) does trajectory keeping; a position-tracking
# attitude loop would command large second swings (tilt to remove the drift
# it itself created). Velocity weight is kept tiny for drift damping.
Q_LQR = np.diag([0.0, 0.005, 300.0, 60.0, 0.02])
R_LQR = np.array([[10.0]])


class LQRController(ControllerBase):
    def __init__(self, q: np.ndarray | None = None, r: float = 10.0,
                 tau_gimbal: float = 0.05) -> None:
        super().__init__()
        self.Q = Q_LQR.copy() if q is None else np.asarray(q, dtype=float)
        self.R = np.array([[float(r)]])
        self.tau_gimbal = float(tau_gimbal)
        self._K = {"x": None, "y": None}
        self._sched: tuple | None = None
        self._delta_est = np.zeros(2)  # (dy, dz) internal actuator model
        self._last_cmd = np.zeros(2)
        self._last_t: float | None = None

    def _gains(self, T, m, l_arm, J_yy):
        if self._sched is not None:
            T0, m0 = self._sched
            if abs(T - T0) < 0.05 * abs(T0) and abs(m - m0) < 0.05 * abs(m0):
                return
        for axis in ("x", "y"):
            A, B = lateral_linear_model(T, m, l_arm, J_yy, self.tau_gimbal, axis)
            P = solve_continuous_are(A, B, self.Q, self.R)
            self._K[axis] = np.linalg.solve(self.R, B.T @ P).ravel()
        self._sched = (T, m)

    def compute(self, t, nav: NavState, gcmd, info: dict) -> ControlCommand:
        t0 = time.perf_counter()
        throttle, q_des, T_cmd = self._allocate(nav, gcmd, info)

        dt = (t - self._last_t) if self._last_t is not None else 0.02
        self._last_t = t
        self._delta_est += (self._last_cmd - self._delta_est) * min(dt / self.tau_gimbal, 1.0)

        if not gcmd.engine_on:
            cmd = ControlCommand(0.0, 0.0, 0.0,
                                 rcs_torque_B=self._rcs_cmd(nav, q_des, info))
            self.compute_time = time.perf_counter() - t0
            return cmd

        T, l_arm, J_yy = self._authority(info, T_cmd)
        self._gains(T, info["mass"], l_arm, J_yy)

        # Current tilts (east, north) and tilt rates; desired tilts from q_des.
        th_e, th_n = lateral_tilts(nav.q)
        w_e, w_n = lateral_rates(nav.omega_B)
        # Desired thrust direction in I.
        from ..math.quaternion import quat_rotate
        d_des = quat_rotate(q_des, np.array([1.0, 0.0, 0.0]))
        th_e_ff, th_n_ff = lateral_tilts_from_dir(d_des)

        r_ref = gcmd.r_ref if gcmd.r_ref is not None else np.zeros(3)
        v_ref = gcmd.v_ref if gcmd.v_ref is not None else np.zeros(3)

        x_x = np.array([nav.r_I[0], nav.v_I[0], th_e, w_e, self._delta_est[0]])
        xr_x = np.array([r_ref[0], v_ref[0], th_e_ff, 0.0, 0.0])
        x_y = np.array([nav.r_I[1], nav.v_I[1], th_n, w_n, self._delta_est[1]])
        xr_y = np.array([r_ref[1], v_ref[1], th_n_ff, 0.0, 0.0])

        dy_cmd = float(-(self._K["x"] @ (x_x - xr_x)))
        dz_cmd = float(-(self._K["y"] @ (x_y - xr_y)))
        lim = self._gimbal_limits(info)
        dy_cmd = float(np.clip(dy_cmd, -lim, lim))
        dz_cmd = float(np.clip(dz_cmd, -lim, lim))
        self._last_cmd = np.array([dy_cmd, dz_cmd])

        cmd = ControlCommand(throttle=throttle, gimbal_y=dy_cmd,
                             gimbal_z=dz_cmd, rcs_torque_B=np.zeros(3))
        self.compute_time = time.perf_counter() - t0
        return cmd
