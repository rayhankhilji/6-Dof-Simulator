"""Cascaded PID attitude controller for thrust-vector control.

Outer loop: attitude error ``e = quat_error(q_des, q)`` (body frame) ->
rate command ``w_c = Kp e`` (saturated at 30 deg/s). Inner loop: PID on the
rate error producing a desired body moment ``M_c``; gimbal angles invert
the TVC geometry ``M_y ~ -T l dy``, ``M_z ~ -T l dz`` (small-angle), i.e.
``d_i = -M_ci / (T l)``. Gains are normalized by ``J`` and the moment arm
authority ``T l`` so behaviour is consistent as mass/thrust vary. With the
engine off, the same PD law commands ``rcs_torque_B``.
"""

from __future__ import annotations

import time

import numpy as np

from ..control_types import ControlCommand
from ..math.quaternion import quat_error
from ..navigation.base import NavState
from .base import ControllerBase


class PIDAttitudeController(ControllerBase):
    def __init__(self, kp_att: float = 1.3, kp_rate: float = 0.8,
                 ki_rate: float = 0.15, rate_max: float = np.radians(25.0)) -> None:
        super().__init__()
        self.kp_att = float(kp_att)
        self.kp_rate = float(kp_rate)
        self.ki_rate = float(ki_rate)
        self.rate_max = float(rate_max)
        self._int = np.zeros(3)
        self._last_t: float | None = None

    def compute(self, t, nav: NavState, gcmd, info: dict) -> ControlCommand:
        t0 = time.perf_counter()
        throttle, q_des, T_cmd = self._allocate(nav, gcmd, info)
        e = quat_error(q_des, nav.q)

        dt = (t - self._last_t) if self._last_t is not None else 0.0
        self._last_t = t

        if not gcmd.engine_on:
            cmd = ControlCommand(0.0, 0.0, 0.0,
                                 rcs_torque_B=self._rcs_cmd(nav, q_des, info))
            self.compute_time = time.perf_counter() - t0
            return cmd

        T, l_arm, J_yy = self._authority(info, T_cmd)
        w_c = np.clip(self.kp_att * e, -self.rate_max, self.rate_max)
        # Map inertial-frame-free: rates are body rates; e is body frame.
        e_w = w_c - nav.omega_B

        # Desired body moment (diag J approximation).
        J = np.diag(info.get("J", np.eye(3)))
        lim = self._gimbal_limits(info)

        # Provisional gimbal demand from P+current I; used to gate the
        # integrator (conditional integration: stop accumulating while the
        # gimbal is railed, and bound the integral so its moment
        # contribution stays inside TVC authority, preventing the
        # windup limit-cycle that otherwise oscillates the attitude).
        d0 = np.array([-self.kp_rate * J[1] * e_w[1] / max(T * l_arm, 1e-6),
                       -self.kp_rate * J[2] * e_w[2] / max(T * l_arm, 1e-6)])
        sat = np.abs(d0) >= 0.98 * lim
        for k in (1, 2):
            di = k - 1
            if not (sat[di] and np.sign(e_w[k]) == np.sign(d0[di])):
                self._int[k] += e_w[k] * dt
        # Moment from integral must stay well inside T*l*sin(lim) so the
        # rail can still be commanded in the opposite direction.
        i_max = 0.25 * (T * l_arm * np.sin(lim)) / max(self.ki_rate * J[1], 1e-9)
        self._int = np.clip(self._int, -i_max, i_max)

        M_c = J * (self.kp_rate * e_w + self.ki_rate * self._int)

        # Invert M = -T l d (per axis: M_y -> dy, M_z -> dz; M_x -> none).
        d = np.zeros(2)
        d[0] = -M_c[1] / max(T * l_arm, 1e-6)   # delta_y
        d[1] = -M_c[2] / max(T * l_arm, 1e-6)   # delta_z
        d = np.clip(d, -lim, lim)

        cmd = ControlCommand(throttle=throttle, gimbal_y=float(d[0]),
                             gimbal_z=float(d[1]), rcs_torque_B=np.zeros(3))
        self.compute_time = time.perf_counter() - t0
        return cmd
