"""Geometric (SO(3) quaternion) controller with feedback linearization.

Desired body moment:

    M_des = +K_R e_R - K_w w + w x (J w)

with ``e_R`` the quaternion error vector (body frame, pointing along the
rotation needed to reach ``q_des``, so the proportional term is positive). Gains are scaled by
the inertia: ``K_R = 4 J``, ``K_w = 2.5 J`` per axis. The gimbal command
inverts the exact TVC moment geometry ``M_y = -T l sin(dy)``,
``M_z = -T l sin(dz)`` -> ``d_i = -asin(clip(M_i/(T l)))``; when the moment
request would exceed the gimbal limit the tilt request is scaled down
(anti-saturation).
"""

from __future__ import annotations

import time

import numpy as np

from ..control_types import ControlCommand
from ..math.quaternion import quat_error
from ..navigation.base import NavState
from .base import ControllerBase


class GeometricController(ControllerBase):
    def __init__(self, k_r: float = 4.0, k_w: float = 2.5) -> None:
        super().__init__()
        self.k_r = float(k_r)
        self.k_w = float(k_w)

    def compute(self, t, nav: NavState, gcmd, info: dict) -> ControlCommand:
        t0 = time.perf_counter()
        throttle, q_des, T_cmd = self._allocate(nav, gcmd, info)
        e = quat_error(q_des, nav.q)          # body-frame attitude error
        J = np.diag(info.get("J", np.eye(3)))
        w = nav.omega_B

        if not gcmd.engine_on:
            cmd = ControlCommand(0.0, 0.0, 0.0,
                                 rcs_torque_B=self._rcs_cmd(nav, q_des, info))
            self.compute_time = time.perf_counter() - t0
            return cmd

        T, l_arm, J_yy = self._authority(info, T_cmd)

        # Feedback-linearized desired moment (diag-J approximation).
        # e is the rotation vector taking current -> desired attitude, so the
        # proportional term must drive omega *along* +e: M = +k_r J e - ...
        wxJw = np.cross(w, J * w)
        M_des = self.k_r * J * e - self.k_w * J * w + wxJw

        # Exact inversion M_i = -T l sin(d_i); anti-saturation scaling.
        lim = self._gimbal_limits(info)
        M_lim = T * l_arm * np.sin(lim)
        scale = max(1.0, abs(M_des[1]) / M_lim, abs(M_des[2]) / M_lim)
        M_des = M_des / scale
        d_y = -np.arcsin(np.clip(M_des[1] / max(T * l_arm, 1e-6), -1.0, 1.0))
        d_z = -np.arcsin(np.clip(M_des[2] / max(T * l_arm, 1e-6), -1.0, 1.0))

        cmd = ControlCommand(throttle=throttle, gimbal_y=float(d_y),
                             gimbal_z=float(d_z), rcs_torque_B=np.zeros(3))
        self.compute_time = time.perf_counter() - t0
        return cmd
