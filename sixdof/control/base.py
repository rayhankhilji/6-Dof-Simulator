"""Shared controller scaffolding.

All controllers implement ``compute(t, nav, gcmd, info) -> ControlCommand``.
Common flow: allocate (throttle, q_des) from the commanded acceleration via
``allocation.thrust_and_attitude_from_accel``; attitude error
``e = quat_error(q_des, q)`` (body frame); feedback produces a desired
moment that maps to gimbal angles through the Phase-1 sign convention
(``M_y ~ -T l sin(dy)``, ``M_z ~ -T l sin(dz)``); with the engine off the
same PD law drives ``rcs_torque_B``. Each controller records
``compute_time``.
"""

from __future__ import annotations

import time

import numpy as np

from ..control_types import ControlCommand
from ..math.quaternion import quat_error
from ..navigation.base import NavState
from .allocation import thrust_and_attitude_from_accel


class ControllerBase:
    """Common allocation/RCS plumbing for TVC controllers."""

    def __init__(self) -> None:
        self.compute_time: float = 0.0

    # ------------------------------------------------------------------
    def _allocate(self, nav: NavState, gcmd, info: dict):
        """Return (throttle, q_des, T_cmd) from the guidance command."""
        if gcmd.engine_on:
            throttle, q_des = thrust_and_attitude_from_accel(
                gcmd.accel_cmd_I, info["mass"], info["thrust_min"],
                info["thrust_max"], nav.q,
            )
            T_cmd = throttle * info["thrust_max"]
        else:
            throttle = 0.0
            T_cmd = 0.0
            q_des = gcmd.q_des if gcmd.q_des is not None else nav.q.copy()
        return throttle, q_des, T_cmd

    def _rcs_cmd(self, nav: NavState, q_des, info: dict,
                 k_r: float = 3.0, k_w: float = 6.0) -> np.ndarray:
        """RCS-only attitude PD when the main engine is off."""
        e = quat_error(q_des, nav.q)
        J = np.diag(info.get("J", np.eye(3)))
        torque = k_r * J * e - k_w * J * nav.omega_B
        return torque

    def _gimbal_limits(self, info: dict) -> float:
        return float(info.get("gimbal_max", np.radians(8.0)))

    def _authority(self, info: dict, T_cmd: float) -> tuple[float, float, float]:
        """(T, l_arm, J_yy) current TVC authority parameters."""
        T = max(T_cmd, info.get("thrust_min", 0.0))
        l_arm = float(info.get("l_arm", max(info.get("cg_x", 8.0), 1.0)))
        J_yy = float(info.get("J_yy", np.diag(info.get("J", np.eye(3)))[1]))
        return T, l_arm, J_yy

    def compute(self, t, nav: NavState, gcmd, info: dict) -> ControlCommand:
        raise NotImplementedError
