"""Shared powered-descent guidance logic.

Common flow:

1. Coast until the 1-D hoverslam ignition estimate
   ``h <= h_ignite = v_z^2 / (2 * 0.85 * a_max_net) + 150`` m, with
   ``a_max_net = T_max/m - g``. During coast the engine is off and guidance
   requests a retrograde-ish attitude (body +x along -v_hat, blended toward
   +z so tilt stays within 25 deg).
2. Powered descent toward r_f = (0,0,0) with target velocity v_f.
3. Terminal: below ``h_terminal`` switch the velocity target to
   ``v_f = (0,0,-1.5)`` m/s vertical.
"""

from __future__ import annotations

import numpy as np

from ..control_types import GuidanceCommand
from ..math.quaternion import quat_from_two_vectors, quat_rotate
from ..navigation.base import NavState

Z_HAT = np.array([0.0, 0.0, 1.0])


class PoweredDescentBase:
    """Ignition/terminal logic shared by powered-descent guidance laws."""

    def __init__(self, h_terminal: float = 30.0, tilt_max_deg: float = 25.0) -> None:
        self.h_terminal = float(h_terminal)
        self.tilt_max = np.radians(tilt_max_deg)
        self.ignited = False
        self.phase = "coast"
        self._a_filt: np.ndarray | None = None
        self._a_t_last: float | None = None

    # ------------------------------------------------------------------
    def _a_max_net(self, info: dict) -> float:
        return info["thrust_max"] / info["mass"] - info["g"]

    def _h_ignite(self, v_z: float, info: dict) -> float:
        a = max(0.85 * self._a_max_net(info), 1e-3)
        return v_z * v_z / (2.0 * a) + 150.0

    def _retrograde_q_des(self, v_I: np.ndarray) -> np.ndarray:
        """Desired attitude: +x_B along -v, blended to <= tilt_max from +z."""
        v_h = np.linalg.norm(v_I)
        if v_h < 1.0:
            desired = Z_HAT.copy()
        else:
            desired = -v_I / v_h
        cos_tilt = np.clip(desired @ Z_HAT, -1.0, 1.0)
        tilt = np.arccos(cos_tilt)
        if tilt > self.tilt_max:
            horiz = desired - (desired @ Z_HAT) * Z_HAT
            n = np.linalg.norm(horiz)
            if n > 1e-9:
                horiz /= n
                desired = Z_HAT * np.cos(self.tilt_max) + horiz * np.sin(self.tilt_max)
            else:
                desired = Z_HAT.copy()
        return quat_from_two_vectors(np.array([1.0, 0.0, 0.0]), desired)

    def _targets(self, r_I: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
        """Reference (r_f, v_f, phase): terminal vertical descent below h_terminal.

        The terminal aim point sits 0.5 m below the surface so the replanned
        descent keeps flying through z=0 instead of asymptotically hovering
        at the ZEM/ZEV stand-off equilibrium created by the t_go floor.
        """
        if r_I[2] < self.h_terminal:
            return np.array([0.0, 0.0, -0.5]), np.array([0.0, 0.0, -1.5]), "terminal"
        return np.zeros(3), np.array([0.0, 0.0, -3.0]), "powered_descent"

    def _coast_cmd(self, nav: NavState, info: dict) -> GuidanceCommand:
        return GuidanceCommand(
            accel_cmd_I=np.zeros(3), throttle=0.0, engine_on=False,
            q_des=self._retrograde_q_des(nav.v_I), phase="coast",
            r_ref=nav.r_I.copy(), v_ref=nav.v_I.copy(), t_go=0.0,
        )

    def _throttle_hint(self, a_cmd: np.ndarray, info: dict) -> float:
        t = info["mass"] * np.linalg.norm(a_cmd)
        return float(np.clip(t / max(info["thrust_max"], 1e-9), 0.0, 1.0))

    def _maybe_ignite(self, nav: NavState, info: dict) -> bool:
        if self.ignited:
            return True
        h = nav.r_I[2]
        v_z = nav.v_I[2]
        if v_z < 0.0 and h <= self._h_ignite(v_z, info):
            self.ignited = True
        return self.ignited

    # ------------------------------------------------------------------
    def _lateral_servo(self, nav: NavState, rf: np.ndarray, t_go: float,
                       v_lat_max: float = 25.0,
                       kv_lat: float = 0.45) -> np.ndarray:
        """Bounded-approach-speed lateral divert; returns a_lat (x, y).

        The classic ``6*ZEM/T^2 - 2*ZEV/T`` lateral law builds up
        ~2*|offset|/T of lateral speed mid-divert and then has to slew the
        thrust azimuth ~180 deg to remove it -- more than the slow TVC
        attitude loop (max ~0.2 rad/s^2) can track, so the demand
        saturates, overshoots and oscillates. A bounded-velocity servo
        keeps the thrust azimuth steady toward the pad throughout, and the
        desired approach speed fades to zero below ~100 m so the final
        descent only damps residual lateral velocity (landing upright with
        a small offset beats tipping over or limit-cycling while trying to
        erase it).
        """
        tau_app = float(np.clip(0.35 * t_go, 4.0, 12.0))
        fade = float(np.clip(nav.r_I[2] / 100.0, 0.0, 1.0))
        a_lat = np.zeros(2)
        for k in (0, 1):
            v_des = np.clip((rf[k] - nav.r_I[k]) / tau_app,
                            -v_lat_max, v_lat_max) * fade
            a_lat[k] = kv_lat * (v_des - nav.v_I[k])
        return a_lat

    def _filter_lateral(self, a_cmd: np.ndarray, t: float,
                        tau: float = 1.5) -> np.ndarray:
        """First-order low-pass on the *lateral* accel command.

        TVC lateral authority comes from tilting the vehicle, and the
        attitude loop here is slow (max ~0.2 rad/s^2 from gimbal
        authority / J). Demanded lateral accelerations that change faster
        than ~0.3-0.5 Hz excite the attitude loop into limit-cycle
        oscillations, so the lateral channels are filtered to a bandwidth
        the attitude loop can actually track. The vertical channel is left
        unfiltered -- braking response time matters near the ground.
        """
        dt = 0.0
        if self._a_t_last is not None:
            dt = t - self._a_t_last
        self._a_t_last = t
        if self._a_filt is None or dt <= 0.0:
            self._a_filt = a_cmd.copy()
            return a_cmd
        alpha = float(np.clip(dt / max(tau, 1e-9), 0.0, 1.0))
        self._a_filt[:2] += (a_cmd[:2] - self._a_filt[:2]) * alpha
        out = a_cmd.copy()
        out[:2] = self._a_filt[:2]
        return out
