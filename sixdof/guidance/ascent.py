"""Ascent guidance: gravity turn with pitch kick, then tangent steering.

Phases:

- ``vertical``: until ``h_kick`` (500 m), thrust along +z_I.
- ``pitch_kick``: linear tilt to ``kick_angle`` (4 deg) toward East over 5 s.
- ``gravity_turn``: zero angle of attack -- thrust along ``unit(v_I)``
  until stage-1 burnout (prop < 0.5% triggers ``vehicle.stage()`` via the
  simulation staging hook; a 2 s coast precedes stage-2 ignition).
- ``tangent_steering`` (stage 2): linear tangent steering
  ``tan(theta) = tan(theta_0) - c t`` with the steering rate ``c`` chosen so
  the flight-path angle reaches ~0 at cutoff; the run stops when the
  apogee estimate ``z + v_z^2/(2g)`` exceeds ``h_target`` (200 km) -- this is
  a targeting heuristic, not a true orbit insertion.
"""

from __future__ import annotations

import numpy as np

from ..control_types import GuidanceCommand
from ..math.quaternion import quat_from_two_vectors, quat_multiply, quat_from_axis_angle
from ..navigation.base import NavState

Z_HAT = np.array([0.0, 0.0, 1.0])
E_HAT = np.array([1.0, 0.0, 0.0])


class GravityTurnGuidance:
    """Gravity-turn ascent guidance with staging logic."""

    def __init__(
        self,
        h_kick: float = 500.0,
        kick_angle_deg: float = 4.0,
        kick_duration: float = 5.0,
        h_target: float = 200e3,
        v_target: float = 7790.0,
        coast_after_sep: float = 2.0,
    ) -> None:
        self.h_kick = float(h_kick)
        self.kick_angle = np.radians(kick_angle_deg)
        self.kick_duration = float(kick_duration)
        self.h_target = float(h_target)
        self.v_target = float(v_target)
        self.coast_after_sep = float(coast_after_sep)

        self.phase = "vertical"
        self._kick_t0: float | None = None
        self._theta0: float | None = None   # tangent-steering start angle
        self._steer_t0: float | None = None
        self._sep_t: float | None = None
        self._staged = False

    def _thrust_dir(self, nav: NavState, t) -> tuple[np.ndarray, float, str]:
        """Desired thrust direction in I, throttle, phase."""
        h = nav.r_I[2]
        v = nav.v_I
        speed = np.linalg.norm(v)

        if self.phase == "vertical":
            if h >= self.h_kick:
                self.phase = "pitch_kick"
                self._kick_t0 = t
            return Z_HAT, 1.0, "vertical"

        if self.phase == "pitch_kick":
            frac = min((t - self._kick_t0) / self.kick_duration, 1.0)
            ang = self.kick_angle * frac
            # Tilt from +z toward +x (East).
            d = Z_HAT * np.cos(ang) + E_HAT * np.sin(ang)
            if frac >= 1.0:
                self.phase = "gravity_turn"
            return d, 1.0, "pitch_kick"

        if self.phase == "gravity_turn":
            if speed > 1.0:
                d = v / speed
            else:
                d = Z_HAT
            return d, 1.0, "gravity_turn"

        if self.phase == "stage_sep_coast":
            return Z_HAT, 0.0, "stage_sep_coast"

        # tangent_steering
        gamma = np.arctan2(v[2], np.hypot(v[0], v[1]))  # flight-path angle
        if self._theta0 is None:
            self._theta0 = gamma
            self._steer_t0 = t
        # Linear tangent steering toward horizontal at v_target.
        # theta_dot = -tan(theta0)/T_s; pick T_s from remaining dv estimate.
        dv_left = max(self.v_target - np.hypot(v[0], v[1]), 100.0)
        a_t = 9.0  # rough stage-2 accel [m/s^2]
        T_s = dv_left / a_t
        c = np.tan(self._theta0) / T_s
        theta = np.arctan(np.tan(self._theta0) - c * (t - self._steer_t0))
        theta = max(theta, 0.0)
        d = np.array([np.cos(theta), 0.0, np.sin(theta)])
        return d, 1.0, "tangent_steering"

    def compute(self, t, nav: NavState, info: dict) -> GuidanceCommand:
        # Staging: when stage-1 prop < 0.5% request sep; sim calls
        # vehicle.stage() through the staging hook. Then 2 s coast.
        prop = info.get("prop_remaining", 1.0)
        stage = info.get("stage_index", 0)
        if stage == 0 and prop <= 0.005 * info.get("stage_prop_capacity", 1.0) and self.phase == "gravity_turn":
            self.phase = "stage_sep_coast"
            self._sep_t = t
        if self.phase == "stage_sep_coast" and self._sep_t is not None:
            if t - self._sep_t >= self.coast_after_sep:
                self.phase = "tangent_steering"

        d, throttle, phase = self._thrust_dir(nav, t)
        if self.phase == "stage_sep_coast":
            throttle = 0.0
        q_des = quat_from_two_vectors(np.array([1.0, 0.0, 0.0]), d)
        accel = d * (info["thrust_max"] / info["mass"]) * throttle
        return GuidanceCommand(
            accel_cmd_I=accel, throttle=throttle, engine_on=throttle > 0,
            q_des=q_des, phase=phase, r_ref=nav.r_I.copy(), v_ref=nav.v_I.copy(),
        )
