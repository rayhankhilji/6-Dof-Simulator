"""Actuator models: gimbal servo and throttle lag + delay.

Gimbal: second-order servo per axis,

    d2 delta = wn^2 (delta_cmd - delta) - 2 zeta wn d delta

integrated with semi-implicit Euler, with angle and rate limits.

Throttle: first-order lag with time constant ``throttle_tau`` plus a pure
transport delay ``delay_s`` implemented as a FIFO. Output is the effective
throttle passed to dynamics: nonzero commands are clamped up to the
throttleable floor; ``fail_at_t`` permanently kills the engine after that
time and ``thrust_scale`` applies a constant dispersion (applied as an
effective-throttle scale).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .control_types import ControlCommand


class GimbalActuator:
    """Two-axis second-order gimbal servo (delta_y, delta_z)."""

    def __init__(self, gimbal_max: float, gimbal_rate_max: float,
                 wn: float = 2.0 * np.pi * 4.0, zeta: float = 0.7) -> None:
        self.gimbal_max = float(gimbal_max)
        self.rate_max = float(gimbal_rate_max)
        self.wn = float(wn)
        self.zeta = float(zeta)
        self.delta = np.zeros(2)   # actual deflection [rad]
        self.delta_dot = np.zeros(2)

    def step(self, cmd: np.ndarray, dt: float) -> np.ndarray:
        """Advance toward ``cmd`` (2-vector of rad) by ``dt``; return actual."""
        cmd = np.clip(np.asarray(cmd, dtype=float), -self.gimbal_max, self.gimbal_max)
        # Semi-implicit Euler on the servo dynamics.
        dd = self.wn**2 * (cmd - self.delta) - 2.0 * self.zeta * self.wn * self.delta_dot
        self.delta_dot = np.clip(self.delta_dot + dd * dt, -self.rate_max, self.rate_max)
        self.delta = np.clip(self.delta + self.delta_dot * dt, -self.gimbal_max, self.gimbal_max)
        return self.delta.copy()


class ThrottleActuator:
    """First-order throttle lag with transport delay and failure hooks."""

    def __init__(
        self,
        tau: float,
        min_frac: float,
        delay_s: float = 0.05,
        fail_at_t: Optional[float] = None,
        thrust_scale: float = 1.0,
    ) -> None:
        self.tau = float(tau)
        self.min_frac = float(min_frac)
        self.delay_s = float(delay_s)
        self.fail_at_t = fail_at_t
        self.thrust_scale = float(thrust_scale)
        self.value = 0.0          # current lagged (pre-scale) throttle
        self._fifo: deque = deque()  # (t_available, cmd)
        self._last_cmd = 0.0       # most recently released command

    def step(self, cmd: float, t: float, dt: float) -> float:
        """Advance the actuator; return effective throttle in [0,1]."""
        self._fifo.append((t + self.delay_s, float(cmd)))
        while self._fifo and self._fifo[0][0] <= t:
            self._last_cmd = self._fifo.popleft()[1]
        delayed_cmd = self._last_cmd

        # First-order lag (exact discrete update).
        a = 1.0 - np.exp(-dt / self.tau)
        self.value += a * (delayed_cmd - self.value)

        eff = self.value
        if eff > 0.0:
            eff = float(np.clip(eff, self.min_frac, 1.0))
        eff *= self.thrust_scale
        if self.fail_at_t is not None and t >= self.fail_at_t:
            eff = 0.0
        return eff


@dataclass
class ActuatorSuite:
    """Gimbal + throttle bundle stepping a ``ControlCommand``."""

    gimbal: GimbalActuator
    throttle: ThrottleActuator

    def step(self, cmd: ControlCommand, t: float, dt: float) -> dict:
        """Return ``u`` dict for dynamics: {throttle, gimbal=(dy, dz)}."""
        g = self.gimbal.step(np.array([cmd.gimbal_y, cmd.gimbal_z]), dt)
        th = self.throttle.step(cmd.throttle, t, dt)
        return {"throttle": th, "gimbal": (float(g[0]), float(g[1]))}
