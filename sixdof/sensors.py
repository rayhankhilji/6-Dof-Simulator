"""Sensor models for the 6-DOF simulator.

All sensors take an ``rng`` (np.random.Generator) and have a ``rate_hz``;
each produces a sample only when ``t`` crosses its sample period (tracked via
``last_sample_t``). ``SensorSuite.sample`` returns a ``Measurements`` with
fields set to None for sensors that did not produce a sample that tick.

IMU error model (tactical grade defaults):
    f_meas = M_align ((I + S_a) f_true) + b_a + n_a
    w_meas = M_align ((I + S_g) w_true) + b_g + n_g
with constant biases drawn at init, first-order bias random walks, white
noise scaled by sqrt(rate), scale-factor error, and a small-angle
misalignment drawn at init.

GPS: ENU position/velocity with white noise, a pure transport delay
implemented as a FIFO of truth snapshots (measurement carries ``t_valid``),
scheduled outage windows, random dropouts, and extra near-ground multipath
noise below 50 m (up to 3x at the surface).

Barometer: pressure from the atmosphere model plus white noise and a slow
Gauss-Markov bias; altitude obtained by inverting the US76 layer model with
sigma propagated as sigma_h = sigma_p / (rho g).

Radar altimeter: slant range z / cos(tilt) along the body -x axis to z=0,
valid for altitude in [0, max_range] and tilt < 30 deg.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .environment.atmosphere import G0, R_AIR, USStandardAtmosphere1976, _H_B, _L_B
from .environment.gravity import gravity_inertial
from .math.quaternion import quat_rotate, quat_rotate_inverse, skew
from .state import IM, IQ, IR, IV, IW, State

_SAMPLE_TOL = 1e-9


def _due(last_t: Optional[float], t: float, period: float) -> bool:
    """True when a sample period has elapsed since ``last_t``."""
    if last_t is None:
        return True
    return t - last_t >= period - _SAMPLE_TOL


class Sensor:
    """Base sensor: sampling bookkeeping."""

    def __init__(self, rate_hz: float, rng: np.random.Generator) -> None:
        self.rate_hz = float(rate_hz)
        self.rng = rng
        self.last_sample_t: Optional[float] = None

    @property
    def period(self) -> float:
        return 1.0 / self.rate_hz

    def due(self, t: float) -> bool:
        return _due(self.last_sample_t, t, self.period)

    def mark(self, t: float) -> None:
        self.last_sample_t = t


class IMU(Sensor):
    """Tactical-grade strapdown IMU.

    Measures specific force in the body frame ``f_B = R(q)^T (a_I - g_I)``
    and body rate ``omega_B``, with bias / random-walk / white noise /
    scale-factor / misalignment errors.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        rate_hz: float = 200.0,
        accel_bias_sigma: float = 9.80665e-3,       # 1 mg
        gyro_bias_sigma: float = np.radians(1.0 / 3600.0),  # 1 deg/hr
        accel_bias_rw: float = 0.0,                  # m/s^2 per sqrt(s)
        gyro_bias_rw: float = 0.0,                   # rad/s per sqrt(s)
        accel_noise_density: float = 100e-6 * 9.80665,  # 100 ug/sqrt(Hz)
        gyro_noise_density: float = np.radians(0.005),    # 0.005 deg/sqrt(s)
        scale_factor_sigma: float = 300e-6,          # 300 ppm
        misalign_sigma: float = 5e-4,                # 0.5 mrad
    ) -> None:
        super().__init__(rate_hz, rng)
        self.accel_bias = rng.standard_normal(3) * accel_bias_sigma
        self.gyro_bias = rng.standard_normal(3) * gyro_bias_sigma
        self.accel_bias_rw = accel_bias_rw
        self.gyro_bias_rw = gyro_bias_rw
        self.accel_noise = accel_noise_density * np.sqrt(rate_hz)
        self.gyro_noise = gyro_noise_density * np.sqrt(rate_hz)
        self.accel_sf = 1.0 + rng.standard_normal(3) * scale_factor_sigma
        self.gyro_sf = 1.0 + rng.standard_normal(3) * scale_factor_sigma
        self.accel_mis = np.eye(3) + skew(rng.standard_normal(3) * misalign_sigma)
        self.gyro_mis = np.eye(3) + skew(rng.standard_normal(3) * misalign_sigma)

    def sample(self, t: float, x_true: np.ndarray, xdot_true: np.ndarray, env) -> tuple | None:
        """Return ``(accel_B, gyro_B)`` or None if not due."""
        if not self.due(t):
            return None
        self.mark(t)
        dt = self.period
        # Bias random walk.
        if self.accel_bias_rw > 0:
            self.accel_bias += self.accel_bias_rw * np.sqrt(dt) * self.rng.standard_normal(3)
        if self.gyro_bias_rw > 0:
            self.gyro_bias += self.gyro_bias_rw * np.sqrt(dt) * self.rng.standard_normal(3)

        s = State.from_array(x_true)
        a_I = xdot_true[IV]
        g_I = gravity_inertial(s.r_I)
        f_true = quat_rotate_inverse(s.q, a_I - g_I)
        w_true = s.omega_B

        accel = self.accel_mis @ (self.accel_sf * f_true) + self.accel_bias \
            + self.accel_noise * self.rng.standard_normal(3)
        gyro = self.gyro_mis @ (self.gyro_sf * w_true) + self.gyro_bias \
            + self.gyro_noise * self.rng.standard_normal(3)
        return accel, gyro


class GPS(Sensor):
    """ENU position/velocity receiver with latency, outages and multipath.

    A measurement produced at time ``t`` corresponds to truth at
    ``t - delay_s`` (FIFO of truth snapshots) and carries ``t_valid``.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        rate_hz: float = 10.0,
        pos_sigma: tuple = (1.5, 1.5, 3.0),
        vel_sigma: float = 0.1,
        delay_s: float = 0.1,
        outage_windows: list | None = None,
        p_dropout: float = 0.0,
        multipath_max_factor: float = 3.0,
        multipath_alt: float = 50.0,
    ) -> None:
        super().__init__(rate_hz, rng)
        self.pos_sigma = np.asarray(pos_sigma, dtype=float)
        self.vel_sigma = float(vel_sigma)
        self.delay_s = float(delay_s)
        self.outage_windows = outage_windows or []
        self.p_dropout = float(p_dropout)
        self.multipath_max_factor = float(multipath_max_factor)
        self.multipath_alt = float(multipath_alt)
        self._fifo: deque = deque()

    def in_outage(self, t: float) -> bool:
        return any(a <= t <= b for a, b in self.outage_windows)

    def sample(self, t: float, x_true: np.ndarray, xdot_true: np.ndarray, env):
        """Return ``(pos, vel, t_valid)`` or None."""
        s = State.from_array(x_true)
        self._fifo.append((t, s.r_I.copy(), s.v_I.copy()))
        # Drop entries older than the delay horizon (keep one for interpolation).
        while len(self._fifo) > 2 and self._fifo[1][0] <= t - self.delay_s:
            self._fifo.popleft()
        if not self.due(t):
            return None
        self.mark(t)
        if self.in_outage(t) or self.rng.random() < self.p_dropout:
            return None
        # Find the truth snapshot closest to t - delay.
        t_valid_target = t - self.delay_s
        ts = np.array([e[0] for e in self._fifo])
        idx = int(np.argmin(np.abs(ts - t_valid_target)))
        t_valid, r_v, v_v = self._fifo[idx]
        # Near-ground multipath scales noise up linearly to max at z=0.
        factor = 1.0 + (self.multipath_max_factor - 1.0) * max(
            0.0, 1.0 - r_v[2] / self.multipath_alt
        )
        pos = r_v + self.pos_sigma * factor * self.rng.standard_normal(3)
        vel = v_v + self.vel_sigma * factor * self.rng.standard_normal(3)
        return pos, vel, t_valid


class Barometer(Sensor):
    """Static-pressure altimeter with white noise and a slow GM bias."""

    def __init__(
        self,
        rng: np.random.Generator,
        atmosphere: USStandardAtmosphere1976,
        rate_hz: float = 50.0,
        noise_sigma: float = 10.0,      # Pa
        bias_sigma: float = 20.0,       # Pa, GM steady-state std
        bias_tau: float = 300.0,        # s
    ) -> None:
        super().__init__(rate_hz, rng)
        self.atm = atmosphere
        self.noise_sigma = float(noise_sigma)
        self.bias_sigma = float(bias_sigma)
        self.bias_tau = float(bias_tau)
        self.bias = 0.0
        self._last_t: Optional[float] = None

    def _step_bias(self, t: float) -> None:
        if self._last_t is None:
            self._last_t = t
            return
        dt = t - self._last_t
        self._last_t = t
        if dt <= 0.0:
            return
        phi = np.exp(-dt / self.bias_tau)
        self.bias = phi * self.bias + self.bias_sigma * np.sqrt(1 - phi**2) * self.rng.standard_normal()

    def pressure_to_altitude(self, p: float) -> float:
        """Invert the US76 model: geopotential altitude for pressure ``p`` [m]."""
        # Base pressures are monotone decreasing with altitude; find the layer
        # whose base pressure >= p > next layer's base (or the 86 km value).
        p_b = self.atm._p_b / self.atm.density_scale
        T_b = self.atm._T_b
        p_86 = self.atm._p_86 / self.atm.density_scale
        bounds = np.concatenate([p_b, [p_86]])
        hs = np.concatenate([_H_B, [86000.0]])
        # Pressure decreases with altitude; find the layer straddling p.
        for i in range(7):
            if p <= bounds[i] * (1.0 + 1e-12) and p > bounds[i + 1]:
                if _L_B[i] == 0.0:
                    return float(_H_B[i] - R_AIR * T_b[i] / G0 * np.log(p / bounds[i]))
                T = T_b[i] * (p / bounds[i]) ** (-R_AIR * _L_B[i] / G0)
                return float(_H_B[i] + (T - T_b[i]) / _L_B[i])
        if p > bounds[0]:
            return 0.0
        # Above 86 km: exponential tail inversion.
        rho = p / (R_AIR * self.atm._T_86)
        return float(86000.0 - 7000.0 * np.log(rho / (self.atm._rho_86 / self.atm.density_scale)))

    def sample(self, t: float, x_true: np.ndarray, xdot_true: np.ndarray, env):
        """Return ``(altitude, sigma_h)`` or None."""
        if not self.due(t):
            return None
        self.mark(t)
        self._step_bias(t)
        z = float(np.asarray(x_true)[IR][2])
        _, p_true, rho_true, _ = self.atm.properties(z)
        p_meas = float(p_true) + self.bias + self.noise_sigma * self.rng.standard_normal()
        h = self.pressure_to_altitude(p_meas)
        # sigma_h = sigma_p * |dh/dp| = sigma_p / (rho g); use the white-noise
        # sigma only (the GM bias is a correlated error, not white noise).
        sigma_p = self.noise_sigma
        g = 9.80665
        sigma_h = sigma_p / max(float(rho_true) * g, 1e-9)
        return h, sigma_h


class RadarAltimeter(Sensor):
    """Slant-range altimeter along the body -x axis to the z=0 plane.

    Valid only when altitude in [0, ``max_range``] and tilt < 30 deg.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        rate_hz: float = 25.0,
        max_range: float = 2000.0,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__(rate_hz, rng)
        self.max_range = float(max_range)
        self.dropout_prob = float(dropout_prob)

    def sample(self, t: float, x_true: np.ndarray, xdot_true: np.ndarray, env):
        """Return slant range [m] or None."""
        if not self.due(t):
            return None
        self.mark(t)
        s = State.from_array(x_true)
        z = s.r_I[2]
        if z < 0.0 or z > self.max_range:
            return None
        if self.rng.random() < self.dropout_prob:
            return None
        x_axis_I = quat_rotate(s.q, np.array([1.0, 0.0, 0.0]))
        cos_tilt = float(x_axis_I[2])
        if cos_tilt < np.cos(np.radians(30.0)):
            return None
        slant = z / cos_tilt
        sigma = 0.1 + 0.005 * slant
        return float(slant + sigma * self.rng.standard_normal())


@dataclass
class Measurements:
    """One tick of sensor outputs; None where no sample was produced."""

    t: float
    imu_accel: Optional[np.ndarray] = None
    imu_gyro: Optional[np.ndarray] = None
    gps_pos: Optional[np.ndarray] = None
    gps_vel: Optional[np.ndarray] = None
    gps_t_valid: Optional[float] = None
    baro_alt: Optional[float] = None
    baro_sigma: Optional[float] = None
    radar_range: Optional[float] = None


@dataclass
class SensorSuite:
    """Bundle of sensors producing a combined ``Measurements`` per tick."""

    imu: Optional[IMU] = None
    gps: Optional[GPS] = None
    barometer: Optional[Barometer] = None
    radar: Optional[RadarAltimeter] = None

    def sample(self, t: float, x_true: np.ndarray, xdot_true: np.ndarray, env) -> Measurements:
        m = Measurements(t=t)
        if self.imu is not None:
            out = self.imu.sample(t, x_true, xdot_true, env)
            if out is not None:
                m.imu_accel, m.imu_gyro = out
        if self.gps is not None:
            out = self.gps.sample(t, x_true, xdot_true, env)
            if out is not None:
                m.gps_pos, m.gps_vel, m.gps_t_valid = out
        if self.barometer is not None:
            out = self.barometer.sample(t, x_true, xdot_true, env)
            if out is not None:
                m.baro_alt, m.baro_sigma = out
        if self.radar is not None:
            out = self.radar.sample(t, x_true, xdot_true, env)
            if out is not None:
                m.radar_range = out
        return m
