"""Multiplicative (error-state) EKF for strapdown INS + GPS/baro/radar.

15-dim error state ``dx = [dr, dv, dtheta, db_a, db_g]`` (see
``navigation.base``). Nominal strapdown mechanization:

    q <- q x exp(0.5 (w_m - b_g) dt)
    a_I = R(q)(f_m - b_a) + g(r)
    v <- v + a_I dt ;  r <- r + v_old dt + 0.5 a_I dt^2

Continuous error dynamics F with blocks
    dv/dtheta = -R(q) [f_m - b_a]x ,  dv/db_a = -R(q)
    dtheta/dtheta = -[w_m - b_g]x   ,  dtheta/db_g = -I
    dr/dv = I
discretized as ``Phi ~ I + F dt + 0.5 (F dt)^2``; process noise
``Q = G Qc G^T dt`` from IMU noise densities and bias random walks.

Updates use the Joseph covariance form and 99% chi-square innovation gating.
GPS latency is handled by a short buffer of nominal (t, r, v): the innovation
is formed against the buffered state nearest ``t_valid`` and the correction
applied to the current state (delayed-measurement / current-state-correction
approximation). After each update the error state is injected into the
nominal (``q <- q x [1, dtheta/2]`` normalized) and reset to zero.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..environment.gravity import gravity_inertial
from ..math.quaternion import (
    quat_conjugate,
    quat_integrate,
    quat_multiply,
    quat_normalize,
    quat_rotate,
    skew,
)
from .base import Navigator, NavState

# Chi-square 99% gates.
_CHI2_3 = 11.3449  # 3-dof
_CHI2_1 = 6.6349   # 1-dof


def numerical_jacobian(func, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian of ``func(x)`` (scalar or vector output)."""
    x = np.asarray(x, dtype=float)
    f0 = np.atleast_1d(func(x))
    J = np.zeros((f0.size, x.size))
    for i in range(x.size):
        dx = np.zeros_like(x)
        dx[i] = eps
        J[:, i] = (np.atleast_1d(func(x + dx)) - np.atleast_1d(func(x - dx))) / (2 * eps)
    return J


def inject_error(q: np.ndarray, dtheta: np.ndarray) -> np.ndarray:
    """Nominal attitude update ``q <- q x [1, dtheta/2]`` (normalized)."""
    dq = np.concatenate([[1.0], 0.5 * np.asarray(dtheta)])
    return quat_normalize(quat_multiply(q, dq))


def propagate_nominal(st: NavState, f_m, w_m, dt) -> NavState:
    """Strapdown mechanization shared by the EKF and UKF sigma points.

    ``q <- q x exp(0.5 (w_m - b_g) dt)``, ``a_I = R(q)(f_m - b_a) + g(r)``,
    trapezoidal-style position update with the new specific force.
    """
    w = np.asarray(w_m) - st.gyro_bias
    f = np.asarray(f_m) - st.accel_bias
    q_new = quat_integrate(st.q, w, dt)
    a_I = quat_rotate(q_new, f) + gravity_inertial(st.r_I)
    v_new = st.v_I + a_I * dt
    r_new = st.r_I + st.v_I * dt + 0.5 * a_I * dt * dt
    return NavState(r_new, v_new, q_new, st.accel_bias.copy(),
                    st.gyro_bias.copy(), st.P, st.t + dt)


def process_noise(R_b2i: np.ndarray, dt: float, na, ng, rwa, rwg) -> np.ndarray:
    """``G Qc G^T dt`` process-noise increment for the 15-dim error state."""
    Q = np.zeros((15, 15))
    Q[3:6, 3:6] = (R_b2i @ R_b2i.T) * na**2 * dt   # accel noise -> dv
    Q[6:9, 6:9] = np.eye(3) * ng**2 * dt           # gyro noise -> dtheta
    Q[9:12, 9:12] = np.eye(3) * rwa**2 * dt
    Q[12:15, 12:15] = np.eye(3) * rwg**2 * dt
    return Q


class EKF(Navigator):
    """Multiplicative error-state EKF.

    Parameters
    ----------
    r0, v0, q0 : initial nominal state.
    P0_diag : (15,) initial error-state std devs (squared internally).
    accel_noise_density, gyro_noise_density : continuous white-noise
        densities [m/s^2/sqrt(Hz)], [rad/s/sqrt(Hz)].
    accel_bias_rw, gyro_bias_rw : bias random-walk densities [per sqrt(s)].
    gps_pos_sigma, gps_vel_sigma : measurement noise std devs.
    """

    def __init__(
        self,
        r0, v0, q0,
        P0_diag=None,
        accel_noise_density: float = 1e-3,
        gyro_noise_density: float = 1e-5,
        accel_bias_rw: float = 0.0,
        gyro_bias_rw: float = 0.0,
        gps_pos_sigma=(1.5, 1.5, 3.0),
        gps_vel_sigma: float = 0.1,
        t0: float = 0.0,
    ) -> None:
        self.nav = NavState(
            r_I=np.asarray(r0, dtype=float), v_I=np.asarray(v0, dtype=float),
            q=quat_normalize(np.asarray(q0, dtype=float)),
            accel_bias=np.zeros(3), gyro_bias=np.zeros(3),
            P=np.diag(np.asarray(P0_diag if P0_diag is not None else
                                [3.0]*3 + [0.5]*3 + list(np.radians([5.0]*3)) +
                                [1e-2]*3 + [1e-4]*3, dtype=float) ** 2),
            t=float(t0),
        )
        self.na = float(accel_noise_density)
        self.ng = float(gyro_noise_density)
        self.rwa = float(accel_bias_rw)
        self.rwg = float(gyro_bias_rw)
        self.gps_pos_sigma = np.asarray(gps_pos_sigma, dtype=float)
        self.gps_vel_sigma = float(gps_vel_sigma)
        self.innovation_log: list = []
        self._history: deque = deque(maxlen=300)  # (t, r, v)

    @property
    def estimate(self) -> NavState:
        return self.nav

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _F(self, f_m, w_m) -> np.ndarray:
        st = self.nav
        R = _dcm(st.q)
        w = np.asarray(w_m) - st.gyro_bias
        f = np.asarray(f_m) - st.accel_bias
        F = np.zeros((15, 15))
        F[0:3, 3:6] = np.eye(3)            # dr/dv
        F[3:6, 6:9] = -R @ skew(f)         # dv/dtheta
        F[3:6, 9:12] = -R                  # dv/db_a
        F[6:9, 6:9] = -skew(w)             # dtheta/dtheta
        F[6:9, 12:15] = -np.eye(3)         # dtheta/db_g
        return F

    def predict(self, imu_accel, imu_gyro, dt) -> None:
        st = self.nav
        F = self._F(imu_accel, imu_gyro)
        Phi = np.eye(15) + F * dt + 0.5 * (F * dt) @ (F * dt)
        Q = process_noise(_dcm(st.q), dt, self.na, self.ng, self.rwa, self.rwg)
        self.nav = propagate_nominal(st, imu_accel, imu_gyro, dt)
        self.nav.P = Phi @ st.P @ Phi.T + Q
        self.nav.P = 0.5 * (self.nav.P + self.nav.P.T)
        self._history.append((self.nav.t, self.nav.r_I.copy(), self.nav.v_I.copy()))

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------
    def _apply(self, H, R_meas, innov, kind, gate) -> bool:
        """Joseph-form update + chi-square gate. Returns True if applied."""
        P = self.nav.P
        S = H @ P @ H.T + R_meas
        nis = float(innov.T @ np.linalg.solve(S, innov))
        self.innovation_log.append((self.nav.t, kind, np.asarray(innov).copy(), S.copy()))
        if nis > gate:
            return False
        K = P @ H.T @ np.linalg.inv(S)
        dx = K @ innov
        self.nav.P = (np.eye(15) - K @ H) @ P @ (np.eye(15) - K @ H).T + K @ R_meas @ K.T
        self.nav.P = 0.5 * (self.nav.P + self.nav.P.T)
        self._inject(dx)
        return True

    def _inject(self, dx) -> None:
        st = self.nav
        st.r_I = st.r_I + dx[0:3]
        st.v_I = st.v_I + dx[3:6]
        st.q = inject_error(st.q, dx[6:9])
        st.accel_bias = st.accel_bias + dx[9:12]
        st.gyro_bias = st.gyro_bias + dx[12:15]

    def _buffered_state(self, t_valid):
        if not self._history:
            return self.nav.r_I, self.nav.v_I
        ts = np.array([e[0] for e in self._history])
        i = int(np.argmin(np.abs(ts - t_valid)))
        return self._history[i][1], self._history[i][2]

    def update(self, meas) -> None:
        if meas.gps_pos is not None:
            r_valid, v_valid = self._buffered_state(meas.gps_t_valid or meas.t)
            innov = np.asarray(meas.gps_pos) - r_valid
            H = np.zeros((3, 15)); H[0:3, 0:3] = np.eye(3)
            Rm = np.diag(self.gps_pos_sigma**2)
            self._apply(H, Rm, innov, "gps_pos", _CHI2_3)

            innov_v = np.asarray(meas.gps_vel) - v_valid
            Hv = np.zeros((3, 15)); Hv[0:3, 3:6] = np.eye(3)
            Rv = np.eye(3) * self.gps_vel_sigma**2
            self._apply(Hv, Rv, innov_v, "gps_vel", _CHI2_3)

        if meas.baro_alt is not None:
            innov = np.array([meas.baro_alt - self.nav.r_I[2]])
            H = np.zeros((1, 15)); H[0, 2] = 1.0
            Rm = np.array([[meas.baro_sigma**2]])
            self._apply(H, Rm, innov, "baro", _CHI2_1)

        if meas.radar_range is not None:
            st = self.nav

            def h_of(dx_att):
                q_p = inject_error(st.q, dx_att)
                axis_z = quat_rotate(q_p, np.array([1.0, 0.0, 0.0]))[2]
                return st.r_I[2] / max(axis_z, 1e-6)

            h_pred = h_of(np.zeros(3))
            H_att = numerical_jacobian(h_of, np.zeros(3))
            H = np.zeros((1, 15))
            H[0, 2] = 1.0 / max(quat_rotate(st.q, np.array([1.0, 0.0, 0.0]))[2], 1e-6)
            H[0, 6:9] = H_att
            innov = np.array([meas.radar_range - h_pred])
            sigma = 0.1 + 0.005 * abs(h_pred)
            self._apply(H, np.array([[sigma**2]]), innov, "radar", _CHI2_1)


def _dcm(q) -> np.ndarray:
    from ..math.quaternion import quat_to_dcm
    return quat_to_dcm(q)
