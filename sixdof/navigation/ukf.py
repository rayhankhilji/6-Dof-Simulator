"""USQUE-style quaternion UKF over the same 15-dim error-state structure.

Sigma points are drawn in the 15-dim error space of the current covariance;
each sigma point's attitude is ``q_i = q_nom x exp(dtheta_i)`` and points are
propagated through the same strapdown mechanization as the EKF
(``ekf.propagate_nominal``). The mean attitude is recovered via the
rotation-vector mean of the sigma points about the zeroth point (one
refinement pass). Measurement updates use the unscented transform of the
EKF's measurement functions; GPS latency uses the same buffered-history
approximation as the EKF.

Scaled-UT parameters: alpha=1e-3, beta=2, kappa=0.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..math.quaternion import (
    quat_conjugate,
    quat_multiply,
    quat_normalize,
    quat_rotate,
)
from .base import Navigator, NavState
from .ekf import _CHI2_1, _CHI2_3, _dcm, inject_error, numerical_jacobian, process_noise, propagate_nominal

_N = 15
_ALPHA = 1e-3
_BETA = 2.0
_KAPPA = 0.0


def _rotvec_of(q: np.ndarray) -> np.ndarray:
    """Rotation vector of a unit quaternion (2*vec for small angles)."""
    q = quat_normalize(q)
    if q[0] < 0:
        q = -q
    return 2.0 * q[1:4]


def _exp_map(dtheta: np.ndarray) -> np.ndarray:
    dq = np.concatenate([[1.0], 0.5 * np.asarray(dtheta)])
    return quat_normalize(dq)


class UKF(Navigator):
    """Quaternion UKF. Same constructor args and interface as :class:`EKF`."""

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
        self._history: deque = deque(maxlen=300)

        lam = _ALPHA**2 * (_N + _KAPPA) - _N
        self._scale = _N + lam
        self._wm = np.full(2 * _N + 1, 1.0 / (2 * self._scale))
        self._wc = self._wm.copy()
        self._wm[0] = lam / self._scale
        self._wc[0] = self._wm[0] + (1 - _ALPHA**2 + _BETA)

    @property
    def estimate(self) -> NavState:
        return self.nav

    # ------------------------------------------------------------------
    def _sigma_states(self, st: NavState) -> list[NavState]:
        """Sigma-point states: error-space offsets injected into ``st``."""
        P = 0.5 * (st.P + st.P.T)
        try:
            S = np.linalg.cholesky(self._scale * P)
        except np.linalg.LinAlgError:
            S = np.linalg.cholesky(self._scale * (P + 1e-12 * np.eye(_N)))
        pts = [st.copy()]
        for i in range(_N):
            for sgn in (1.0, -1.0):
                dx = sgn * S[:, i]
                pts.append(NavState(
                    st.r_I + dx[0:3], st.v_I + dx[3:6],
                    quat_normalize(quat_multiply(st.q, _exp_map(dx[6:9]))),
                    st.accel_bias + dx[9:12], st.gyro_bias + dx[12:15],
                    st.P, st.t,
                ))
        return pts

    def _recover_mean(self, pts: list[NavState]) -> tuple[NavState, np.ndarray]:
        """Mean state + error vectors of each sigma point about the mean."""
        wm = self._wm
        r = sum(wm[i] * pts[i].r_I for i in range(len(pts)))
        v = sum(wm[i] * pts[i].v_I for i in range(len(pts)))
        ba = sum(wm[i] * pts[i].accel_bias for i in range(len(pts)))
        bg = sum(wm[i] * pts[i].gyro_bias for i in range(len(pts)))
        t = pts[0].t

        # Attitude mean via rotation-vector averaging about sigma 0.
        q_mean = pts[0].q
        for _ in range(2):
            e = np.zeros(3)
            for i in range(len(pts)):
                e += wm[i] * _rotvec_of(quat_multiply(quat_conjugate(q_mean), pts[i].q))
            q_mean = quat_normalize(quat_multiply(q_mean, _exp_map(e)))

        dxs = np.zeros((len(pts), _N))
        for i, p in enumerate(pts):
            dxs[i, 0:3] = p.r_I - r
            dxs[i, 3:6] = p.v_I - v
            dxs[i, 6:9] = _rotvec_of(quat_multiply(quat_conjugate(q_mean), p.q))
            dxs[i, 9:12] = p.accel_bias - ba
            dxs[i, 12:15] = p.gyro_bias - bg
        return NavState(r, v, q_mean, ba, bg, self.nav.P, t), dxs

    # ------------------------------------------------------------------
    def predict(self, imu_accel, imu_gyro, dt) -> None:
        st = self.nav
        pts = [propagate_nominal(p, imu_accel, imu_gyro, dt) for p in self._sigma_states(st)]
        mean, dxs = self._recover_mean(pts)
        P = np.einsum("i,ij,ik->jk", self._wc, dxs, dxs)
        P += process_noise(_dcm(mean.q), dt, self.na, self.ng, self.rwa, self.rwg)
        mean.P = 0.5 * (P + P.T)
        self.nav = mean
        self._history.append((mean.t, mean.r_I.copy(), mean.v_I.copy()))

    # ------------------------------------------------------------------
    def _meas_update(self, h_func, z, Rm, kind, gate, h_at_mean=None) -> bool:
        """Unscented measurement update for scalar/vector measurement ``z``."""
        st = self.nav
        pts = self._sigma_states(st)
        _, dxs = self._recover_mean(pts)
        zs = np.array([np.atleast_1d(h_func(p)) for p in pts])
        z_mean = np.einsum("i,ij->j", self._wm, zs)
        dz = zs - z_mean
        S = np.einsum("i,ij,ik->jk", self._wc, dz, dz) + Rm
        Pxz = np.einsum("i,ij,ik->jk", self._wc, dxs, dz)
        # For delayed measurements the innovation is formed against the
        # buffered past state (h_at_mean); S/K still use the current spread.
        z_pred = h_at_mean if h_at_mean is not None else z_mean
        innov = np.atleast_1d(z - z_pred)
        nis = float(innov.T @ np.linalg.solve(S, innov))
        self.innovation_log.append((st.t, kind, innov.copy(), S.copy()))
        if nis > gate:
            return False
        K = Pxz @ np.linalg.inv(S)
        dx = K @ innov
        P_new = st.P - K @ S @ K.T
        self.nav.P = 0.5 * (P_new + P_new.T)
        # Inject correction into nominal.
        self.nav.r_I = st.r_I + dx[0:3]
        self.nav.v_I = st.v_I + dx[3:6]
        self.nav.q = inject_error(st.q, dx[6:9])
        self.nav.accel_bias = st.accel_bias + dx[9:12]
        self.nav.gyro_bias = st.gyro_bias + dx[12:15]
        return True

    def _buffered_state(self, t_valid):
        if not self._history:
            return self.nav.r_I, self.nav.v_I
        ts = np.array([e[0] for e in self._history])
        i = int(np.argmin(np.abs(ts - t_valid)))
        return self._history[i][1], self._history[i][2]

    def update(self, meas) -> None:
        if meas.gps_pos is not None:
            r_valid, v_valid = self._buffered_state(meas.gps_t_valid or meas.t)
            self._meas_update(
                lambda p: p.r_I, meas.gps_pos,
                np.diag(self.gps_pos_sigma**2), "gps_pos", _CHI2_3,
                h_at_mean=r_valid,
            )
            self._meas_update(
                lambda p: p.v_I, meas.gps_vel,
                np.eye(3) * self.gps_vel_sigma**2, "gps_vel", _CHI2_3,
                h_at_mean=v_valid,
            )

        if meas.baro_alt is not None:
            self._meas_update(
                lambda p: p.r_I[2], [meas.baro_alt],
                np.array([[meas.baro_sigma**2]]), "baro", _CHI2_1,
            )

        if meas.radar_range is not None:
            def h(p):
                axis_z = quat_rotate(p.q, np.array([1.0, 0.0, 0.0]))[2]
                return p.r_I[2] / max(axis_z, 1e-6)
            z_pred = h(self.nav)
            sigma = 0.1 + 0.005 * abs(z_pred)
            self._meas_update(h, [meas.radar_range], np.array([[sigma**2]]),
                              "radar", _CHI2_1)
