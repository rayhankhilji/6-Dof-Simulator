"""Navigation filter interfaces and shared state container.

Error-state convention (15-dim, shared by EKF and UKF):

    dx = [dr_I(3), dv_I(3), dtheta(3), db_a(3), db_g(3)]

where ``dtheta`` is the small-angle attitude error in the body frame such
that the true attitude is recovered via ``q_true = q_nom ⊗ [1, dtheta/2]``
(to first order). Biases are additive corrections: ``f_true = f_m - b_a``,
``w_true = w_m - b_g``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class NavState:
    """Navigation solution: nominal state plus covariance.

    Attributes
    ----------
    r_I, v_I : (3,) arrays
        Estimated inertial position/velocity [m], [m/s].
    q : (4,) array
        Estimated body-to-inertial quaternion.
    accel_bias, gyro_bias : (3,) arrays
        Estimated IMU biases.
    P : (15,15) array
        Error-state covariance (ordering as in module docstring).
    t : float
        Validity time [s].
    """

    r_I: np.ndarray
    v_I: np.ndarray
    q: np.ndarray
    accel_bias: np.ndarray
    gyro_bias: np.ndarray
    P: np.ndarray
    t: float = 0.0

    def copy(self) -> "NavState":
        return NavState(
            self.r_I.copy(), self.v_I.copy(), self.q.copy(),
            self.accel_bias.copy(), self.gyro_bias.copy(), self.P.copy(), self.t,
        )


class Navigator(ABC):
    """Common interface for navigation filters."""

    @abstractmethod
    def predict(self, imu_accel: np.ndarray, imu_gyro: np.ndarray, dt: float) -> None:
        """Propagate the estimate by ``dt`` using one IMU sample."""

    @abstractmethod
    def update(self, meas) -> None:
        """Apply a ``Measurements`` update (skipped fields are ignored)."""

    @property
    @abstractmethod
    def estimate(self) -> NavState:
        """Current navigation state."""
