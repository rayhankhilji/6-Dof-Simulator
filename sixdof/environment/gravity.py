"""Gravity model for the flat-Earth ENU inertial frame.

Inverse-square gravity along the local vertical only:

    g(r) = -mu / (R_e + z)^2 * z_hat

Lateral curvature of the gravity field is neglected (flat-Earth
approximation), consistent with the ENU frame assumption; Earth rotation
is neglected as well.
"""

from __future__ import annotations

import numpy as np

MU_EARTH = 3.986004418e14  # m^3/s^2
R_EARTH = 6371000.0  # m
G0 = 9.80665  # m/s^2


def gravity_inertial(r_I: np.ndarray) -> np.ndarray:
    """Gravity acceleration in the ENU inertial frame at position ``r_I``.

    Parameters
    ----------
    r_I : (3,) or (N, 3) array
        Position(s) in the inertial frame [m].

    Returns
    -------
    (3,) or (N, 3) array
        Gravity vector [m/s^2], pointing along -z.
    """
    r = np.asarray(r_I, dtype=float)
    z = r[..., 2]
    g = -MU_EARTH / (R_EARTH + z) ** 2
    out = np.zeros(r.shape)
    out[..., 2] = g
    return out
