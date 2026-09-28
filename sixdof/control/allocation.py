"""Thrust/attitude allocation and the linear lateral model for TVC.

Given a desired thrust acceleration ``a_cmd_I`` [m/s^2], the thrust
magnitude is ``T = clip(m |a|, T_min, T_max)`` and the desired body x-axis
is ``unit(a_cmd_I)``; ``q_des`` is the minimal rotation from the current
attitude to that direction (roll left free).

Lateral linear model per *horizontal inertial* channel, state
``[p, v, th, w, d]`` with ``th`` the small tilt of the thrust axis toward
that inertial direction and ``d`` the driving gimbal angle (first-order
lag ``tau``). The roll reference is the canonical descent attitude
(``x_B ~ Up``, ``y_B ~ North``, ``z_B ~ West``); under that convention
both channels share the same dynamics:

    v' = (T/m)(th + d),   th' = w,   w' = -(T l/J) d,   d' = (d_cmd - d)/tau

i.e. the gimbal accelerates the CG *toward* its slew direction while the
moment about the CG tips the nose the *opposite* way (the classic TVC
geometry for an engine mounted below the CG):

- East channel driven by ``d = dy``: +dy slews the thrust toward -z_B
  (which is +x_I, East) so ``v_x' > 0``; the moment is about -y_B so the
  nose tips West (``th_e' = +w_y``, ``w_y' = -(T l/J) dy``).
- North channel driven by ``d = dz``: +dz slews the thrust toward +y_B
  (North) so ``v_y' > 0``; the moment is about -z_B so the nose tips
  South (``th_n' = +w_z``, ``w_z' = -(T l/J) dz``).

Signs are verified against the nonlinear dynamics in test_control.py.
"""

from __future__ import annotations

import numpy as np

from ..math.quaternion import (
    quat_from_two_vectors,
    quat_multiply,
    quat_rotate,
)

_EPS_A = 1e-3


def thrust_and_attitude_from_accel(
    a_cmd_I: np.ndarray, m: float, T_min: float, T_max: float, q_now: np.ndarray
) -> tuple[float, np.ndarray]:
    """Allocate thrust magnitude and desired attitude from ``a_cmd_I``.

    Returns ``(throttle, q_des)`` with throttle in [0,1] (relative to T_max).
    """
    a_cmd_I = np.asarray(a_cmd_I, dtype=float)
    a_mag = np.linalg.norm(a_cmd_I)
    T = float(np.clip(m * a_mag, T_min, T_max))
    throttle = T / T_max

    if a_mag < _EPS_A:
        dir_des = np.array([0.0, 0.0, 1.0])
    else:
        dir_des = a_cmd_I / a_mag
    x_b_now = quat_rotate(q_now, np.array([1.0, 0.0, 0.0]))
    q_delta = quat_from_two_vectors(x_b_now, dir_des)
    q_des = quat_multiply(q_delta, q_now)
    return throttle, q_des


def lateral_linear_model(T: float, m: float, l_arm: float, J_yy: float,
                         tau_gimbal: float, axis: str = "y"):
    """Continuous-time (A, B) for a horizontal lateral channel.

    ``axis="y"`` (North) is driven by gimbal_z; ``axis="x"`` (East) is
    driven by gimbal_y. Under the canonical descent roll convention
    (``y_B ~ North``, ``z_B ~ West``) both channels share the same
    (A, B); the ``axis`` argument only documents which gimbal drives it.
    """
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    acc_th, acc_d, mom_d = T / m, T / m, -T * l_arm / J_yy
    A = np.array([
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, acc_th, 0.0, acc_d],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, mom_d],
        [0.0, 0.0, 0.0, 0.0, -1.0 / tau_gimbal],
    ])
    B = np.array([[0.0], [0.0], [0.0], [0.0], [1.0 / tau_gimbal]])
    return A, B


def lateral_tilts(q: np.ndarray) -> tuple[float, float]:
    """Small-angle tilt of body +x toward inertial East (``th_e``) and
    North (``th_n``): components of the thrust axis along x_I / y_I."""
    x_b_I = quat_rotate(q, np.array([1.0, 0.0, 0.0]))
    return float(x_b_I[0]), float(x_b_I[1])


def lateral_rates(omega_B: np.ndarray) -> tuple[float, float]:
    """(d/dt th_east, d/dt th_north) from body rates for a near-upright
    vehicle under the canonical roll convention (``y_B ~ North``,
    ``z_B ~ West``): th_e' = +w_y (nose East via +y_B rotation),
    th_n' = +w_z."""
    return float(omega_B[1]), float(omega_B[2])


def lateral_tilts_from_dir(dir_I: np.ndarray) -> tuple[float, float]:
    """Tilts (east, north) implied by a desired thrust direction."""
    d = np.asarray(dir_I, dtype=float)
    return float(d[0]), float(d[1])
