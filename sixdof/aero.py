"""Aerodynamic forces and moments in the body frame.

Relative wind:

    v_rel_I = v_I - wind_I,   v_rel_B = R^T v_rel_I,   V = |v_rel_B|

Total angle of attack (angle between body x-axis and the relative wind):

    alpha = atan2(sqrt(v_y^2 + v_z^2), v_x)

Mach = V / a(h),  dynamic pressure q = 0.5 rho V^2.

Force model (body frame):
- Axial drag: -Cd(M) q S along the relative wind direction in body frame.
- Normal force: N = -Cn_alpha * sin(alpha) * q S * e_n, where ``e_n`` is
  the direction of the lateral component of ``v_rel_B`` -- i.e. the body
  is pushed *opposite* its sideways motion through the air (weathervane
  convention), acting at the center of pressure x_cp. The sin(alpha)
  factor keeps the small-angle Cn_alpha*alpha slope but vanishes for
  aligned base-first flight (alpha ~ 180 deg). With CP behind the
  CG relative to the direction of travel this is statically restoring;
  e.g. for base-first descent (v_rel_B ~ -x_B) CP *above* the CG
  stabilizes, for nose-first ascent CP *below* the CG stabilizes.
- Moment from normal force: M = (x_cp - x_cg) x_hat x F_normal_B.
- Damping moments: M += q S L * c_damp * (omega L / (2V)) for pitch/yaw
  (cm_damping) and roll (cl_roll_damping).

For V < 1e-3 m/s all outputs are zero.
"""

from __future__ import annotations

import numpy as np

from .math.quaternion import quat_rotate_inverse

_V_MIN = 1e-3


def aero_forces_moments(
    r_I: np.ndarray,
    v_I: np.ndarray,
    q: np.ndarray,
    omega_B: np.ndarray,
    aero,
    cp_x: float,
    cg_x: float,
    length: float,
    atm,
    wind_I: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compute aerodynamic force and moment in the body frame.

    Parameters
    ----------
    r_I, v_I : (3,) arrays
        Inertial position and velocity.
    q : (4,) array
        Body-to-inertial attitude quaternion.
    omega_B : (3,) array
        Body angular rate in body frame [rad/s].
    aero : AeroConfig
        Aerodynamic coefficients.
    cp_x, cg_x : float
        Center of pressure and CG along x_B from the vehicle base [m].
    length : float
        Reference length [m].
    atm : USStandardAtmosphere1976
    wind_I : (3,) array
        Wind velocity in the inertial frame [m/s].

    Returns
    -------
    F_B, M_B : (3,) arrays
        Aero force and moment in the body frame [N], [N m].
    aux : dict
        Diagnostic quantities: mach, alpha, q_dyn, v_rel_B.
    """
    v_rel_I = np.asarray(v_I, dtype=float) - np.asarray(wind_I, dtype=float)
    v_rel_B = quat_rotate_inverse(q, v_rel_I)
    V = np.linalg.norm(v_rel_B)
    aux = {"mach": 0.0, "alpha": 0.0, "q_dyn": 0.0, "v_rel_B": v_rel_B}
    if V < _V_MIN:
        return np.zeros(3), np.zeros(3), aux

    _, _, rho, a = atm.properties(float(r_I[2]))
    mach = V / float(a)
    q_dyn = 0.5 * float(rho) * V * V

    alpha = np.arctan2(np.hypot(v_rel_B[1], v_rel_B[2]), v_rel_B[0])

    # Axial drag opposite the relative wind.
    e_rel = v_rel_B / V
    F_drag = -aero.cd(mach) * q_dyn * aero.ref_area * e_rel

    # Normal force perpendicular to x_B in the plane of v_rel, opposing the
    # lateral component of the air-relative velocity (weathervane force).
    # Magnitude scales with sin(alpha) -- equal to the classic Cn_alpha*alpha
    # slender-body slope at small alpha, but correctly vanishing near
    # alpha = 180 deg (aligned base-first descent), where the linear-in-alpha
    # form would wrongly produce its largest force.
    e_n_plane = np.array([0.0, v_rel_B[1], v_rel_B[2]])
    n_norm = np.linalg.norm(e_n_plane)
    if n_norm > 1e-9:
        e_n_plane /= n_norm
    F_normal = -aero.cn_alpha * np.sin(alpha) * q_dyn * aero.ref_area * e_n_plane

    F_B = F_drag + F_normal

    # Moment: normal force applied at CP relative to CG.
    r_cp = np.array([cp_x - cg_x, 0.0, 0.0])
    M_B = np.cross(r_cp, F_normal)

    # Damping moments, nondimensional rate omega * L / (2V).
    w_hat = omega_B * length / (2.0 * V)
    qsl = q_dyn * aero.ref_area * length
    M_B += qsl * np.array(
        [aero.cl_roll_damping * w_hat[0], aero.cm_damping * w_hat[1], aero.cm_damping * w_hat[2]]
    )

    aux.update(mach=mach, alpha=alpha, q_dyn=q_dyn)
    return F_B, M_B, aux
