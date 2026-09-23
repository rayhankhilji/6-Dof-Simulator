"""Rigid-body 6-DOF equations of motion and integrators.

State ``x = [r_I, v_I, q, omega_B, m]`` (see ``sixdof.state``). Equations:

    r_dot   = v_I
    v_dot   = R(q) F_B / m + g(r_I)
    q_dot   = 0.5 q x [0, omega_B]
    J w_dot = M_B - omega_B x (J omega_B)     (J time-varying; dJ/dt neglected)
    m_dot   = -mdot_engine

Thrust model: the engine at the gimbal point deflects by (delta_y, delta_z)
[rad]. The thrust unit vector in the body frame is

    t_B = [cos(dy) cos(dz),  sin(dz),  -sin(dy) cos(dz)]

so a positive delta_z slews the thrust toward +y_B, producing a moment
about -z_B (nose yaws toward -y_B: thrust steers the vehicle opposite to
the nozzle deflection, as for a real gimbaled engine). A positive delta_y
slews the thrust toward -z_B, producing a moment about -y_B. Both signs
assume the gimbal point below the CG (r_gimbal - r_cg along -x_B).

Thrust acts at the gimbal point: M = (r_gimbal - r_cg) x F_thrust_B.

Earth rotation and the dJ/dt term in Euler's equation are neglected
(flat-Earth ENU frame).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .aero import aero_forces_moments
from .environment.gravity import gravity_inertial
from .math.quaternion import (
    quat_derivative,
    quat_normalize,
    quat_rotate,
)
from .state import IM, IQ, IR, IV, IW, State


def thrust_direction_body(delta_y: float, delta_z: float) -> np.ndarray:
    """Unit thrust direction in the body frame for gimbal angles (dy, dz)."""
    return np.array(
        [
            np.cos(delta_y) * np.cos(delta_z),
            np.sin(delta_z),
            -np.sin(delta_y) * np.cos(delta_z),
        ]
    )


class RigidBodyDynamics:
    """6-DOF equations of motion for a ``Vehicle``.

    Parameters
    ----------
    vehicle : Vehicle
    atmosphere : USStandardAtmosphere1976
    wind : WindModel
    """

    def __init__(self, vehicle, atmosphere, wind) -> None:
        self.vehicle = vehicle
        self.atmosphere = atmosphere
        self.wind = wind

    def derivatives(self, t: float, x: np.ndarray, u: dict) -> np.ndarray:
        """State derivative ``xdot`` for control ``u``.

        ``u`` keys: ``throttle`` in [0,1] (post-actuator commanded value),
        ``gimbal`` = (delta_y, delta_z) [rad]. Throttle handling lives in
        ``EngineConfig.thrust_and_mdot``: 0 = engine off; any nonzero command
        below ``thrust_min_frac`` is clamped up to the throttleable floor.
        """
        s = State.from_array(x)
        veh = self.vehicle
        stage = veh.active_stage_config

        # State mass is authoritative; prop_remaining (and hence CG/J) is
        # derived from it so mass properties track the integrated mass.
        m, cg_x, J = veh.mass_properties(s.m)

        # --- Thrust ---
        throttle = float(u.get("throttle", 0.0))
        gimbal = np.asarray(u.get("gimbal", (0.0, 0.0)), dtype=float)
        _, p_amb, _, _ = self.atmosphere.properties(s.r_I[2])
        dry = veh.config._dry_mass_at_or_below(veh.active_stage)
        prop = s.m - dry
        if prop <= 0.0 or throttle <= 0.0:
            thrust_mag, mdot = 0.0, 0.0
        else:
            thrust_mag, mdot = stage.engine.thrust_and_mdot(throttle, float(p_amb))
        F_thrust_B = thrust_mag * thrust_direction_body(gimbal[0], gimbal[1])

        # --- Aerodynamics ---
        wind_I = self.wind.wind_at(s.r_I)
        F_aero_B, M_aero_B, _ = aero_forces_moments(
            s.r_I, s.v_I, s.q, s.omega_B,
            stage.aero, stage.cp_offset, cg_x, stage.length,
            self.atmosphere, wind_I,
        )

        F_B = F_thrust_B + F_aero_B
        g_I = gravity_inertial(s.r_I)
        F_I = quat_rotate(s.q, F_B) + m * g_I

        # Moments about the CG.
        r_gimbal = np.array([stage.gimbal_point + veh.config.stage_base(veh.active_stage) - cg_x, 0.0, 0.0])
        M_B = np.cross(r_gimbal, F_thrust_B) + M_aero_B

        xdot = np.zeros(14)
        xdot[IR] = s.v_I
        xdot[IV] = F_I / m
        xdot[IQ] = quat_derivative(s.q, s.omega_B)
        xdot[IW] = np.linalg.solve(J, M_B - np.cross(s.omega_B, J @ s.omega_B))
        xdot[IM] = -mdot
        return xdot

    def burn_update(self, u: dict, dt: float) -> None:
        """Consume propellant for a step (keeps ``prop_remaining`` in sync)."""
        throttle = float(u.get("throttle", 0.0))
        if throttle <= 0.0 or self.vehicle.prop_remaining <= 0.0:
            return
        stage = self.vehicle.active_stage_config
        t_vac = min(max(throttle, stage.engine.thrust_min_frac), 1.0) * stage.engine.thrust_max_vac
        self.vehicle.burn(t_vac / (stage.engine.isp_vac * 9.80665), dt)


def integrate_rk4(f, t: float, x: np.ndarray, u: dict, dt: float) -> np.ndarray:
    """One classic RK4 step of ``xdot = f(t, x, u)``; quaternion renormalized."""
    k1 = f(t, x, u)
    k2 = f(t + dt / 2.0, x + dt * k1 / 2.0, u)
    k3 = f(t + dt / 2.0, x + dt * k2 / 2.0, u)
    k4 = f(t + dt, x + dt * k3, u)
    x_new = x + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    x_new[IQ] = quat_normalize(x_new[IQ])
    return x_new


def integrate_step(f, t: float, x: np.ndarray, u: dict, dt: float, method: str = "rk4") -> np.ndarray:
    """Advance one step; ``method`` is 'rk4' (default) or 'rk45' (adaptive)."""
    if method == "rk4":
        return integrate_rk4(f, t, x, u, dt)
    if method == "rk45":
        sol = solve_ivp(
            lambda tt, xx: f(tt, xx, u),
            (t, t + dt),
            x,
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
        )
        x_new = sol.y[:, -1]
        x_new[IQ] = quat_normalize(x_new[IQ])
        return x_new
    raise ValueError(f"unknown integration method {method!r}")


def touchdown_check(x: np.ndarray) -> tuple[bool, dict]:
    """Return ``(touched, metrics)``; ``touched`` when z <= 0.

    Metrics: vertical_speed [m/s], lateral_speed [m/s], tilt_angle_deg
    (angle between body x-axis and inertial z), lateral_offset [m].
    """
    s = State.from_array(x)
    touched = s.r_I[2] <= 0.0
    x_axis_I = quat_rotate(s.q, np.array([1.0, 0.0, 0.0]))
    tilt = np.degrees(np.arccos(np.clip(x_axis_I[2], -1.0, 1.0)))
    metrics = {
        "vertical_speed": abs(s.v_I[2]),
        "lateral_speed": float(np.hypot(s.v_I[0], s.v_I[1])),
        "tilt_angle_deg": float(tilt),
        "lateral_offset": float(np.hypot(s.r_I[0], s.r_I[1])),
    }
    return touched, metrics
