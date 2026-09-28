"""Closed-loop simulation harness.

Loop per physics step (``dt``, RK4):

    truth -> sensors -> navigation -> guidance -> control -> actuators -> dynamics

IMU samples are taken from the true derivative at that instant; navigation
``predict`` runs at IMU rate and ``update`` consumes whatever sensors
produced that tick. Guidance and controller are duck-typed
(``compute(t, nav_state, vehicle_info)``) and run at ``control_rate_hz``
with zero-order hold between calls. Wind gusts advance once per physics step.

This phase ships ``NullGuidance`` / ``NullController`` (engines off,
centered gimbal) and ``PerfectNavigator`` (returns ground truth) so the loop
can be exercised end to end; real controllers arrive in Phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

import numpy as np

from .control_types import ControlCommand, GuidanceCommand
from .math.quaternion import quat_rotate_inverse
from .dynamics import RigidBodyDynamics, integrate_rk4, touchdown_check
from .environment.gravity import gravity_inertial
from .navigation.base import NavState, Navigator
from .state import IM, IQ, IR, IV, IW, STATE_SIZE, State


class PerfectNavigator(Navigator):
    """Navigator that returns ground truth (for controller tuning/tests)."""

    def __init__(self) -> None:
        self._x: Optional[np.ndarray] = None
        self._t = 0.0
        self.innovation_log: list = []

    def set_truth(self, x: np.ndarray, t: float) -> None:
        self._x = np.asarray(x, dtype=float).copy()
        self._t = float(t)

    def predict(self, imu_accel, imu_gyro, dt) -> None:  # truth-based; no-op
        pass

    def update(self, meas) -> None:
        pass

    @property
    def estimate(self) -> NavState:
        s = State.from_array(self._x)
        return NavState(s.r_I, s.v_I, s.q, np.zeros(3), np.zeros(3),
                        np.zeros((15, 15)), self._t, s.omega_B.copy())


class NullGuidance:
    """Guidance that requests nothing (idle phase)."""

    def compute(self, t, nav_state, vehicle_info) -> GuidanceCommand:
        return GuidanceCommand(phase="coast")


class NullController:
    """Controller that keeps the engine off and the gimbal centered."""

    def compute(self, t, nav_state, guidance_cmd, vehicle_info) -> ControlCommand:
        return ControlCommand(throttle=0.0, gimbal_y=0.0, gimbal_z=0.0)


@dataclass
class SimResult:
    """Time histories from a ``Simulation.run``."""

    t: np.ndarray                 # (N,)
    x: np.ndarray                 # (N, 14) true state
    x_nav: np.ndarray             # (N, 10) nav r,v,q
    P_diag: np.ndarray            # (N, 15) nav covariance diagonal
    u: np.ndarray                 # (N, 3) throttle, gimbal_y, gimbal_z
    phase: np.ndarray             # (N,) guidance phase labels
    aux: np.ndarray               # (N, 4) mach, alpha, q_dyn, thrust
    events: list = field(default_factory=list)          # (t, str)
    touchdown: Optional[dict] = None
    accel_cmd: Optional[np.ndarray] = None    # (N, 3) guidance accel cmd
    q_des: Optional[np.ndarray] = None        # (N, 4) desired attitude
    r_ref: Optional[np.ndarray] = None        # (N, 3)
    v_ref: Optional[np.ndarray] = None        # (N, 3)
    compute_time: Optional[np.ndarray] = None # (N,) controller compute time [s]
    fallback_count: int = 0
    prop_used: float = 0.0

    def save_npz(self, path) -> None:
        np.savez(
            path, t=self.t, x=self.x, x_nav=self.x_nav, P_diag=self.P_diag,
            u=self.u, phase=self.phase, aux=self.aux,
            accel_cmd=self.accel_cmd, q_des=self.q_des,
            r_ref=self.r_ref, v_ref=self.v_ref, compute_time=self.compute_time,
            events=np.array([e[1] for e in self.events]),
            event_times=np.array([e[0] for e in self.events]),
            touchdown=np.array([self.touchdown], dtype=object),
        )

    def summary(self, landing_success_fn=None, prop_remaining: float = 0.0) -> dict:
        """Aggregate metrics: touchdown, fuel, success, effort, compute."""
        out = {
            "t_end": float(self.t[-1]) if len(self.t) else 0.0,
            "fuel_used_kg": float(self.prop_used),
            "fallback_count": int(self.fallback_count),
        }
        if self.touchdown is not None:
            out.update(self.touchdown)
        else:
            out.update({k: float("nan") for k in
                        ("vertical_speed", "lateral_speed", "tilt_angle_deg", "lateral_offset")})
        if landing_success_fn is not None:
            ok, mode = landing_success_fn(self.touchdown, prop_remaining)
            out["success"], out["failure_mode"] = ok, mode
        if len(self.u):
            out["max_gimbal_deg"] = float(np.degrees(np.abs(self.u[:, 1:]).max()))
            dt = np.diff(self.t, prepend=self.t[0])
            out["control_effort"] = float(np.sum((self.u[:, 1:] ** 2) * dt[:, None]))
        if len(self.x):
            lat_err = np.hypot(self.x[:, 0], self.x[:, 1])
            out["rms_lateral_offset"] = float(np.sqrt(np.mean(lat_err**2)))
        if self.compute_time is not None and len(self.compute_time):
            out["compute_time_mean_ms"] = float(np.mean(self.compute_time) * 1e3)
            out["compute_time_max_ms"] = float(np.max(self.compute_time) * 1e3)
        return out


class Simulation:
    """Closed-loop harness.

    Parameters
    ----------
    vehicle : Vehicle
    atmosphere, wind : environment models
    sensors : SensorSuite or None
    navigator : Navigator or None (no nav updates if None)
    guidance, controller : duck-typed compute() callables
    actuators : ActuatorSuite or None (commands pass through if None)
    dt : float
        Physics step [s].
    control_rate_hz : float
        Guidance/control call rate.
    rng : np.random.Generator
        For wind gusts and any stochastic elements owned by the sim.
    """

    def __init__(
        self,
        vehicle,
        atmosphere,
        wind,
        sensors=None,
        navigator: Optional[Navigator] = None,
        guidance=None,
        controller=None,
        actuators=None,
        dt: float = 0.01,
        control_rate_hz: float = 50.0,
        rng: Optional[np.random.Generator] = None,
        staging_callback=None,
    ) -> None:
        self.staging_callback = staging_callback
        self._staged = False
        self._last_gyro: Optional[np.ndarray] = None
        self.vehicle = vehicle
        self.atmosphere = atmosphere
        self.wind = wind
        self.sensors = sensors
        self.navigator = navigator
        self.guidance = guidance or NullGuidance()
        self.controller = controller or NullController()
        self.actuators = actuators
        self.dt = float(dt)
        self.control_period = 1.0 / float(control_rate_hz)
        self.rng = rng or np.random.default_rng(0)
        self.dyn = RigidBodyDynamics(vehicle, atmosphere, wind)
        self.env = SimpleNamespace(atmosphere=atmosphere, wind=wind)

    def _vehicle_info(self, x) -> dict:
        m, cg, J = self.vehicle.mass_properties(x[IM])
        stage = self.vehicle.active_stage_config
        eng = stage.engine
        _, p_amb, _, _ = self.atmosphere.properties(x[IR][2])
        t_max = eng.thrust_and_mdot(1.0, float(p_amb))[0]
        t_min = eng.thrust_and_mdot(eng.thrust_min_frac, float(p_amb))[0]
        g = float(np.linalg.norm(gravity_inertial(x[IR])))
        gimbal_x = stage.gimbal_point + self.vehicle.config.stage_base(self.vehicle.active_stage)
        return {
            "mass": m, "cg_x": cg, "J": J, "J_yy": float(J[1, 1]),
            "l_arm": max(cg - gimbal_x, 0.5),
            "stage": stage, "engine": eng,
            "thrust_max": t_max, "thrust_min": t_min, "g": g,
            "prop_remaining": self.vehicle.prop_remaining,
            "stage_index": self.vehicle.active_stage,
            "stage_prop_capacity": stage.prop_mass,
            "gimbal_max": eng.gimbal_max, "gimbal_tau": eng.gimbal_tau,
        }

    def run(self, x0: np.ndarray, t_end: float, stop_on_touchdown: bool = True) -> SimResult:
        dt = self.dt
        x = np.asarray(x0, dtype=float).copy()
        t = 0.0
        u = {"throttle": 0.0, "gimbal": (0.0, 0.0)}
        ucmd = ControlCommand()
        phase = "init"
        last_control_t = -np.inf
        last_nav_t = 0.0
        events: list = []

        T, X, XN, PD, U, PH, AX = [], [], [], [], [], [], []
        AC, QD, RR, VR, CT = [], [], [], [], []
        gcmd = GuidanceCommand()
        prop0 = self.vehicle.prop_remaining
        n_steps = int(np.ceil(t_end / dt))
        for _ in range(n_steps):
            # True derivative at this instant (for IMU specific force).
            xdot = self.dyn.derivatives(t, x, u)

            # Sensors + navigation.
            meas = self.sensors.sample(t, x, xdot, self.env) if self.sensors else None
            if self.navigator is not None:
                if hasattr(self.navigator, "set_truth"):
                    self.navigator.set_truth(x, t)
                else:
                    if meas is not None and meas.imu_accel is not None:
                        self.navigator.predict(meas.imu_accel, meas.imu_gyro,
                                               max(t - last_nav_t, 1e-6))
                        last_nav_t = t
                        self._last_gyro = meas.imu_gyro
                    if meas is not None:
                        self.navigator.update(meas)
                    # Feed estimated body rate (gyro minus bias) to controllers.
                    if self._last_gyro is not None:
                        est = self.navigator.estimate
                        est.omega_B = self._last_gyro - est.gyro_bias

            # Guidance / control at the control rate (zero-order hold).
            if t - last_control_t >= self.control_period - 1e-9:
                nav_state = self.navigator.estimate if self.navigator else None
                vinfo = self._vehicle_info(x)
                gcmd = self.guidance.compute(t, nav_state, vinfo)
                phase = getattr(gcmd, "phase", "run")
                # Staging hook: guidance requests separation.
                if (self.staging_callback is not None
                        and phase == "stage_sep_coast" and not self._staged
                        and self.vehicle.active_stage + 1 < len(self.vehicle.config.stages)):
                    self.staging_callback(self.vehicle, t)
                    self._staged = True
                    events.append((t, "stage_sep"))
                ucmd = self.controller.compute(t, nav_state, gcmd, vinfo)
                last_control_t = t

            if self.actuators is not None:
                u = self.actuators.step(ucmd, t, dt)
            else:
                u = {"throttle": ucmd.throttle, "gimbal": (ucmd.gimbal_y, ucmd.gimbal_z)}

            # Physics step.
            x = integrate_rk4(self.dyn.derivatives, t, x, u, dt)
            self.dyn.burn_update(u, dt)
            airspeed = float(np.linalg.norm(x[IV] - self.wind.wind_at(x[IR])))
            self.wind.step(dt, airspeed, self.rng)
            t += dt

            # Record.
            T.append(t)
            X.append(x.copy())
            if self.navigator is not None:
                est = self.navigator.estimate
                XN.append(np.concatenate([est.r_I, est.v_I, est.q]))
                PD.append(np.diag(est.P))
            else:
                XN.append(np.full(10, np.nan))
                PD.append(np.full(15, np.nan))
            U.append([u["throttle"], u["gimbal"][0], u["gimbal"][1]])
            PH.append(phase)
            AC.append(np.asarray(gcmd.accel_cmd_I, dtype=float))
            QD.append(np.asarray(gcmd.q_des, dtype=float)
                      if gcmd.q_des is not None else np.full(4, np.nan))
            RR.append(np.asarray(gcmd.r_ref, dtype=float)
                      if gcmd.r_ref is not None else np.full(3, np.nan))
            VR.append(np.asarray(gcmd.v_ref, dtype=float)
                      if gcmd.v_ref is not None else np.full(3, np.nan))
            CT.append(getattr(self.controller, "compute_time", np.nan))
            _, _, rho, a_snd = self.atmosphere.properties(x[IR][2])
            v_rel_I = x[IV] - self.wind.wind_at(x[IR])
            V = np.linalg.norm(v_rel_I)
            v_rel_B = quat_rotate_inverse(x[IQ], v_rel_I)
            alpha = np.arctan2(np.hypot(v_rel_B[1], v_rel_B[2]), v_rel_B[0]) if V > 1e-3 else 0.0
            AX.append([V / a_snd if a_snd > 0 else 0.0, alpha,
                       0.5 * float(rho) * V * V, u["throttle"]])

            for ev in self.vehicle.events:
                pair = (ev["t"], ev["event"])
                if pair not in events:
                    events.append(pair)
            touched, metrics = touchdown_check(x)
            if stop_on_touchdown and touched:
                events.append((t, "touchdown"))
                return self._result(T, X, XN, PD, U, PH, AX, AC, QD, RR, VR, CT,
                                    events, metrics, prop0)

        return self._result(T, X, XN, PD, U, PH, AX, AC, QD, RR, VR, CT,
                            events, None, prop0)

    def _result(self, T, X, XN, PD, U, PH, AX, AC, QD, RR, VR, CT,
                events, touchdown, prop0) -> SimResult:
        return SimResult(
            np.array(T), np.array(X), np.array(XN), np.array(PD),
            np.array(U), np.array(PH, dtype=object), np.array(AX),
            events=events, touchdown=touchdown,
            accel_cmd=np.array(AC), q_des=np.array(QD),
            r_ref=np.array(RR), v_ref=np.array(VR),
            compute_time=np.array(CT),
            fallback_count=int(getattr(self.guidance, "fallback_count", 0)),
            prop_used=prop0 - self.vehicle.prop_remaining,
        )
