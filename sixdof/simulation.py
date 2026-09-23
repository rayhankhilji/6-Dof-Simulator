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
                        np.zeros((15, 15)), self._t)


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

    def save_npz(self, path) -> None:
        np.savez(
            path, t=self.t, x=self.x, x_nav=self.x_nav, P_diag=self.P_diag,
            u=self.u, phase=self.phase, aux=self.aux,
            events=np.array([e[1] for e in self.events]),
            event_times=np.array([e[0] for e in self.events]),
            touchdown=np.array([self.touchdown], dtype=object),
        )


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
    ) -> None:
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
        return {
            "mass": m, "cg_x": cg, "J": J, "stage": stage,
            "engine": stage.engine, "prop_remaining": self.vehicle.prop_remaining,
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
        n_steps = int(np.ceil(t_end / dt))
        for _ in range(n_steps):
            # True derivative at this instant (for IMU specific force).
            xdot = self.dyn.derivatives(t, x, u)

            # Sensors + navigation.
            meas = self.sensors.sample(t, x, xdot, self.env) if self.sensors else None
            if self.navigator is not None:
                if hasattr(self.navigator, "set_truth"):
                    self.navigator.set_truth(x, t)
                elif meas is not None and meas.imu_accel is not None:
                    self.navigator.predict(meas.imu_accel, meas.imu_gyro, t - last_nav_t)
                    last_nav_t = t
                if meas is not None:
                    self.navigator.update(meas)

            # Guidance / control at the control rate (zero-order hold).
            if t - last_control_t >= self.control_period - 1e-9:
                nav_state = self.navigator.estimate if self.navigator else None
                vinfo = self._vehicle_info(x)
                gcmd = self.guidance.compute(t, nav_state, vinfo)
                phase = getattr(gcmd, "phase", "run")
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
                return SimResult(
                    np.array(T), np.array(X), np.array(XN), np.array(PD),
                    np.array(U), np.array(PH), np.array(AX),
                    events=events, touchdown=metrics,
                )

        return SimResult(
            np.array(T), np.array(X), np.array(XN), np.array(PD),
            np.array(U), np.array(PH), np.array(AX), events=events,
        )
