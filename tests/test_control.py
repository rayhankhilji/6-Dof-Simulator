import numpy as np
import pytest

from sixdof.actuators import ActuatorSuite, GimbalActuator, RCSActuator, ThrottleActuator
from sixdof.control import make_controller
from sixdof.control.allocation import lateral_linear_model
from sixdof.control_types import ControlCommand, GuidanceCommand
from sixdof.dynamics import RigidBodyDynamics, integrate_rk4
from sixdof.environment.atmosphere import USStandardAtmosphere1976
from sixdof.environment.wind import WindModel
from sixdof.math.quaternion import quat_from_axis_angle, quat_rotate
from sixdof.simulation import PerfectNavigator, Simulation
from sixdof.state import IW, initial_state
from sixdof.vehicle import small_landing_vehicle, Vehicle


def _hover_setup():
    veh = Vehicle(small_landing_vehicle())
    veh.prop_remaining = 10_000.0
    atm = USStandardAtmosphere1976()
    dyn = RigidBodyDynamics(veh, atm, WindModel())
    return dyn, veh


def _info_for(veh, dyn, x):
    m, cg, J = veh.mass_properties(x[13])
    st = veh.active_stage_config
    eng = st.engine
    _, p, _, _ = dyn.atmosphere.properties(x[2])
    tmax = eng.thrust_and_mdot(1.0, float(p))[0]
    return {
        "mass": m, "cg_x": cg, "J": J, "J_yy": float(J[1, 1]),
        "l_arm": cg - st.gimbal_point, "thrust_max": tmax,
        "thrust_min": eng.thrust_min_frac * tmax, "g": 9.81,
        "gimbal_max": eng.gimbal_max, "prop_remaining": veh.prop_remaining,
        "stage_index": 0, "stage_prop_capacity": st.prop_mass,
    }


def test_linear_model_signs_vs_nonlinear():
    dyn, veh = _hover_setup()
    m, cg, J = veh.mass_properties()
    l_arm = cg - veh.active_stage_config.gimbal_point
    T = m * 9.81  # hover thrust
    q_up = quat_from_axis_angle([0, 1, 0], -np.pi / 2)
    x = initial_state([0, 0, 100.0], [0, 0, 0], q_up, [0, 0, 0], m)

    throttle = T / veh.active_stage_config.engine.thrust_max_vac
    # +delta_z step: thrust slews toward +y_B (~North) -> vy' > 0; moment
    # about -z_B -> w_z' < 0.
    u = {"throttle": throttle, "gimbal": (0.0, 0.01)}
    xdot = dyn.derivatives(0.0, x, u)
    assert xdot[4] > 0.0      # vy' inertial > 0
    assert xdot[IW][2] < 0.0  # w_z' < 0
    # +delta_y step: thrust slews toward -z_B (~East) -> vx' > 0; moment
    # about -y_B -> w_y' < 0 (nose tips West: the TVC "sign reversal").
    u = {"throttle": throttle, "gimbal": (0.01, 0.0)}
    xdot = dyn.derivatives(0.0, x, u)
    assert xdot[3] > 0.0      # vx' inertial > 0
    assert xdot[IW][1] < 0.0  # w_y' < 0 -> east tilt rate negative

    # The per-channel linear model must reproduce these signs: for a
    # gimbal deflection d, v' > 0 (CG pushed along the slew) and w' < 0
    # (nose tips the opposite way).
    for axis in ("x", "y"):
        A, _ = lateral_linear_model(T, m, l_arm, float(J[1, 1]), 0.05, axis)
        xdot_lin = A @ np.array([0.0, 0.0, 0.0, 0.0, 0.01])
        assert xdot_lin[1] > 0.0, f"axis {axis}: lateral accel sign"
        assert xdot_lin[3] < 0.0, f"axis {axis}: tilt-rate accel sign"


class _HoverGuidance:
    def compute(self, t, nav, info):
        return GuidanceCommand(accel_cmd_I=np.array([0.0, 0.0, info["g"]]),
                               throttle=0.0, engine_on=True, phase="hover",
                               r_ref=np.array([0, 0, 100.0]),
                               v_ref=np.zeros(3))


def _tilt_deg(x):
    xb = quat_rotate(x[6:10], np.array([1.0, 0, 0]))
    return np.degrees(np.arccos(np.clip(xb[2], -1, 1)))


@pytest.mark.parametrize("name", ["pid", "lqr", "nonlinear", "mpc"])
def test_tilt_regulation(name):
    dyn, veh = _hover_setup()
    m = veh.mass_properties()[0]
    eng = veh.active_stage_config.engine
    q0 = quat_from_axis_angle([0, 1, 0], -(np.pi / 2 - np.radians(10.0)))
    x0 = initial_state([0, 0, 100.0], [0, 0, 0], q0, [0, 0, 0], m)

    nav = PerfectNavigator()
    ctrl = make_controller(name)
    act = ActuatorSuite(
        gimbal=GimbalActuator(eng.gimbal_max, eng.gimbal_rate_max),
        throttle=ThrottleActuator(eng.throttle_tau, eng.thrust_min_frac, delay_s=0.0),
        rcs=RCSActuator(),
    )
    sim = Simulation(veh, dyn.atmosphere, dyn.wind, sensors=None,
                     navigator=nav, guidance=_HoverGuidance(), controller=ctrl,
                     actuators=act, dt=0.005, control_rate_hz=50.0)
    res = sim.run(x0, t_end=8.0, stop_on_touchdown=False)
    tilts = np.array([_tilt_deg(row) for row in res.x])
    t6 = np.argmin(np.abs(res.t - 6.0))
    assert tilts[t6] < 1.0, f"{name}: tilt {tilts[t6]:.2f} deg at 6 s"
    assert np.abs(res.u[:, 1:]).max() <= eng.gimbal_max + 1e-9
    assert res.x[-1, 2] > 0.0  # still hovering


def test_mpc_respects_box():
    dyn, veh = _hover_setup()
    m = veh.mass_properties()[0]
    eng = veh.active_stage_config.engine
    q0 = quat_from_axis_angle([0, 1, 0], -(np.pi / 2 - np.radians(15.0)))
    x0 = initial_state([0, 0, 100.0], [0, 0, 0], q0, [0, 0, 0], m)
    nav = PerfectNavigator()
    ctrl = make_controller("mpc")
    act = ActuatorSuite(
        gimbal=GimbalActuator(eng.gimbal_max, eng.gimbal_rate_max),
        throttle=ThrottleActuator(eng.throttle_tau, eng.thrust_min_frac, delay_s=0.0),
        rcs=RCSActuator(),
    )
    sim = Simulation(veh, dyn.atmosphere, dyn.wind, sensors=None,
                     navigator=nav, guidance=_HoverGuidance(), controller=ctrl,
                     actuators=act, dt=0.005, control_rate_hz=50.0)
    res = sim.run(x0, t_end=4.0, stop_on_touchdown=False)
    assert np.abs(res.u[:, 1:]).max() <= eng.gimbal_max + 1e-9
