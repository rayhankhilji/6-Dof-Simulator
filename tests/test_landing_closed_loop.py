import numpy as np
import pytest

from sixdof.actuators import ActuatorSuite, GimbalActuator, RCSActuator, ThrottleActuator
from sixdof.control import make_controller
from sixdof.guidance import make_guidance
from sixdof.navigation import EKF
from sixdof.scenarios import landing_scenario, landing_success
from sixdof.sensors import Barometer, GPS, IMU, RadarAltimeter, SensorSuite
from sixdof.simulation import PerfectNavigator, Simulation


def _actuators(veh, rng=None):
    eng = veh.active_stage_config.engine
    return ActuatorSuite(
        gimbal=GimbalActuator(eng.gimbal_max, eng.gimbal_rate_max),
        throttle=ThrottleActuator(eng.throttle_tau, eng.thrust_min_frac, delay_s=0.05),
        rcs=RCSActuator(),
        rng=rng,
    )


def _run(guidance_name, controller_name, nav_kind, seed=0):
    veh, atm, wind, x0, meta = landing_scenario(np.random.default_rng(seed))
    if nav_kind == "ekf":
        nav = EKF(
            x0[0:3] + np.array([3.0, -2.0, 4.0]),
            x0[3:6] + np.array([0.3, -0.2, 0.2]),
            x0[6:10],
            accel_noise_density=100e-6 * 9.80665,
            gyro_noise_density=np.radians(0.005),
            accel_bias_rw=1e-5, gyro_bias_rw=1e-7,
        )
        sensors = SensorSuite(
            imu=IMU(np.random.default_rng(1)),
            gps=GPS(np.random.default_rng(2)),
            barometer=Barometer(np.random.default_rng(3), atm),
            radar=RadarAltimeter(np.random.default_rng(4)),
        )
    else:
        nav = PerfectNavigator()
        sensors = None
    sim = Simulation(
        veh, atm, wind, sensors=sensors, navigator=nav,
        guidance=make_guidance(guidance_name),
        controller=make_controller(controller_name),
        actuators=_actuators(veh), dt=0.01, control_rate_hz=50.0,
        rng=np.random.default_rng(seed + 100),
    )
    res = sim.run(x0, t_end=90.0, stop_on_touchdown=True)
    ok, mode = landing_success(res.touchdown, veh.prop_remaining)
    return res, ok, mode


def test_landing_optimal_mpc_perfect_nav():
    res, ok, mode = _run("optimal", "mpc", "perfect")
    assert ok, f"landing failed: {mode} metrics={res.touchdown}"


def test_landing_zemzev_pid_ekf():
    res, ok, mode = _run("zemzev", "pid", "ekf")
    assert ok, f"landing failed: {mode} metrics={res.touchdown}"
