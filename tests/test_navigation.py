import numpy as np
import pytest

from sixdof.dynamics import RigidBodyDynamics, integrate_rk4
from sixdof.environment.atmosphere import USStandardAtmosphere1976
from sixdof.environment.wind import WindModel
from sixdof.math.quaternion import quat_error, quat_from_axis_angle
from sixdof.navigation import EKF, UKF
from sixdof.sensors import Barometer, GPS, IMU, RadarAltimeter, SensorSuite
from sixdof.state import IQ, IV, initial_state
from sixdof.vehicle import small_landing_vehicle, Vehicle

DT = 0.01
T_END = 30.0


def _truth_trajectory():
    """30 s powered vertical-ish flight with a small initial tilt."""
    veh = Vehicle(small_landing_vehicle())
    atm = USStandardAtmosphere1976()
    dyn = RigidBodyDynamics(veh, atm, WindModel())
    m = veh.mass_properties()[0]
    q0 = quat_from_axis_angle([0.0, 1.0, 0.0], -(np.pi / 2 - np.radians(2.0)))
    x = initial_state([0, 0, 500.0], [0, 0, 0], q0, [0, 0, 0], m)
    u = {"throttle": 0.75, "gimbal": (0.0, 0.0)}
    traj = []
    t = 0.0
    for _ in range(int(T_END / DT)):
        xdot = dyn.derivatives(t, x, u)
        traj.append((t, x.copy(), xdot.copy()))
        x = integrate_rk4(dyn.derivatives, t, x, u, DT)
        dyn.burn_update(u, DT)
        t += DT
    return dyn, traj


@pytest.fixture(scope="module")
def truth():
    return _truth_trajectory()


def _suite(rng, atm, outage=None):
    return SensorSuite(
        imu=IMU(np.random.default_rng(rng)),
        gps=GPS(np.random.default_rng(rng + 1), outage_windows=outage or []),
        barometer=Barometer(np.random.default_rng(rng + 2), atm),
        radar=RadarAltimeter(np.random.default_rng(rng + 3)),
    )


def _run_filter(nav, suite, dyn, traj):
    """Drive a navigator with the sensor suite over the stored truth."""
    last_imu_t = 0.0
    errs = []
    for t, x, xdot in traj:
        meas = suite.sample(t, x, xdot, dyn.atmosphere)
        if meas.imu_accel is not None:
            nav.predict(meas.imu_accel, meas.imu_gyro, max(t - last_imu_t, 1e-6))
            last_imu_t = t
        nav.update(meas)
        est = nav.estimate
        errs.append((t, np.linalg.norm(est.r_I - x[0:3]),
                     np.linalg.norm(est.v_I - x[IV]),
                     np.degrees(np.linalg.norm(quat_error(x[IQ], est.q)))))
    return np.array([(e[0], *e[1:]) for e in errs]), nav


def _fresh_nav(cls, traj, **kw):
    t0, x0, _ = traj[0]
    # Seed with a small initial error to exercise convergence.
    r0 = x0[0:3] + np.array([5.0, -4.0, 6.0])
    v0 = x0[IV] + np.array([0.5, -0.4, 0.3])
    q0 = x0[IQ]
    return cls(r0, v0, q0,
               accel_noise_density=100e-6 * 9.80665,
               gyro_noise_density=np.radians(0.005),
               accel_bias_rw=1e-5, gyro_bias_rw=1e-7,
               t0=0.0, **kw)


def test_ekf_with_gps(truth):
    dyn, traj = truth
    atm = dyn.atmosphere
    nav = _fresh_nav(EKF, traj)
    errs, _ = _run_filter(nav, _suite(10, atm), dyn, traj)
    assert errs[-1, 1] < 3.0     # pos err [m]
    assert errs[-1, 2] < 0.3     # vel err [m/s]
    assert errs[-1, 3] < 0.5     # att err [deg]


def test_ekf_gps_dropout_deadreckon(truth):
    dyn, traj = truth
    nav = _fresh_nav(EKF, traj)
    suite = _suite(20, dyn.atmosphere, outage=[(T_END - 10.0, T_END + 10.0)])
    errs, _ = _run_filter(nav, suite, dyn, traj)
    # Horizontal error grows during the outage but stays bounded.
    assert errs[-1, 1] < 30.0
    assert errs[-1, 1] > errs[np.argmin(np.abs(errs[:, 0] - (T_END - 10))), 1] * 0.5


def test_ekf_nis_consistency(truth):
    dyn, traj = truth
    nav = _fresh_nav(EKF, traj)
    _, nav = _run_filter(nav, _suite(30, dyn.atmosphere), dyn, traj)
    dof = {"gps_pos": 3, "gps_vel": 3, "baro": 1, "radar": 1}
    nis = {}
    for t, kind, innov, S in nav.innovation_log:
        nis.setdefault(kind, []).append(float(innov.T @ np.linalg.solve(S, innov)))
    for kind, vals in nis.items():
        vals = np.array(vals)
        gate = {"gps_pos": 11.34, "gps_vel": 11.34, "baro": 6.63, "radar": 6.63}[kind]
        vals = vals[vals < gate * 3]  # drop gated-out outliers
        mean_nis = vals.mean()
        assert 0.3 * dof[kind] < mean_nis < 3.0 * dof[kind], (kind, mean_nis)


def test_ekf_covariance_spd(truth):
    dyn, traj = truth
    nav = _fresh_nav(EKF, traj)
    _run_filter(nav, _suite(40, dyn.atmosphere), dyn, traj)
    P = nav.estimate.P
    np.testing.assert_allclose(P, P.T, atol=1e-10)
    assert np.all(np.linalg.eigvalsh(P) > -1e-9)


def test_ukf_with_gps(truth):
    dyn, traj = truth
    nav = _fresh_nav(UKF, traj)
    errs, _ = _run_filter(nav, _suite(50, dyn.atmosphere), dyn, traj)
    assert errs[-1, 1] < 5.0
    assert errs[-1, 2] < 0.5
