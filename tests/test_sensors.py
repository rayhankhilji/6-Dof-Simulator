import numpy as np
import pytest

from sixdof.dynamics import RigidBodyDynamics
from sixdof.environment.atmosphere import USStandardAtmosphere1976
from sixdof.environment.wind import WindModel
from sixdof.math.quaternion import quat_from_axis_angle
from sixdof.sensors import Barometer, GPS, IMU, RadarAltimeter, SensorSuite
from sixdof.state import IV, initial_state
from sixdof.vehicle import small_landing_vehicle, Vehicle

# x_B pointing up (+z_I).
Q_UP = quat_from_axis_angle([0.0, 1.0, 0.0], -np.pi / 2)


def _truth(z=0.0, q=None):
    veh = Vehicle(small_landing_vehicle())
    m = veh.mass_properties()[0]
    return initial_state([0, 0, z], [0, 0, 0], q if q is not None else Q_UP,
                         [0, 0, 0], m)


def test_imu_at_rest_specific_force():
    rng = np.random.default_rng(0)
    imu = IMU(rng)
    x = _truth(z=10.0)
    xdot = np.zeros(14)  # at rest: a_I = 0 -> f_B = -R^T g = +g along x_B
    accels, gyros = [], []
    for i in range(200):
        out = imu.sample(i * imu.period, x, xdot, None)
        if out is not None:
            accels.append(out[0])
            gyros.append(out[1])
    accels = np.array(accels)
    # Specific force ~ +g along body x (up), other axes ~0; bias within 5 sigma.
    g = 9.81
    assert np.abs(accels[:, 0].mean() - g) < 5 * 9.80665e-3 + 5 * imu.accel_noise / np.sqrt(len(accels))
    assert np.abs(accels[:, 1].mean()) < 5 * 9.80665e-3 + 5 * imu.accel_noise / np.sqrt(len(accels))
    assert np.abs(np.array(gyros).mean(axis=0)).max() < 5 * np.radians(1.0 / 3600.0) + 5 * imu.gyro_noise / np.sqrt(len(gyros))


def test_gps_latency_and_outage():
    rng = np.random.default_rng(1)
    gps = GPS(rng, delay_s=0.1, outage_windows=[(5.0, 6.0)], p_dropout=0.0)
    x = _truth(z=100.0)
    xdot = np.zeros(14)
    got = {}
    t = 0.0
    while t < 2.0:
        out = gps.sample(t, x, xdot, None)
        if out is not None:
            got[round(t, 3)] = out
        t += 0.02
    # Measurement at t=1.0 corresponds to truth at ~0.9.
    assert got[1.0][2] == pytest.approx(0.9, abs=0.02)
    # Outage window produces nothing.
    t = 5.0
    none_count = 0
    while t < 6.0:
        if gps.sample(t, x, xdot, None) is None:
            none_count += 1
        t += 0.1
    assert none_count > 5


def test_baro_round_trip():
    rng = np.random.default_rng(2)
    atm = USStandardAtmosphere1976()
    baro = Barometer(rng, atm)
    baro.noise_sigma = 0.0
    baro.bias_sigma = 0.0
    for h in (0.0, 5000.0, 15000.0, 30000.0):
        _, p, _, _ = atm.properties(h)
        assert baro.pressure_to_altitude(float(p)) == pytest.approx(h, abs=1.0)


def test_radar_slant_range():
    rng = np.random.default_rng(3)
    radar = RadarAltimeter(rng)
    xdot = np.zeros(14)
    # Tilted 20 deg about y: x_B pitched from vertical.
    tilt = np.radians(20.0)
    q = quat_from_axis_angle([0, 1, 0], -(np.pi / 2 - tilt))
    x = _truth(z=500.0, q=q)
    out = radar.sample(0.0, x, xdot, None)
    assert out is not None
    assert out == pytest.approx(500.0 / np.cos(tilt), rel=0.02)
    # Above max range -> None.
    x_hi = _truth(z=3000.0)
    assert radar.sample(1.0, x_hi, xdot, None) is None
    # Tilted 45 deg -> invalid.
    q_bad = quat_from_axis_angle([0, 1, 0], -(np.pi / 2 - np.radians(45)))
    x_bad = _truth(z=500.0, q=q_bad)
    assert radar.sample(2.0, x_bad, xdot, None) is None


def test_sensor_suite_fields():
    rng = np.random.default_rng(4)
    atm = USStandardAtmosphere1976()
    suite = SensorSuite(
        imu=IMU(np.random.default_rng(5)),
        gps=GPS(np.random.default_rng(6)),
        barometer=Barometer(np.random.default_rng(7), atm),
        radar=RadarAltimeter(np.random.default_rng(8)),
    )
    x = _truth(z=100.0)
    xdot = np.zeros(14)
    m = suite.sample(0.0, x, xdot, atm)
    assert m.imu_accel is not None and m.imu_gyro is not None
    assert m.gps_pos is not None and m.gps_t_valid == pytest.approx(0.0)
    assert m.baro_alt is not None and m.baro_sigma > 0
    assert m.radar_range is not None
