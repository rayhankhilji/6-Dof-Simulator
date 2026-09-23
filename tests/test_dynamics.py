import numpy as np
import pytest

from sixdof.dynamics import RigidBodyDynamics, integrate_rk4, touchdown_check
from sixdof.environment.atmosphere import USStandardAtmosphere1976
from sixdof.environment.gravity import MU_EARTH, R_EARTH, gravity_inertial
from sixdof.environment.wind import WindModel
from sixdof.state import IM, IQ, IV, IW, initial_state
from sixdof.math.quaternion import quat_from_axis_angle
from sixdof.vehicle import small_landing_vehicle, two_stage_launcher, Vehicle

# Attitude with body x_B (thrust axis, engine->nose) aligned to +z_I (up).
Q_UPRIGHT = quat_from_axis_angle([0.0, 1.0, 0.0], -np.pi / 2)


def make_dyn(vacuum=False):
    cfg = small_landing_vehicle()
    veh = Vehicle(cfg)
    atm = USStandardAtmosphere1976(density_scale=0.0 if vacuum else 1.0)
    wind = WindModel()
    return RigidBodyDynamics(veh, atm, wind), veh


def upright_state(veh, z=1000.0, m=None):
    m = veh.mass_properties()[0] if m is None else m
    return initial_state(
        r=[0.0, 0.0, z], v=[0.0, 0.0, 0.0], q=Q_UPRIGHT,
        w=[0.0, 0.0, 0.0], m=m,
    )


def test_free_fall_vacuum():
    dyn, veh = make_dyn(vacuum=True)
    x = upright_state(veh)
    u = {"throttle": 0.0, "gimbal": (0.0, 0.0)}
    dt = 0.005
    for _ in range(int(1.0 / dt)):
        x = integrate_rk4(dyn.derivatives, 0.0, x, u, dt)
    g = MU_EARTH / (R_EARTH + 1000.0) ** 2
    assert x[2] == pytest.approx(1000.0 - 0.5 * g * 1.0, abs=1e-3)
    assert np.linalg.norm(x[IQ]) == pytest.approx(1.0, abs=1e-12)
    assert x[IM] == pytest.approx(veh.mass_properties()[0])


def test_torque_free_spin():
    dyn, veh = make_dyn(vacuum=True)
    x = upright_state(veh)
    x[IW] = np.array([0.3, 0.0, 0.0])  # spin about principal axis x_B
    u = {"throttle": 0.0, "gimbal": (0.0, 0.0)}
    w0 = np.linalg.norm(x[IW])
    for _ in range(200):
        x = integrate_rk4(dyn.derivatives, 0.0, x, u, 0.005)
    assert np.linalg.norm(x[IW]) == pytest.approx(w0, abs=1e-10)


def test_hover():
    dyn, veh = make_dyn()  # sea-level-ish atmosphere; hover at 500 m
    x = upright_state(veh, z=500.0)
    m = x[IM]
    stage = veh.active_stage_config
    _, p, _, _ = dyn.atmosphere.properties(500.0)
    g = MU_EARTH / (R_EARTH + 500.0) ** 2
    # Throttle so delivered thrust equals weight.
    throttle = (m * g + p * stage.engine.nozzle_exit_area * stage.engine.n_engines) / (
        stage.engine.thrust_max_vac
    )
    u = {"throttle": throttle, "gimbal": (0.0, 0.0)}
    dt = 0.005
    v0 = x[5]
    for _ in range(int(1.0 / dt)):
        x = integrate_rk4(dyn.derivatives, 0.0, x, u, dt)
        dyn.burn_update(u, dt)
    assert abs(x[5] - v0) < 0.05


def test_gimbal_sign_convention():
    dyn, veh = make_dyn(vacuum=True)
    x = upright_state(veh)
    # Positive delta_z must give M_z < 0 (thrust slewed +y_B with pivot below CG).
    u = {"throttle": 1.0, "gimbal": (0.0, 0.05)}
    xdot = dyn.derivatives(0.0, x, u)
    assert xdot[IW][2] < 0.0
    # Positive delta_y must give M_y < 0.
    u = {"throttle": 1.0, "gimbal": (0.05, 0.0)}
    xdot = dyn.derivatives(0.0, x, u)
    assert xdot[IW][1] < 0.0


def test_staging():
    cfg = two_stage_launcher()
    veh = Vehicle(cfg)
    m0, cg0, J0 = veh.mass_properties()
    veh.prop_remaining = 0.0
    m_before = veh.mass_properties()[0]
    s1_dry = cfg.stages[0].dry_mass
    veh.stage()
    m_after, cg1, J1 = veh.mass_properties()
    assert m_before - m_after == pytest.approx(s1_dry)
    assert veh.prop_remaining == cfg.stages[1].prop_mass
    assert np.all(np.linalg.eigvals(J1) > 0)
    assert J1[1, 1] < J0[1, 1]


def test_energy_conservation_ballistic():
    dyn, veh = make_dyn(vacuum=True)
    x = upright_state(veh, z=1000.0)
    x[IV] = np.array([50.0, 30.0, 100.0])
    u = {"throttle": 0.0, "gimbal": (0.0, 0.0)}

    def energy(xx):
        return 0.5 * np.dot(xx[IV], xx[IV]) - MU_EARTH / (R_EARTH + xx[2])

    e0 = energy(x)
    dt = 0.01
    t = 0.0
    for _ in range(int(30.0 / dt)):
        x = integrate_rk4(dyn.derivatives, t, x, u, dt)
        t += dt
    assert abs(energy(x) - e0) / abs(e0) < 1e-6


def test_touchdown_metrics():
    dyn, veh = make_dyn(vacuum=True)
    x = upright_state(veh, z=-1.0)
    touched, metrics = touchdown_check(x)
    assert touched
    assert metrics["tilt_angle_deg"] == pytest.approx(0.0)
    assert metrics["vertical_speed"] == pytest.approx(0.0)
