import numpy as np
import pytest

from sixdof.guidance import OptimalGuidance, PolynomialGuidance, ZEMZEVGuidance
from sixdof.navigation.base import NavState
from sixdof.scenarios import landing_scenario


def _nav(r, v):
    return NavState(np.asarray(r, float), np.asarray(v, float),
                    np.array([1.0, 0, 0, 0]), np.zeros(3), np.zeros(3),
                    np.eye(15), t=0.0)


def _info(m=37000.0, t_max=800e3, t_min=280e3, g=9.81, prop=10_000.0):
    return {"mass": m, "thrust_max": t_max, "thrust_min": t_min, "g": g,
            "prop_remaining": prop}


def _simulate_point_mass(guidance, r0, v0, info, dt=0.02, t_max=120.0):
    """3-DOF integrator r'' = u + g driven by guidance accel_cmd."""
    r, v = np.asarray(r0, float).copy(), np.asarray(v0, float).copy()
    g_vec = np.array([0.0, 0.0, -info["g"]])
    m = info["mass"]
    fuel = 0.0
    t = 0.0
    traj = []
    while t < t_max and r[2] > 0.0:
        nav = _nav(r, v)
        gc = guidance.compute(t, nav, info)
        a = gc.accel_cmd_I if gc.engine_on else np.zeros(3)
        a_mag = np.linalg.norm(a)
        t_avail = info["thrust_max"] / m
        if a_mag > t_avail:
            a = a * t_avail / a_mag
            a_mag = t_avail
        v = v + (a + g_vec) * dt
        r = r + v * dt
        fuel += m * a_mag * dt / (311.0 * 9.80665)
        t += dt
        traj.append((t, r.copy(), v.copy()))
    return r, v, fuel, traj


def test_zemzev_lands_point_mass():
    g = ZEMZEVGuidance()
    g.ignited = True  # skip coast logic for the integrator test
    r, v, fuel, _ = _simulate_point_mass(g, [-400.0, 200.0, 3000.0],
                                         [30.0, -15.0, -180.0], _info())
    assert np.linalg.norm(r) < 0.5
    # Terminal velocity at or below the -1.5 m/s descent target (it lands
    # softly at ~0 because ZEM/ZEV re-plans to the end).
    assert np.linalg.norm(v) <= 2.0


def test_polynomial_boundary_conditions():
    g = PolynomialGuidance()
    rng = np.random.default_rng(0)
    r0 = rng.standard_normal(3) * 500 + [0, 0, 1500]
    v0 = rng.standard_normal(3) * 30 + [0, 0, -80]
    rf, vf = np.zeros(3), np.array([0, 0, -3.0])
    g_vec = np.array([0, 0, -9.81])
    T = 20.0
    C = g.solve_coeffs(r0, v0, rf, vf, T, g_vec)
    # Integrate u(tau) = c0 + c1 t + c2 t^2 analytically.
    c0, c1, c2 = C[0], C[1], C[2]
    vT = v0 + g_vec * T + c0 * T + c1 * T**2 / 2 + c2 * T**3 / 3
    rT = r0 + v0 * T + 0.5 * g_vec * T**2 + c0 * T**2 / 2 + c1 * T**3 / 6 + c2 * T**4 / 12
    aT = c0 + c1 * T + c2 * T**2
    np.testing.assert_allclose(vT, vf, atol=1e-8)
    np.testing.assert_allclose(rT, rf, atol=1e-6)
    np.testing.assert_allclose(aT, g.a_f, atol=1e-8)


def test_optimal_solve_and_fuel():
    g = OptimalGuidance()
    nav = _nav([-400.0, 200.0, 3000.0], [30.0, -15.0, -180.0])
    info = _info()
    rf, vf = np.zeros(3), np.array([0, 0, -3.0])
    res = g._solve(nav, info, rf, vf)
    assert res is not None
    assert g.solve_times[-1] < 3.0
    plan = g.plan_trajectory()
    np.testing.assert_allclose(plan["r"][:, -1], rf, atol=1.0)
    np.testing.assert_allclose(plan["v"][:, -1], vf, atol=0.5)
    # Fuel should not exceed a ZEM/ZEV plan on the same model (+1%).
    gz = ZEMZEVGuidance(); gz.ignited = True
    _, _, fuel_zem, _ = _simulate_point_mass(gz, [-400.0, 200.0, 3000.0],
                                           [30.0, -15.0, -180.0], info)
    assert g.fuel_predicted <= fuel_zem * 1.01


def test_landing_scenario_shape():
    veh, atm, wind, x0, meta = landing_scenario(np.random.default_rng(0))
    assert x0.shape == (14,)
    assert veh.prop_remaining == 10_000.0
    assert meta.scenario == "landing"
    veh2, _, _, x02, _ = landing_scenario(np.random.default_rng(0),
                                          wind_speed=8.0, gust_sigma=1.5,
                                          mass_offset=500.0,
                                          pos_offset=np.array([10, 0, -5]))
    np.testing.assert_allclose(x02[0:3], [-390, 200, 2995])
    assert x02[13] == pytest.approx(x0[13] + 500.0)
