import numpy as np
import pytest

from sixdof.environment.atmosphere import USStandardAtmosphere1976
from sixdof.environment.wind import WindModel
from sixdof.math.quaternion import quat_from_axis_angle
from sixdof.simulation import (
    NullController,
    NullGuidance,
    PerfectNavigator,
    Simulation,
)
from sixdof.state import initial_state
from sixdof.vehicle import small_landing_vehicle, Vehicle


def test_freefall_touchdown_and_result():
    veh = Vehicle(small_landing_vehicle())
    atm = USStandardAtmosphere1976(density_scale=0.0)  # vacuum: no drag
    sim = Simulation(
        veh, atm, WindModel(),
        sensors=None,
        navigator=PerfectNavigator(),
        guidance=NullGuidance(),
        controller=NullController(),
        actuators=None,
        dt=0.005,
        control_rate_hz=50.0,
    )
    m = veh.mass_properties()[0]
    q_up = quat_from_axis_angle([0, 1, 0], -np.pi / 2)
    x0 = initial_state([0, 0, 500.0], [0, 0, 0], q_up, [0, 0, 0], m)
    res = sim.run(x0, t_end=60.0, stop_on_touchdown=True)

    assert res.touchdown is not None
    g = 9.80665  # near enough at 500 m
    expected_v = np.sqrt(2.0 * 9.8 * 500.0)
    assert res.touchdown["vertical_speed"] == pytest.approx(expected_v, rel=0.05)
    # Consistent array lengths.
    n = len(res.t)
    assert res.x.shape == (n, 14)
    assert res.x_nav.shape == (n, 10)
    assert res.P_diag.shape == (n, 15)
    assert res.u.shape == (n, 3)
    assert res.aux.shape == (n, 4)
    assert (n, "touchdown") == (len(res.t), res.events[-1][1])

    # save_npz round trip.
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "res.npz")
        res.save_npz(p)
        data = np.load(p, allow_pickle=True)
        assert data["x"].shape == (n, 14)
        assert data["t"].shape == (n,)
