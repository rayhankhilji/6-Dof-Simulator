"""Standard scenarios: powered-descent landing and two-stage ascent.

Landing nominal: start 3 km up, ~184 m/s, body +x tilted 10 deg from vertical
toward the direction opposite the horizontal velocity (engine roughly
pointing into the descent for a retro burn), 10 t propellant.

All dispersion kwargs default to nominal and are consumed by Monte Carlo
runs; unknown keys are ignored via ``**dispersions``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from .environment.atmosphere import USStandardAtmosphere1976
from .environment.wind import WindModel
from .math.quaternion import quat_from_axis_angle, quat_from_two_vectors, quat_multiply
from .state import initial_state
from .vehicle import Vehicle, small_landing_vehicle, two_stage_launcher


def landing_scenario(rng: np.random.Generator | None = None, **dispersions):
    """Build the powered-descent landing scenario.

    Returns ``(vehicle, atmosphere, wind, x0, meta)``.
    """
    rng = rng or np.random.default_rng(0)
    cfg = small_landing_vehicle()
    veh = Vehicle(cfg)
    veh.prop_remaining = 10_000.0

    # --- environment ---
    atm = USStandardAtmosphere1976(density_scale=1.0 + dispersions.get("density_scale", 0.0))
    ws = dispersions.get("wind_speed", 0.0)
    wdir = dispersions.get("wind_dir", rng.uniform(0, 2 * np.pi))
    steady = np.array([ws * np.cos(wdir), ws * np.sin(wdir), 0.0])
    wind = WindModel(steady_wind_I=steady, sigma_gust=dispersions.get("gust_sigma", 0.0))

    # --- initial state ---
    r0 = np.array([-400.0, 200.0, 3000.0]) + np.asarray(dispersions.get("pos_offset", np.zeros(3)))
    v0 = np.array([30.0, -15.0, -180.0]) + np.asarray(dispersions.get("vel_offset", np.zeros(3)))

    # Attitude: +x_B tilted 10 deg from +z toward -horizontal-velocity.
    v_h = v0.copy(); v_h[2] = 0.0
    opp = -v_h / np.linalg.norm(v_h)  # horizontal direction opposing motion
    # Target x_B direction: tilt from +z toward `opp` by 10 deg.
    x_b_dir = (np.array([0, 0, 1.0]) * np.cos(np.radians(10.0))
               + opp * np.sin(np.radians(10.0)))
    q0 = quat_from_two_vectors(np.array([1.0, 0.0, 0.0]), x_b_dir)

    m0 = veh.config.total_mass(0, veh.prop_remaining) + dispersions.get("mass_offset", 0.0)
    x0 = initial_state(r0, v0, q0, np.zeros(3), m0)

    meta = SimpleNamespace(
        engine_fail_t=dispersions.get("engine_fail_t"),
        thrust_scale=dispersions.get("thrust_scale", 1.0),
        gps_outage_windows=dispersions.get("gps_outage_windows", []),
        sensor_noise_scale=dispersions.get("sensor_noise_scale", 1.0),
        meas_delay=dispersions.get("meas_delay"),
        control_noise_sigma=dispersions.get("control_noise_sigma", 0.0),
        scenario="landing",
    )
    return veh, atm, wind, x0, meta


def ascent_scenario():
    """Two-stage launcher on the pad, vertical, at rest."""
    cfg = two_stage_launcher()
    veh = Vehicle(cfg)
    atm = USStandardAtmosphere1976()
    wind = WindModel()
    q_up = quat_from_axis_angle([0.0, 1.0, 0.0], -np.pi / 2)  # +x_B up
    m0 = veh.mass_properties()[0]
    x0 = initial_state([0.0, 0.0, 0.1], [0.0, 0.0, 0.0], q_up, np.zeros(3), m0)
    meta = SimpleNamespace(scenario="ascent")
    return veh, atm, wind, x0, meta


def landing_success(metrics: dict, prop_remaining: float) -> tuple[bool, str]:
    """Evaluate touchdown metrics against landing success criteria.

    Criteria: |v_z| <= 3 m/s, lateral speed <= 1.5 m/s, tilt <= 5 deg,
    lateral offset <= 10 m, propellant remaining >= 0. ``failure_mode`` is
    the first violated criterion, or ``"success"``.
    """
    if metrics is None:
        return False, "crash_before_pad"
    checks = [
        ("hard_landing", metrics["vertical_speed"] <= 3.0),
        ("lateral_velocity", metrics["lateral_speed"] <= 1.5),
        ("tipover", metrics["tilt_angle_deg"] <= 5.0),
        ("miss_pad", metrics["lateral_offset"] <= 10.0),
        ("fuel_exhausted", prop_remaining > 0.0),
    ]
    for mode, ok in checks:
        if not ok:
            return False, mode
    return True, "success"
