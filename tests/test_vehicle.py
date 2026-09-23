import numpy as np
import pytest

from sixdof.vehicle import (
    Vehicle,
    falcon9_like_first_stage,
    small_landing_vehicle,
    two_stage_launcher,
)


def test_inertia_positive_definite():
    for cfg in (small_landing_vehicle(), two_stage_launcher()):
        veh = Vehicle(cfg)
        for frac in (1.0, 0.5, 0.0):
            veh.prop_remaining = cfg.stages[0].prop_mass * frac
            m, cg, J = veh.mass_properties()
            assert np.all(np.linalg.eigvals(J) > 0)
            assert J[1, 1] == pytest.approx(J[2, 2])  # axisymmetric
            assert m > 0
            assert 0.0 <= cg <= sum(s.length for s in cfg.stages)


def test_cg_moves_toward_base_as_prop_depletes():
    cfg = small_landing_vehicle()
    veh = Vehicle(cfg)
    cgs = []
    for frac in (1.0, 0.8, 0.6, 0.4):
        veh.prop_remaining = cfg.stages[0].prop_mass * frac
        cgs.append(veh.mass_properties()[1])
    assert np.all(np.diff(cgs) < 0)  # CG descends toward the base
    # Empty CG is the dry-structure CG, below the full-stack CG.
    veh.prop_remaining = 0.0
    assert veh.mass_properties()[1] < cgs[0]


def test_mass_bookkeeping():
    cfg = two_stage_launcher()
    veh = Vehicle(cfg)
    m_full = veh.mass_properties()[0]
    expected = (
        cfg.stages[0].dry_mass + cfg.stages[0].prop_mass
        + cfg.stages[1].dry_mass + cfg.stages[1].prop_mass
        + cfg.payload_mass
    )
    assert m_full == pytest.approx(expected)

    burned = veh.burn(300.0, 10.0)
    assert burned == pytest.approx(3000.0)
    assert veh.prop_remaining == pytest.approx(cfg.stages[0].prop_mass - 3000.0)

    veh.stage()
    assert veh.active_stage == 1
    assert veh.events[0]["dropped_dry_mass"] == cfg.stages[0].dry_mass
    assert veh.mass_properties()[0] == pytest.approx(
        cfg.stages[1].dry_mass + cfg.stages[1].prop_mass + cfg.payload_mass
    )
    with pytest.raises(RuntimeError):
        veh.stage()


def test_engine_thrust_backpressure():
    eng = falcon9_like_first_stage().engine
    t_vac, mdot = eng.thrust_and_mdot(1.0, 0.0)
    assert t_vac == pytest.approx(eng.thrust_max_vac)
    assert mdot == pytest.approx(eng.thrust_max_vac / (eng.isp_vac * 9.80665))
    t_sl, _ = eng.thrust_and_mdot(1.0, 101325.0)
    assert t_sl == pytest.approx(eng.thrust_max_vac - 101325.0 * eng.nozzle_exit_area)
    t_off, mdot_off = eng.thrust_and_mdot(0.0, 0.0)
    assert t_off == 0.0 and mdot_off == 0.0
