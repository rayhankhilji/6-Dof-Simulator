import numpy as np
import pytest

from sixdof.actuators import ActuatorSuite, GimbalActuator, ThrottleActuator
from sixdof.control_types import ControlCommand


def test_gimbal_limits_and_settling():
    g = GimbalActuator(gimbal_max=np.radians(8), gimbal_rate_max=np.radians(20))
    dt = 0.005
    cmd = np.array([np.radians(30.0), np.radians(-30.0)])  # beyond limits
    max_rate = 0.0
    prev = g.delta.copy()
    t = 0.0
    for _ in range(int(2.0 / dt)):
        g.step(cmd, dt)
        rate = np.abs(g.delta - prev) / dt
        max_rate = max(max_rate, rate.max())
        prev = g.delta.copy()
        assert np.abs(g.delta).max() <= g.gimbal_max + 1e-9
        t += dt
    assert max_rate <= np.radians(20.0) + 1e-6
    # Settled to the clamped command within 2%.
    np.testing.assert_allclose(
        g.delta, [np.radians(8.0), -np.radians(8.0)], rtol=0.02, atol=1e-4
    )


def test_throttle_delay_and_lag():
    tau, delay = 0.1, 0.05
    th = ThrottleActuator(tau=tau, min_frac=0.4, delay_s=delay)
    dt = 0.001
    t = 0.0
    vals = []
    for _ in range(int(1.0 / dt)):
        vals.append((t, th.step(1.0, t, dt)))
        t += dt
    vals = np.array(vals)
    # Nothing before the delay elapses.
    assert vals[vals[:, 0] < delay - dt, 1].max() == 0.0
    # ~63% of the way at delay + tau (relative to command 1.0 -> eff floor ok).
    i = np.argmin(np.abs(vals[:, 0] - (delay + tau)))
    assert vals[i, 1] == pytest.approx(0.63, abs=0.08)
    # Eventually reaches ~1.0.
    assert vals[-1, 1] == pytest.approx(1.0, abs=0.01)


def test_throttle_floor_and_failure():
    th = ThrottleActuator(tau=0.05, min_frac=0.4, delay_s=0.0, fail_at_t=0.5,
                          thrust_scale=1.0)
    dt = 0.005
    # Command below floor clamps up to floor.
    for i in range(100):
        out = th.step(0.1, i * dt, dt)
    assert out == pytest.approx(0.4, abs=0.02)
    # After fail_at_t the output is zero.
    th2 = ThrottleActuator(tau=0.05, min_frac=0.4, delay_s=0.0, fail_at_t=0.5)
    t = 0.0
    out = 0.0
    while t < 1.0:
        out = th2.step(1.0, t, dt)
        if t > 0.55:
            assert out == 0.0
        t += dt


def test_actuator_suite():
    suite = ActuatorSuite(
        gimbal=GimbalActuator(np.radians(8), np.radians(20)),
        throttle=ThrottleActuator(0.1, 0.4, delay_s=0.0),
    )
    u = suite.step(ControlCommand(throttle=1.0, gimbal_y=0.01, gimbal_z=-0.01), 0.0, 0.005)
    assert set(u.keys()) == {"throttle", "gimbal", "rcs_torque_B"}
    assert len(u["gimbal"]) == 2
