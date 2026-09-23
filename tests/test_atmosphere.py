import numpy as np
import pytest

from sixdof.environment.atmosphere import USStandardAtmosphere1976


def test_sea_level():
    atm = USStandardAtmosphere1976()
    T, p, rho, a = atm.properties(0.0)
    assert T == pytest.approx(288.15)
    assert p == pytest.approx(101325.0)
    assert rho == pytest.approx(1.225, abs=1e-3)
    assert a == pytest.approx(340.29, abs=0.1)


@pytest.mark.parametrize(
    "h,p_ref",
    [
        (11000.0, 22632.0),
        (20000.0, 5474.9),
        (32000.0, 868.02),
        (47000.0, 110.9),
    ],
)
def test_layer_pressures(h, p_ref):
    atm = USStandardAtmosphere1976()
    T, p, rho, a = atm.properties(h)
    assert p == pytest.approx(p_ref, rel=0.01)


def test_11km_temperature():
    atm = USStandardAtmosphere1976()
    T, _, _, _ = atm.properties(11000.0)
    assert T == pytest.approx(216.65, abs=1e-6)


def test_density_monotone_decreasing():
    atm = USStandardAtmosphere1976()
    h = np.linspace(0, 86000, 500)
    _, _, rho, _ = atm.properties(h)
    assert np.all(np.diff(rho) <= 1e-12)


def test_vectorized():
    atm = USStandardAtmosphere1976()
    h = np.array([0.0, 5000.0, 11000.0, 50000.0])
    T, p, rho, a = atm.properties(h)
    assert T.shape == p.shape == rho.shape == a.shape == (4,)
    for i in range(4):
        Ti, pi, ri, ai = atm.properties(h[i])
        np.testing.assert_allclose([T[i], p[i], rho[i], a[i]], [Ti, pi, ri, ai])


def test_above_86km_decays():
    atm = USStandardAtmosphere1976()
    _, _, rho86, _ = atm.properties(86000.0)
    _, _, rho93, _ = atm.properties(93000.0)
    assert rho93 == pytest.approx(rho86 * np.exp(-7000.0 / 7000.0), rel=1e-3)
