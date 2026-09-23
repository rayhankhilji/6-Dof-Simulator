"""Wind model: steady power-law shear plus first-order Gauss-Markov gusts.

Steady profile (power law up to 300 m, constant above):

    |V(z)| = |V_10| (z / 10)^alpha,  z <= 300 m
    |V(z)| = |V_10| (300 / 10)^alpha,  z > 300 m

with direction fixed to the 10 m reference direction.

Gusts are a Dryden-like first-order Gauss-Markov process per horizontal
axis, discretely updated via ``step``:

    g_{k+1} = g_k exp(-V dt / L) + sigma sqrt(1 - exp(-2 V dt / L)) xi

where ``V`` is airspeed and ``L`` the turbulence length scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_H_REF = 10.0
_H_SHEAR_TOP = 300.0


@dataclass
class WindModel:
    """Steady shear + Gauss-Markov gust wind model.

    Parameters
    ----------
    steady_wind_I : (3,) array
        Wind velocity at the 10 m reference altitude [m/s].
    shear_exponent : float
        Power-law shear exponent (default 0.14).
    sigma_gust : float
        Gust standard deviation [m/s]; 0 disables turbulence.
    L_gust : float
        Turbulence length scale [m].
    """

    steady_wind_I: np.ndarray = field(default_factory=lambda: np.zeros(3))
    shear_exponent: float = 0.14
    sigma_gust: float = 0.0
    L_gust: float = 175.0

    def __post_init__(self) -> None:
        self.steady_wind_I = np.asarray(self.steady_wind_I, dtype=float)
        self._gust = np.zeros(3)  # current gust state, horizontal components used

    def step(self, dt: float, airspeed: float, rng: np.random.Generator) -> None:
        """Advance the gust state by ``dt`` seconds at ``airspeed`` [m/s]."""
        if self.sigma_gust <= 0.0 or dt <= 0.0:
            return
        tau_x = max(self.L_gust / max(airspeed, 1.0), 1e-3)
        phi = np.exp(-dt / tau_x)
        q = self.sigma_gust * np.sqrt(1.0 - phi**2)
        self._gust = phi * self._gust + q * rng.standard_normal(3)
        self._gust[2] = 0.0  # horizontal gusts only

    def wind_at(self, r_I: np.ndarray) -> np.ndarray:
        """Wind velocity in the inertial frame at position ``r_I`` [m/s]."""
        z = max(float(np.asarray(r_I)[2]), 0.0)
        h = min(z, _H_SHEAR_TOP)
        factor = (h / _H_REF) ** self.shear_exponent
        v_ref = np.linalg.norm(self.steady_wind_I[:2])
        if v_ref > 0.0:
            direction = self.steady_wind_I / v_ref
            steady = direction * v_ref * factor
        else:
            steady = np.zeros(3)
        return steady + self._gust
