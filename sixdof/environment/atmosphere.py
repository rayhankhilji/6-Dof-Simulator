"""US Standard Atmosphere 1976, geopotential-altitude model 0-86 km.

Seven layers with piecewise-linear temperature:

    T = T_b + L_b (h - h_b)
    p = p_b (T/T_b)^(-g0/(R L_b))        for L_b != 0
    p = p_b exp(-g0 (h - h_b)/(R T_b))   for L_b == 0
    rho = p / (R T),  a = sqrt(gamma R T)

Above 86 km density decays exponentially with a 7 km scale height from the
86 km value; pressure follows consistently (rho R T_86). Below sea level the
altitude is clamped to h = 0.
"""

from __future__ import annotations

import numpy as np

G0 = 9.80665
R_AIR = 287.05287
GAMMA = 1.4
T0 = 288.15
P0 = 101325.0

# Layer base geopotential altitudes [m] and lapse rates [K/m].
_H_B = np.array([0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0])
_L_B = np.array([-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002])

_H_TOP = 86000.0
_SCALE_HEIGHT = 7000.0


class USStandardAtmosphere1976:
    """1976 US Standard Atmosphere, 0-86 km plus exponential tail.

    Parameters
    ----------
    density_scale : float
        Multiplier on density (and pressure, consistently) for Monte Carlo
        dispersions. Default 1.0.
    """

    def __init__(self, density_scale: float = 1.0) -> None:
        self.density_scale = float(density_scale)
        # Precompute base temperature/pressure at each layer boundary.
        self._T_b = np.empty(7)
        self._p_b = np.empty(7)
        T, p = T0, P0
        for i in range(7):
            self._T_b[i] = T
            self._p_b[i] = p
            if i < 6:
                h_next = _H_B[i + 1]
                T_next = T + _L_B[i] * (h_next - _H_B[i])
                if _L_B[i] == 0.0:
                    p = p * np.exp(-G0 * (h_next - _H_B[i]) / (R_AIR * T))
                else:
                    p = p * (T_next / T) ** (-G0 / (R_AIR * _L_B[i]))
                T = T_next
        self._T_86 = self._T_b[6] + _L_B[6] * (_H_TOP - _H_B[6])
        self._p_86 = self._p_b[6] * (self._T_86 / self._T_b[6]) ** (
            -G0 / (R_AIR * _L_B[6])
        )
        self._rho_86 = self._p_86 / (R_AIR * self._T_86)

    def properties(self, h):
        """Return ``(T, p, rho, a)`` at geopotential altitude ``h`` [m].

        ``h`` may be scalar or a numpy array; results match its shape.
        """
        h = np.clip(np.asarray(h, dtype=float), 0.0, None)
        # Layer index 0..6 for h < 86 km; mark 7 for the exponential tail.
        layer = np.clip(np.searchsorted(_H_B, h, side="right") - 1, 0, 7)
        h_cl = np.clip(h, None, _H_TOP)

        T_b = np.take(self._T_b, np.clip(layer, 0, 6))
        p_b = np.take(self._p_b, np.clip(layer, 0, 6))
        L_b = np.take(_L_B, np.clip(layer, 0, 6))
        h_b = np.take(_H_B, np.clip(layer, 0, 6))

        T = T_b + L_b * (h_cl - h_b)
        ratio = T / T_b
        # Nonzero-lapse and isothermal branches; use np.where to stay vectorized.
        exp_term = np.exp(-G0 * (h_cl - h_b) / (R_AIR * np.maximum(T_b, 1.0)))
        p_iso = p_b * exp_term
        p_lapse = p_b * ratio ** (-G0 / (R_AIR * np.where(L_b == 0.0, 1.0, L_b)))
        p = np.where(L_b == 0.0, p_iso, p_lapse)

        # Exponential tail above 86 km.
        rho = p / (R_AIR * T)
        rho_tail = self._rho_86 * np.exp(-(h - _H_TOP) / _SCALE_HEIGHT)
        p_tail = rho_tail * R_AIR * self._T_86
        tail = h > _H_TOP
        rho = np.where(tail, rho_tail, rho)
        p = np.where(tail, p_tail, p)
        T = np.where(tail, self._T_86, T)

        rho = rho * self.density_scale
        p = p * self.density_scale
        a = np.sqrt(GAMMA * R_AIR * T)
        return T, p, rho, a

    def density(self, h):
        """Density [kg/m^3] only."""
        return self.properties(h)[2]

    def pressure(self, h):
        """Pressure [Pa] only."""
        return self.properties(h)[1]
