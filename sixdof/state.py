"""State vector layout for the 6-DOF simulator.

The state is a flat array of length 14::

    x = [r_I(3), v_I(3), q(4), omega_B(3), m(1)]

with r_I / v_I the inertial ENU position/velocity, q the scalar-first
body-to-inertial quaternion, omega_B the body angular rate in body frame,
and m the total vehicle mass [kg].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IR = slice(0, 3)
IV = slice(3, 6)
IQ = slice(6, 10)
IW = slice(10, 13)
IM = 13
STATE_SIZE = 14


@dataclass
class State:
    """Convenience wrapper around the 14-element state array."""

    r_I: np.ndarray
    v_I: np.ndarray
    q: np.ndarray
    omega_B: np.ndarray
    m: float

    @classmethod
    def from_array(cls, x: np.ndarray) -> "State":
        """Wrap a flat 14-element state array."""
        x = np.asarray(x, dtype=float)
        if x.shape != (STATE_SIZE,):
            raise ValueError(f"state must have shape ({STATE_SIZE},), got {x.shape}")
        return cls(x[IR].copy(), x[IV].copy(), x[IQ].copy(), x[IW].copy(), float(x[IM]))

    def to_array(self) -> np.ndarray:
        """Flatten back to the 14-element state array."""
        x = np.zeros(STATE_SIZE)
        x[IR] = self.r_I
        x[IV] = self.v_I
        x[IQ] = self.q
        x[IW] = self.omega_B
        x[IM] = self.m
        return x

    @property
    def position(self) -> np.ndarray:
        return self.r_I

    @property
    def velocity(self) -> np.ndarray:
        return self.v_I

    @property
    def quaternion(self) -> np.ndarray:
        return self.q

    @property
    def angular_rate(self) -> np.ndarray:
        return self.omega_B

    @property
    def mass(self) -> float:
        return self.m


def initial_state(
    r: np.ndarray,
    v: np.ndarray,
    q: np.ndarray,
    w: np.ndarray,
    m: float,
) -> np.ndarray:
    """Build a flat 14-element state vector from components."""
    return State(
        np.asarray(r, dtype=float),
        np.asarray(v, dtype=float),
        np.asarray(q, dtype=float),
        np.asarray(w, dtype=float),
        float(m),
    ).to_array()
