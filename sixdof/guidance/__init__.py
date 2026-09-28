"""Guidance laws: ascent gravity turn and powered descent."""

from .ascent import GravityTurnGuidance
from .base import PoweredDescentBase
from .descent import PolynomialGuidance, ZEMZEVGuidance
from .optimal import OptimalGuidance

GUIDANCE = {
    "zemzev": ZEMZEVGuidance,
    "poly": PolynomialGuidance,
    "optimal": OptimalGuidance,
    "gravity_turn": GravityTurnGuidance,
}


def make_guidance(name: str, **kw):
    """Instantiate a guidance law by name."""
    if name not in GUIDANCE:
        raise ValueError(f"unknown guidance {name!r}; options: {sorted(GUIDANCE)}")
    return GUIDANCE[name](**kw)


__all__ = [
    "GravityTurnGuidance",
    "PoweredDescentBase",
    "PolynomialGuidance",
    "ZEMZEVGuidance",
    "OptimalGuidance",
    "GUIDANCE",
    "make_guidance",
]
