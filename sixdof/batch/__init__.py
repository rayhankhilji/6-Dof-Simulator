"""Batch Monte Carlo landing analysis (Phase 4).

``BatchMonteCarlo`` simulates N dispersed 6-DOF landings simultaneously;
``analysis`` summarizes results and produces the standard figure set.
"""

from .batch_mc import (
    BatchMonteCarlo,
    DispersionConfig,
    MCResult,
    FAILURE_MODES,
    GATE_ALTITUDES,
)
from .analysis import summarize, make_figures, wilson_ci

__all__ = [
    "BatchMonteCarlo",
    "DispersionConfig",
    "MCResult",
    "FAILURE_MODES",
    "GATE_ALTITUDES",
    "summarize",
    "make_figures",
    "wilson_ci",
]
