"""ML component: landing-success prediction from gate-crossing states.

Trains a probabilistic classifier ``P(success | state at altitude gate)``
on Monte Carlo data produced by ``sixdof.batch.BatchMonteCarlo`` (see
``results/monte_carlo*.npz`` and ``scripts/train_success_model.py``).
"""

from .features import (
    OBSERVABLE_FEATURES,
    ORACLE_EXTRA,
    build_features,
    compute_feature_columns,
)
from .train import train_success_model

__all__ = [
    "OBSERVABLE_FEATURES",
    "ORACLE_EXTRA",
    "build_features",
    "compute_feature_columns",
    "train_success_model",
]
