"""
Lightweight core API for IDMR experiments.

Expose high-level estimator classes and data helpers so experiment scripts
don’t need to import the legacy CLI modules directly.
"""

from .data import TextData
from .models import IDCConfig, IDCResult, IDCStats, IDCEstimator
from .simulation import DGPConfig, simulate_dgp

__all__ = [
    "TextData",
    "IDCConfig",
    "IDCResult",
    "IDCStats",
    "IDCEstimator",
    "DGPConfig",
    "simulate_dgp",
]
