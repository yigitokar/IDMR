"""
Lightweight core API for IDMR experiments.

Expose high-level estimator classes and data helpers so experiment scripts
don't need to import the legacy CLI modules directly.
"""

from .data import TextData
from .models import (
    BootstrapConfig,
    BootstrapResult,
    IDCConfig,
    IDCResult,
    IDCStats,
    IDCEstimator,
    SGDConfig,
    SGDResult,
    SGDStats,
    SGDEstimator,
)
from .simulation import DGPConfig, simulate_dgp

__all__ = [
    # Data
    "TextData",
    # IDC Estimation
    "IDCConfig",
    "IDCResult",
    "IDCStats",
    "IDCEstimator",
    # SGD Estimation (for comparison)
    "SGDConfig",
    "SGDResult",
    "SGDStats",
    "SGDEstimator",
    # Bootstrap inference
    "BootstrapConfig",
    "BootstrapResult",
    # Simulation
    "DGPConfig",
    "simulate_dgp",
]
