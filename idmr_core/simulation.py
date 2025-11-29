from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np

from .data import TextData


@dataclass
class DGPConfig:
    name: Literal["A", "B", "C"]
    n: int
    d: int
    p: int
    M_range: Tuple[int, int] = (20, 30)
    seed: int = 1234


def simulate_dgp(cfg: DGPConfig) -> tuple[TextData, np.ndarray]:
    """
    Minimal placeholder simulation for DGP-A/B/C.
    Currently implements a simple Gaussian design (DGP-A style); extend as needed.
    """
    rng = np.random.default_rng(cfg.seed)
    V = rng.normal(size=(cfg.n, cfg.p))
    theta_true = rng.normal(size=(cfg.p, cfg.d))

    # Compute probabilities and sample counts for a multinomial draw per observation.
    eta = V @ theta_true
    probs = np.exp(eta - eta.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)

    M = rng.integers(cfg.M_range[0], cfg.M_range[1] + 1, size=cfg.n)
    C = np.vstack([rng.multinomial(M[i], probs[i]) for i in range(cfg.n)])

    data = TextData.from_arrays(C=C, V=V, M=M)
    return data, theta_true
