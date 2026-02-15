from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np

from .data import TextData


@dataclass
class DGPConfig:
    """Configuration for data generating processes.

    Args:
        name: DGP variant ("A" = clean MNL baseline, "C" = mixture stress-test)
        n: Number of observations
        d: Number of choices/alternatives
        p: Number of covariates (fixed to 5 in paper simulations)
        M_range: Range for total counts (uniform draw for DGP-A)
        seed: Random seed for reproducibility
        theta_seed: Separate seed for generating true theta (if None, uses seed)
    """
    name: Literal["A", "C"]
    n: int
    d: int
    p: int = 5
    M_range: Tuple[int, int] = (20, 30)
    seed: int = 1234
    theta_seed: int | None = None


def _sample_mixture_of_normals(
    rng: np.random.Generator,
    mean1: float,
    std1: float,
    mean2: float,
    std2: float,
    size: int | Tuple[int, ...],
) -> np.ndarray:
    """Sample from a 50/50 mixture of two Gaussians."""
    if isinstance(size, int):
        size = (size,)
    n_samples = int(np.prod(size))

    # Choose component for each sample
    component = rng.choice([0, 1], size=n_samples)

    # Sample from appropriate component
    samples = np.where(
        component == 0,
        rng.normal(mean1, std1, size=n_samples),
        rng.normal(mean2, std2, size=n_samples),
    )
    return samples.reshape(size)


def simulate_dgp(cfg: DGPConfig) -> tuple[TextData, np.ndarray]:
    """
    Simulate data from the specified DGP.

    DGP-A (clean baseline MNL):
        - V_i ~ N(0, I_p) for p covariates
        - M_i ~ Uniform[20, 30] discrete
        - theta* has each element ~ N(0, 1)
        - C_i | (V_i, M_i) ~ Multinomial(M_i, softmax(V_i @ theta*))

    DGP-C (mixture stress-test):
        - V_i from two-component Gaussian mixture: means 0 and 4, std 1
        - M_i from mixture of Gaussians: means 10 and 60, std 1 and 5, rounded to int
        - Same MNL sampling for C_i
        - Creates highly unbalanced choice probabilities

    Returns:
        Tuple of (TextData, theta_true) where theta_true has shape (p, d)
    """
    rng = np.random.default_rng(cfg.seed)
    theta_seed = cfg.theta_seed if cfg.theta_seed is not None else cfg.seed
    theta_rng = np.random.default_rng(theta_seed)

    # Generate true theta (same for both DGPs)
    theta_true = theta_rng.normal(size=(cfg.p, cfg.d))

    if cfg.name == "A":
        # DGP-A: Clean baseline MNL
        # Covariates: V_i ~ N(0, I_p)
        V = rng.normal(size=(cfg.n, cfg.p))

        # Total counts: M_i ~ Uniform[M_range]
        M = rng.integers(cfg.M_range[0], cfg.M_range[1] + 1, size=cfg.n)

    elif cfg.name == "C":
        # DGP-C: Mixture stress-test
        # Covariates: V_i from mixture of N(0,1) and N(4,1)
        V = _sample_mixture_of_normals(rng, mean1=0, std1=1, mean2=4, std2=1, size=(cfg.n, cfg.p))

        # Total counts: use M_range if explicitly set, otherwise default mixture
        if cfg.M_range != (20, 30):
            # Explicit M_range provided — use uniform draw (same as DGP-A)
            M = rng.integers(cfg.M_range[0], cfg.M_range[1] + 1, size=cfg.n)
        else:
            # Default: M_i from mixture of N(10,1) and N(60,5), rounded to int
            M_raw = _sample_mixture_of_normals(rng, mean1=10, std1=1, mean2=60, std2=5, size=cfg.n)
            M = np.maximum(1, np.round(np.abs(M_raw)).astype(int))  # Ensure M >= 1

    else:
        raise NotImplementedError(f"DGP-{cfg.name} not implemented.")

    # Compute multinomial probabilities: P_ik = exp(V_i @ theta_k) / sum_l exp(V_i @ theta_l)
    eta = V @ theta_true
    # Numerical stability: subtract max per row
    eta_stable = eta - eta.max(axis=1, keepdims=True)
    probs = np.exp(eta_stable)
    probs = probs / probs.sum(axis=1, keepdims=True)

    # Sample counts: C_i | (V_i, M_i) ~ Multinomial(M_i, P_i)
    C = np.vstack([rng.multinomial(M[i], probs[i]) for i in range(cfg.n)])

    data = TextData.from_arrays(C=C, V=V, M=M)
    return data, theta_true
