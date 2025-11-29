from __future__ import annotations

from time import perf_counter
from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np

# Lazy imports to avoid hard dependencies at module import time.
try:
    from classesv2 import normalize, textData
    from PB import MDR_v11
except ImportError:  # pragma: no cover - will be raised if deps missing at runtime
    normalize = None
    textData = None
    MDR_v11 = None


@dataclass
class IDCConfig:
    init: Literal["pairwise", "taddy", "poisson"] = "pairwise"
    initial_mu: Optional[Literal["zero", "logm"]] = None  # TODO: wire explicitly; legacy engine infers from init
    S: int = 10
    tol: float = 0.0
    parallel_by_choice: bool = True
    n_workers: Optional[int] = None  # TODO: not yet forwarded to MDR_v11 executor
    store_path: bool = False
    # Placeholders for future extensions
    penalty: Literal["none", "l1"] = "none"  # TODO: not wired
    lambda_: float = 0.0  # TODO: not wired
    poisson_solver: Literal["cvxpy_scs", "cvxpy_mosek"] = "cvxpy_scs"  # TODO: not wired
    device: Literal["cpu", "cuda"] = "cpu"  # TODO: not wired


@dataclass
class IDCStats:
    S_effective: int
    time_total: float
    time_per_iter: float
    init: str
    init_time: float


@dataclass
class IDCResult:
    theta: np.ndarray
    theta_normalized: np.ndarray
    mu: np.ndarray
    stats: IDCStats
    theta_path: Optional[List[np.ndarray]] = None


class IDCEstimator:
    """
    Thin wrapper around the existing MDR_v11 engine.
    Experiments should depend on this class rather than PB.py directly.
    """

    def __init__(self, config: Optional[IDCConfig] = None):
        self.config = config or IDCConfig()
        self._engine: Optional["MDR_v11"] = None
        self.result: Optional[IDCResult] = None

    def _ensure_engine(self):
        if textData is None or MDR_v11 is None or normalize is None:
            raise ImportError(
                "Required modules not available; ensure classesv2.py and PB.py "
                "dependencies are installed (numpy, torch, cvxpy, etc.)."
            )

    def fit(self, C: np.ndarray, V: np.ndarray, M: Optional[np.ndarray] = None) -> IDCResult:
        self._ensure_engine()
        cfg = self.config

        # Ensure consistent float64 dtype for the engine (avoids legacy dtype mismatches).
        C = np.asarray(C, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        M = np.asarray(M if M is not None else C.sum(axis=1), dtype=np.float64)

        td = textData(C, V, M)  # reuse existing data wrapper to match engine expectations
        engine = MDR_v11(textData_obj=td)
        self._engine = engine

        t0 = perf_counter()
        init_start = perf_counter()

        if cfg.init == "pairwise":
            engine.PARALLEL_initialize_theta_PairwiseBinomial()
        elif cfg.init == "taddy":
            engine.initial_mu = "logm"
            engine.PARALLEL_initialize()
        elif cfg.init == "poisson":
            engine.initial_mu = "zero"
            engine.PARALLEL_initialize()
        else:
            raise ValueError(f"Unknown init method: {cfg.init}")

        init_time = perf_counter() - init_start

        theta_path: List[np.ndarray] = []
        if cfg.store_path:
            theta_path.append(engine.theta_mat.copy())

        S_eff = 0
        for s in range(1, cfg.S + 1):
            prev_theta = engine.theta_mat.copy() if cfg.tol > 0 else None

            if cfg.parallel_by_choice:
                engine.PARALLEL_oneRun()
            else:
                engine.oneRun()

            S_eff = s
            if cfg.store_path:
                theta_path.append(engine.theta_mat.copy())

            if cfg.tol > 0 and prev_theta is not None:
                delta = np.linalg.norm(engine.theta_mat - prev_theta) / np.sqrt(engine.theta_mat.size)
                if delta < cfg.tol:
                    break

        theta = engine.theta_mat.copy()
        theta_norm = normalize(theta)
        mu = engine.mu_vec.copy()

        t_total = perf_counter() - t0
        stats = IDCStats(
            S_effective=S_eff,
            time_total=t_total,
            time_per_iter=(t_total - init_time) / max(S_eff, 1),
            init=cfg.init,
            init_time=init_time,
        )

        res = IDCResult(
            theta=theta,
            theta_normalized=theta_norm,
            mu=mu,
            stats=stats,
            theta_path=theta_path if cfg.store_path else None,
        )
        self.result = res
        return res
