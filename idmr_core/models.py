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
class BootstrapConfig:
    """Configuration for parametric bootstrap inference.

    Args:
        B: Number of bootstrap replicates
        seed: Random seed for bootstrap sampling
        return_samples: Whether to return all bootstrap theta samples
        verbose: Print progress during bootstrap
    """
    B: int = 100
    seed: int = 42
    return_samples: bool = False
    verbose: bool = True


@dataclass
class BootstrapResult:
    """Results from parametric bootstrap inference.

    Attributes:
        theta_hat: Point estimate from original data, shape (p, d)
        theta_hat_normalized: Normalized point estimate, shape (p, d)
        variance: Element-wise variance of normalized theta across bootstrap samples, shape (p, d)
        std_error: Element-wise standard error (sqrt of variance), shape (p, d)
        bias: Element-wise bias estimate (mean of bootstrap - point estimate), shape (p, d)
        mean_variance: Average variance across all parameters (scalar)
        mse: Mean squared error if theta_true provided, else None
        bootstrap_samples: List of B normalized theta matrices if return_samples=True
        n_successful: Number of successful bootstrap fits (some may fail)
        time_total: Total time for bootstrap procedure
    """
    theta_hat: np.ndarray
    theta_hat_normalized: np.ndarray
    variance: np.ndarray
    std_error: np.ndarray
    bias: np.ndarray
    mean_variance: float
    mse: Optional[float]
    bootstrap_samples: Optional[List[np.ndarray]]
    n_successful: int
    time_total: float


@dataclass
class IDCConfig:
    init: Literal["pairwise", "taddy", "poisson"] = "pairwise"
    initial_mu: Optional[Literal["zero", "logm"]] = None  # TODO: wire explicitly; legacy engine infers from init
    S: int = 10
    tol: float = 0.0
    parallel_by_choice: bool = True
    n_workers: Optional[int] = None  # TODO: not yet forwarded to MDR_v11 executor
    store_path: bool = False
    # L1 regularization
    penalty: Literal["none", "l1"] = "none"
    lambda_: float = 0.0  # L1 regularization strength (only used when penalty="l1")
    # Placeholders for future extensions
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

        # Determine L1 regularization strength
        lambda_val = cfg.lambda_ if cfg.penalty == "l1" else 0.0

        engine = MDR_v11(textData_obj=td, lambda_=lambda_val)
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

    def bootstrap(
        self,
        C: np.ndarray,
        V: np.ndarray,
        M: Optional[np.ndarray] = None,
        bootstrap_config: Optional[BootstrapConfig] = None,
        theta_true: Optional[np.ndarray] = None,
    ) -> BootstrapResult:
        """
        Perform parametric bootstrap inference for the IDC estimator.

        The parametric bootstrap procedure:
        1. Fit the model on original data to get theta_hat
        2. For each bootstrap replicate b = 1, ..., B:
           - Generate new counts C* from Multinomial(M, softmax(V @ theta_hat))
           - Fit the model on (C*, V, M) to get theta_hat_b
        3. Compute variance and other statistics from the B bootstrap samples

        Args:
            C: Count matrix, shape (n, d)
            V: Covariate matrix, shape (n, p)
            M: Total counts per observation, shape (n,). If None, computed as C.sum(axis=1)
            bootstrap_config: Bootstrap configuration (B, seed, etc.)
            theta_true: True theta for MSE computation (optional)

        Returns:
            BootstrapResult with variance estimates and optionally bootstrap samples
        """
        self._ensure_engine()
        bs_cfg = bootstrap_config or BootstrapConfig()

        t0 = perf_counter()

        # Ensure consistent dtypes
        C = np.asarray(C, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        M = np.asarray(M if M is not None else C.sum(axis=1), dtype=np.float64)

        n, d = C.shape
        p = V.shape[1]

        # Step 1: Fit on original data
        original_result = self.fit(C, V, M)
        theta_hat = original_result.theta
        theta_hat_normalized = original_result.theta_normalized

        # Step 2: Bootstrap replicates
        rng = np.random.default_rng(bs_cfg.seed)
        bootstrap_thetas: List[np.ndarray] = []

        for b in range(bs_cfg.B):
            if bs_cfg.verbose and (b + 1) % 10 == 0:
                print(f"Bootstrap replicate {b + 1}/{bs_cfg.B}")

            try:
                # Generate bootstrap data: C* ~ Multinomial(M, softmax(V @ theta_hat))
                C_star = self._generate_bootstrap_data(V, M, theta_hat, rng)

                # Fit on bootstrap data
                bs_estimator = IDCEstimator(config=self.config)
                bs_result = bs_estimator.fit(C_star, V, M)
                bootstrap_thetas.append(bs_result.theta_normalized)

            except Exception as e:
                if bs_cfg.verbose:
                    print(f"Bootstrap replicate {b + 1} failed: {e}")
                continue

        n_successful = len(bootstrap_thetas)
        if n_successful == 0:
            raise RuntimeError("All bootstrap replicates failed")

        # Step 3: Compute statistics
        theta_stack = np.stack(bootstrap_thetas, axis=0)  # shape (B_successful, p, d)
        theta_mean = theta_stack.mean(axis=0)

        variance = theta_stack.var(axis=0, ddof=1)  # unbiased variance
        std_error = np.sqrt(variance)
        bias = theta_mean - theta_hat_normalized
        mean_variance = variance.mean()

        # MSE if theta_true provided
        mse = None
        if theta_true is not None:
            theta_true_normalized = normalize(theta_true)
            sq_diff = (theta_hat_normalized - theta_true_normalized) ** 2
            mse = sq_diff.mean()

        time_total = perf_counter() - t0

        return BootstrapResult(
            theta_hat=theta_hat,
            theta_hat_normalized=theta_hat_normalized,
            variance=variance,
            std_error=std_error,
            bias=bias,
            mean_variance=mean_variance,
            mse=mse,
            bootstrap_samples=bootstrap_thetas if bs_cfg.return_samples else None,
            n_successful=n_successful,
            time_total=time_total,
        )

    def _generate_bootstrap_data(
        self,
        V: np.ndarray,
        M: np.ndarray,
        theta: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate bootstrap counts from the MNL model.

        Args:
            V: Covariate matrix, shape (n, p)
            M: Total counts, shape (n,)
            theta: Parameter matrix, shape (p, d)
            rng: Random number generator

        Returns:
            C_star: Bootstrap count matrix, shape (n, d)
        """
        n = V.shape[0]

        # Compute multinomial probabilities
        eta = V @ theta
        eta_stable = eta - eta.max(axis=1, keepdims=True)
        probs = np.exp(eta_stable)
        probs = probs / probs.sum(axis=1, keepdims=True)

        # Sample counts
        M_int = M.astype(int)
        C_star = np.vstack([rng.multinomial(M_int[i], probs[i]) for i in range(n)])

        return C_star.astype(np.float64)


# -----------------------------------------------------------------------------
# SGD Estimator (for referee comparison, Table II)
# -----------------------------------------------------------------------------

@dataclass
class SGDConfig:
    """Configuration for SGD-based multinomial regression.

    This estimator is used for comparison with IDC (Table II in the revision).
    The referee wants to see that SGD is sensitive to tuning parameters (κ).

    Args:
        optimizer: Which optimizer to use ("sgd", "adam", "adagrad")
        lr: Learning rate (κ in the to-do list)
        batch_size: Mini-batch size (use n for full-batch GD)
        epochs: Number of passes over the data
        momentum: Momentum for SGD (ignored for adam/adagrad)
        weight_decay: L2 regularization strength
        device: "cpu" or "cuda"
        verbose: Print progress during training
        store_path: Whether to store theta at each epoch
        seed: Random seed for reproducibility
    """
    optimizer: Literal["sgd", "adam", "adagrad"] = "adam"
    lr: float = 0.01
    batch_size: int = 256
    epochs: int = 100
    momentum: float = 0.0
    weight_decay: float = 0.0
    device: Literal["cpu", "cuda"] = "cpu"
    verbose: bool = False
    store_path: bool = False
    seed: int = 42


@dataclass
class SGDStats:
    """Statistics from SGD training."""
    epochs_run: int
    time_total: float
    time_per_epoch: float
    final_loss: float
    optimizer: str
    lr: float
    batch_size: int


@dataclass
class SGDResult:
    """Result from SGD estimator."""
    theta: np.ndarray
    theta_normalized: np.ndarray
    stats: SGDStats
    loss_history: Optional[List[float]] = None
    theta_path: Optional[List[np.ndarray]] = None


class SGDEstimator:
    """
    SGD-based multinomial logistic regression estimator.

    Minimizes the multinomial negative log-likelihood:
        L(θ) = -∑_i ∑_k C_ik * log(π_ik(θ))

    where π_ik(θ) = exp(V_i @ θ_k) / ∑_l exp(V_i @ θ_l)

    This is used for the referee's Table II comparison to show that:
    1. SGD is sensitive to learning rate (κ)
    2. IDC has no such tuning parameter (except S)
    """

    def __init__(self, config: Optional[SGDConfig] = None):
        self.config = config or SGDConfig()
        self.result: Optional[SGDResult] = None

    def fit(self, C: np.ndarray, V: np.ndarray, M: Optional[np.ndarray] = None) -> SGDResult:
        """
        Fit the multinomial model using SGD.

        Args:
            C: Count matrix, shape (n, d)
            V: Covariate matrix, shape (n, p)
            M: Total counts per observation (not used directly, but kept for API consistency)

        Returns:
            SGDResult with fitted theta and training statistics
        """
        import torch
        import torch.nn.functional as F

        cfg = self.config
        t0 = perf_counter()

        # Convert to tensors
        C_t = torch.tensor(C, dtype=torch.float32)
        V_t = torch.tensor(V, dtype=torch.float32)

        n, d = C.shape
        p = V.shape[1]

        # Move to device
        device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
        C_t = C_t.to(device)
        V_t = V_t.to(device)

        # Initialize theta (p x d) with small random values
        torch.manual_seed(cfg.seed)
        theta = torch.randn(p, d, device=device, dtype=torch.float32) * 0.01
        theta = torch.nn.Parameter(theta)

        # Set up optimizer
        if cfg.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                [theta],
                lr=cfg.lr,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
            )
        elif cfg.optimizer == "adam":
            optimizer = torch.optim.Adam(
                [theta],
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )
        elif cfg.optimizer == "adagrad":
            optimizer = torch.optim.Adagrad(
                [theta],
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

        # Training loop
        loss_history: List[float] = []
        theta_path: List[np.ndarray] = []
        batch_size = min(cfg.batch_size, n)

        if cfg.store_path:
            theta_path.append(theta.detach().cpu().numpy().copy())

        for epoch in range(cfg.epochs):
            # Shuffle indices for mini-batching
            perm = torch.randperm(n, device=device)

            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                V_batch = V_t[idx]  # (batch, p)
                C_batch = C_t[idx]  # (batch, d)

                optimizer.zero_grad()

                # Forward: compute log-probabilities
                # eta = V @ theta, shape (batch, d)
                eta = V_batch @ theta

                # Log-softmax for numerical stability
                log_probs = F.log_softmax(eta, dim=1)

                # Multinomial NLL: -sum(C * log_probs)
                # We use the weighted version where C_ik is the count
                loss = -(C_batch * log_probs).sum() / C_batch.sum()

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            loss_history.append(avg_loss)

            if cfg.store_path:
                theta_path.append(theta.detach().cpu().numpy().copy())

            if cfg.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{cfg.epochs}: loss = {avg_loss:.6f}")

        # Extract final theta
        theta_final = theta.detach().cpu().numpy()

        # Normalize theta (same as IDC does)
        if normalize is not None:
            theta_normalized = normalize(theta_final)
        else:
            # Fallback: subtract mean across choices
            theta_normalized = theta_final - theta_final.mean(axis=1, keepdims=True)

        t_total = perf_counter() - t0

        stats = SGDStats(
            epochs_run=cfg.epochs,
            time_total=t_total,
            time_per_epoch=t_total / cfg.epochs,
            final_loss=loss_history[-1] if loss_history else float("nan"),
            optimizer=cfg.optimizer,
            lr=cfg.lr,
            batch_size=batch_size,
        )

        result = SGDResult(
            theta=theta_final,
            theta_normalized=theta_normalized,
            stats=stats,
            loss_history=loss_history,
            theta_path=theta_path if cfg.store_path else None,
        )
        self.result = result
        return result
