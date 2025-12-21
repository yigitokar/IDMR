"""
Verify that the refactored IDCEstimator produces IDENTICAL results to old MDR_v11 code.

Usage:
    UV_CACHE_DIR=.uv-cache uv run python scripts/verify_refactor.py
"""
import numpy as np

np.random.seed(42)

from idmr_core import IDCEstimator, IDCConfig, DGPConfig, simulate_dgp
from PB import MDR_v11
from classesv2 import textData, normalize


def main():
    # Generate data
    cfg = DGPConfig(name="A", n=100, d=10, p=5, M_range=(20, 30), seed=42)
    data, theta_true = simulate_dgp(cfg)
    theta_true_norm = normalize(theta_true)

    print("=" * 60)
    print("OLD CODE (MDR_v11 directly)")
    print("=" * 60)

    # OLD way: use MDR_v11 directly
    td = textData(data.C, data.V, data.M)
    engine = MDR_v11(textData_obj=td)
    engine.PARALLEL_initialize_theta_PairwiseBinomial()
    for _ in range(10):
        engine.PARALLEL_oneRun()
    old_theta_norm = normalize(engine.theta_mat)
    old_mse = ((old_theta_norm - theta_true_norm) ** 2).mean()
    print(f"MSE (old): {old_mse:.8f}")

    print("\n" + "=" * 60)
    print("NEW CODE (IDCEstimator wrapper)")
    print("=" * 60)

    # NEW way: use IDCEstimator
    est = IDCEstimator(IDCConfig(init="pairwise", S=10))
    result = est.fit(data.C, data.V, data.M)
    new_mse = ((result.theta_normalized - theta_true_norm) ** 2).mean()
    print(f"MSE (new): {new_mse:.8f}")

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    # Check if thetas are identical
    diff = np.abs(old_theta_norm - result.theta_normalized).max()
    print(f"Max difference between old and new theta: {diff:.2e}")

    if diff < 1e-10:
        print("✓ IDENTICAL - refactor is correct")
    else:
        print(f"✗ DIFFERENT - max diff = {diff}")
        print("\nNote: Small differences may occur due to parallel execution order.")


if __name__ == "__main__":
    main()
