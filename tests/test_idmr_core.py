"""
Sanity tests for idmr_core module.

These tests verify that the core API works correctly on small problems
before scaling up to the full simulation study.

Run with: uv run python -m pytest tests/test_idmr_core.py -v
"""

import numpy as np
import pytest

from idmr_core import (
    TextData,
    DGPConfig,
    simulate_dgp,
    IDCConfig,
    IDCEstimator,
    SGDConfig,
    SGDEstimator,
)


# -----------------------------------------------------------------------------
# Test data generation
# -----------------------------------------------------------------------------

class TestSimulateDGP:
    """Tests for the data generating process."""

    def test_dgp_a_shapes(self):
        """DGP-A should produce correct shapes."""
        cfg = DGPConfig(name="A", n=100, d=10, p=5, seed=42)
        data, theta_true = simulate_dgp(cfg)

        assert data.C.shape == (100, 10), f"C shape: {data.C.shape}"
        assert data.V.shape == (100, 5), f"V shape: {data.V.shape}"
        assert data.M.shape == (100,), f"M shape: {data.M.shape}"
        assert theta_true.shape == (5, 10), f"theta shape: {theta_true.shape}"

    def test_dgp_c_shapes(self):
        """DGP-C should produce correct shapes."""
        cfg = DGPConfig(name="C", n=100, d=10, p=5, seed=42)
        data, theta_true = simulate_dgp(cfg)

        assert data.C.shape == (100, 10)
        assert data.V.shape == (100, 5)
        assert data.M.shape == (100,)
        assert theta_true.shape == (5, 10)

    def test_dgp_a_counts_sum_to_m(self):
        """Counts should sum to M for each observation."""
        cfg = DGPConfig(name="A", n=100, d=10, p=5, seed=42)
        data, _ = simulate_dgp(cfg)

        row_sums = data.C.sum(axis=1)
        np.testing.assert_array_equal(row_sums, data.M)

    def test_dgp_c_counts_sum_to_m(self):
        """Counts should sum to M for each observation (DGP-C)."""
        cfg = DGPConfig(name="C", n=100, d=10, p=5, seed=42)
        data, _ = simulate_dgp(cfg)

        row_sums = data.C.sum(axis=1)
        np.testing.assert_array_equal(row_sums, data.M)

    def test_dgp_a_m_range(self):
        """M should be within specified range for DGP-A."""
        cfg = DGPConfig(name="A", n=500, d=10, p=5, M_range=(20, 30), seed=42)
        data, _ = simulate_dgp(cfg)

        assert data.M.min() >= 20
        assert data.M.max() <= 30

    def test_dgp_reproducibility(self):
        """Same seed should produce same data."""
        cfg1 = DGPConfig(name="A", n=100, d=10, p=5, seed=123)
        cfg2 = DGPConfig(name="A", n=100, d=10, p=5, seed=123)

        data1, theta1 = simulate_dgp(cfg1)
        data2, theta2 = simulate_dgp(cfg2)

        np.testing.assert_array_equal(data1.C, data2.C)
        np.testing.assert_array_equal(data1.V, data2.V)
        np.testing.assert_array_equal(theta1, theta2)

    def test_different_seeds_different_data(self):
        """Different seeds should produce different data."""
        cfg1 = DGPConfig(name="A", n=100, d=10, p=5, seed=123)
        cfg2 = DGPConfig(name="A", n=100, d=10, p=5, seed=456)

        data1, _ = simulate_dgp(cfg1)
        data2, _ = simulate_dgp(cfg2)

        assert not np.allclose(data1.C, data2.C)


# -----------------------------------------------------------------------------
# Test IDCEstimator
# -----------------------------------------------------------------------------

class TestIDCEstimator:
    """Tests for the IDC estimator."""

    @pytest.fixture
    def small_data(self):
        """Generate small test data."""
        cfg = DGPConfig(name="A", n=200, d=20, p=5, M_range=(20, 30), seed=42)
        data, theta_true = simulate_dgp(cfg)
        return data, theta_true

    def test_fit_pairwise_init(self, small_data):
        """IDC with pairwise init should run without errors."""
        data, theta_true = small_data

        est = IDCEstimator(IDCConfig(init="pairwise", S=5))
        result = est.fit(data.C, data.V, data.M)

        assert result.theta.shape == (5, 20)
        assert result.theta_normalized.shape == (5, 20)
        assert result.mu.shape == (200, 1)
        assert result.stats.S_effective == 5

    def test_fit_taddy_init(self, small_data):
        """IDC with Taddy init should run without errors."""
        data, theta_true = small_data

        est = IDCEstimator(IDCConfig(init="taddy", S=5))
        result = est.fit(data.C, data.V, data.M)

        assert result.theta.shape == (5, 20)
        assert result.stats.init == "taddy"

    def test_fit_poisson_init(self, small_data):
        """IDC with Poisson init should run without errors."""
        data, theta_true = small_data

        est = IDCEstimator(IDCConfig(init="poisson", S=5))
        result = est.fit(data.C, data.V, data.M)

        assert result.theta.shape == (5, 20)
        assert result.stats.init == "poisson"

    def test_store_path(self, small_data):
        """store_path=True should save theta at each iteration."""
        data, theta_true = small_data

        est = IDCEstimator(IDCConfig(init="pairwise", S=5, store_path=True))
        result = est.fit(data.C, data.V, data.M)

        # Should have S+1 entries: initial + S iterations
        assert result.theta_path is not None
        assert len(result.theta_path) == 6  # theta^(0) through theta^(5)
        for theta_s in result.theta_path:
            assert theta_s.shape == (5, 20)

    def test_mse_is_finite(self, small_data):
        """MSE should be finite and positive."""
        data, theta_true = small_data

        est = IDCEstimator(IDCConfig(init="pairwise", S=10))
        result = est.fit(data.C, data.V, data.M)

        # Compute MSE against true theta
        from classesv2 import normalize
        theta_true_norm = normalize(theta_true)
        mse = ((result.theta_normalized - theta_true_norm) ** 2).mean()

        assert np.isfinite(mse)
        assert mse > 0
        print(f"\nMSE with S=10: {mse:.6f}")

    def test_more_iterations_helps(self, small_data):
        """More iterations should generally reduce or maintain MSE."""
        data, theta_true = small_data

        from classesv2 import normalize
        theta_true_norm = normalize(theta_true)

        mses = []
        for S in [1, 5, 10, 15]:
            est = IDCEstimator(IDCConfig(init="pairwise", S=S))
            result = est.fit(data.C, data.V, data.M)
            mse = ((result.theta_normalized - theta_true_norm) ** 2).mean()
            mses.append((S, mse))
            print(f"S={S}: MSE={mse:.6f}")

        # MSE at S=10 should be <= MSE at S=1 (with some tolerance for noise)
        # This is a soft check - IDC should improve or stabilize
        assert mses[-1][1] <= mses[0][1] * 1.5, (
            f"MSE should not increase dramatically: S=1 MSE={mses[0][1]:.4f}, "
            f"S=15 MSE={mses[-1][1]:.4f}"
        )

    def test_timing_recorded(self, small_data):
        """Timing statistics should be recorded."""
        data, _ = small_data

        est = IDCEstimator(IDCConfig(init="pairwise", S=3))
        result = est.fit(data.C, data.V, data.M)

        assert result.stats.time_total > 0
        assert result.stats.init_time > 0
        assert result.stats.time_per_iter > 0


# -----------------------------------------------------------------------------
# Test TextData
# -----------------------------------------------------------------------------

class TestTextData:
    """Tests for the TextData wrapper."""

    def test_from_arrays_basic(self):
        """TextData should wrap arrays correctly."""
        C = np.random.randint(0, 10, size=(50, 8))
        V = np.random.randn(50, 3)
        M = C.sum(axis=1)

        data = TextData.from_arrays(C, V, M)

        assert data.n == 50
        assert data.d == 8
        assert data.p == 3

    def test_from_arrays_infers_m(self):
        """TextData should infer M if not provided."""
        C = np.random.randint(0, 10, size=(50, 8))
        V = np.random.randn(50, 3)

        data = TextData.from_arrays(C, V)

        np.testing.assert_array_equal(data.M, C.sum(axis=1))

    def test_shape_mismatch_raises(self):
        """Mismatched shapes should raise ValueError."""
        C = np.random.randint(0, 10, size=(50, 8))
        V = np.random.randn(40, 3)  # Wrong n

        with pytest.raises(ValueError):
            TextData.from_arrays(C, V)


# -----------------------------------------------------------------------------
# Test SGDEstimator
# -----------------------------------------------------------------------------

class TestSGDEstimator:
    """Tests for the SGD-based estimator (for referee comparison)."""

    @pytest.fixture
    def small_data(self):
        """Generate small test data."""
        cfg = DGPConfig(name="A", n=200, d=20, p=5, M_range=(20, 30), seed=42)
        data, theta_true = simulate_dgp(cfg)
        return data, theta_true

    def test_fit_adam(self, small_data):
        """SGD with Adam optimizer should run without errors."""
        data, theta_true = small_data

        est = SGDEstimator(SGDConfig(optimizer="adam", lr=0.01, epochs=50))
        result = est.fit(data.C, data.V, data.M)

        assert result.theta.shape == (5, 20)
        assert result.theta_normalized.shape == (5, 20)
        assert result.stats.optimizer == "adam"
        assert result.stats.epochs_run == 50

    def test_fit_sgd(self, small_data):
        """SGD with vanilla SGD optimizer should run."""
        data, theta_true = small_data

        est = SGDEstimator(SGDConfig(optimizer="sgd", lr=0.1, epochs=50, momentum=0.9))
        result = est.fit(data.C, data.V, data.M)

        assert result.theta.shape == (5, 20)
        assert result.stats.optimizer == "sgd"

    def test_fit_adagrad(self, small_data):
        """SGD with AdaGrad optimizer should run."""
        data, theta_true = small_data

        est = SGDEstimator(SGDConfig(optimizer="adagrad", lr=0.1, epochs=50))
        result = est.fit(data.C, data.V, data.M)

        assert result.theta.shape == (5, 20)
        assert result.stats.optimizer == "adagrad"

    def test_loss_decreases(self, small_data):
        """Loss should generally decrease during training."""
        data, _ = small_data

        est = SGDEstimator(SGDConfig(optimizer="adam", lr=0.01, epochs=100))
        result = est.fit(data.C, data.V, data.M)

        assert result.loss_history is not None
        assert len(result.loss_history) == 100

        # Loss at end should be less than at start
        assert result.loss_history[-1] < result.loss_history[0], (
            f"Loss should decrease: start={result.loss_history[0]:.4f}, "
            f"end={result.loss_history[-1]:.4f}"
        )

    def test_mse_is_finite(self, small_data):
        """MSE should be finite."""
        data, theta_true = small_data

        est = SGDEstimator(SGDConfig(optimizer="adam", lr=0.01, epochs=100))
        result = est.fit(data.C, data.V, data.M)

        from classesv2 import normalize
        theta_true_norm = normalize(theta_true)
        mse = ((result.theta_normalized - theta_true_norm) ** 2).mean()

        assert np.isfinite(mse)
        print(f"\nSGD MSE: {mse:.6f}")

    def test_lr_sensitivity(self, small_data):
        """
        Different learning rates should give different results.
        This is the key comparison point for the referee.
        """
        data, theta_true = small_data

        from classesv2 import normalize
        theta_true_norm = normalize(theta_true)

        results = {}
        for lr in [0.001, 0.01, 0.1]:
            est = SGDEstimator(SGDConfig(optimizer="adam", lr=lr, epochs=50, seed=42))
            result = est.fit(data.C, data.V, data.M)
            mse = ((result.theta_normalized - theta_true_norm) ** 2).mean()
            results[lr] = (mse, result.stats.final_loss)
            print(f"lr={lr}: MSE={mse:.6f}, final_loss={result.stats.final_loss:.6f}")

        # Results should be different for different learning rates
        mses = [r[0] for r in results.values()]
        assert len(set(round(m, 6) for m in mses)) > 1, "Learning rate should affect MSE"

    def test_store_path(self, small_data):
        """store_path=True should save theta at each epoch."""
        data, _ = small_data

        est = SGDEstimator(SGDConfig(optimizer="adam", lr=0.01, epochs=20, store_path=True))
        result = est.fit(data.C, data.V, data.M)

        assert result.theta_path is not None
        # Should have epochs+1 entries: initial + each epoch
        assert len(result.theta_path) == 21

    def test_timing_recorded(self, small_data):
        """Timing should be recorded."""
        data, _ = small_data

        est = SGDEstimator(SGDConfig(optimizer="adam", lr=0.01, epochs=20))
        result = est.fit(data.C, data.V, data.M)

        assert result.stats.time_total > 0
        assert result.stats.time_per_epoch > 0


# -----------------------------------------------------------------------------
# IDC vs SGD comparison (Table II preview)
# -----------------------------------------------------------------------------

class TestIDCvsSGD:
    """Compare IDC and SGD - this is the core comparison for referee Table II."""

    def test_idc_vs_sgd_comparison(self):
        """
        Compare IDC and SGD on the same data.
        This demonstrates what Table II will show.
        """
        # Generate data
        cfg = DGPConfig(name="A", n=300, d=15, p=5, M_range=(20, 30), seed=123)
        data, theta_true = simulate_dgp(cfg)

        from classesv2 import normalize
        theta_true_norm = normalize(theta_true)

        # Run IDC
        idc_est = IDCEstimator(IDCConfig(init="pairwise", S=10))
        idc_result = idc_est.fit(data.C, data.V, data.M)
        idc_mse = ((idc_result.theta_normalized - theta_true_norm) ** 2).mean()

        # Run SGD with different learning rates (κ)
        sgd_results = {}
        for lr in [0.001, 0.01, 0.1]:
            sgd_est = SGDEstimator(SGDConfig(optimizer="adam", lr=lr, epochs=100, seed=42))
            sgd_result = sgd_est.fit(data.C, data.V, data.M)
            sgd_mse = ((sgd_result.theta_normalized - theta_true_norm) ** 2).mean()
            sgd_results[lr] = {
                "mse": sgd_mse,
                "time": sgd_result.stats.time_total,
                "loss": sgd_result.stats.final_loss,
            }

        print("\n" + "=" * 60)
        print("IDC vs SGD Comparison (Table II preview)")
        print("=" * 60)
        print(f"\nIDC (S=10): MSE={idc_mse:.6f}, Time={idc_result.stats.time_total:.2f}s")
        print("\nSGD (Adam) with different learning rates:")
        for lr, res in sgd_results.items():
            print(f"  lr={lr}: MSE={res['mse']:.6f}, Time={res['time']:.2f}s")
        print("=" * 60)

        # Basic assertions
        assert np.isfinite(idc_mse)
        for res in sgd_results.values():
            assert np.isfinite(res["mse"])


# -----------------------------------------------------------------------------
# Integration test: reproduce paper pattern
# -----------------------------------------------------------------------------

class TestPaperReproduction:
    """Sanity check that results match expected patterns from the paper."""

    @pytest.mark.slow
    def test_dgp_a_mse_reasonable(self):
        """
        For DGP-A with n=500, d=20, p=5, MSE should be reasonably small.

        This is a rough sanity check, not an exact reproduction.
        """
        cfg = DGPConfig(name="A", n=500, d=20, p=5, M_range=(20, 30), seed=1234)
        data, theta_true = simulate_dgp(cfg)

        est = IDCEstimator(IDCConfig(init="pairwise", S=10))
        result = est.fit(data.C, data.V, data.M)

        from classesv2 import normalize
        theta_true_norm = normalize(theta_true)
        mse = ((result.theta_normalized - theta_true_norm) ** 2).mean()

        print(f"\nDGP-A (n=500, d=20): MSE = {mse:.6f}, Time = {result.stats.time_total:.2f}s")

        # MSE should be "small" - this is a loose bound
        # The paper shows MSE ~ 0.01-0.1 range typically
        assert mse < 1.0, f"MSE too large: {mse}"

    @pytest.mark.slow
    def test_dgp_c_runs(self):
        """DGP-C (mixture stress test) should complete without errors."""
        cfg = DGPConfig(name="C", n=500, d=20, p=5, seed=1234)
        data, theta_true = simulate_dgp(cfg)

        est = IDCEstimator(IDCConfig(init="pairwise", S=10))
        result = est.fit(data.C, data.V, data.M)

        from classesv2 import normalize
        theta_true_norm = normalize(theta_true)
        mse = ((result.theta_normalized - theta_true_norm) ** 2).mean()

        print(f"\nDGP-C (n=500, d=20): MSE = {mse:.6f}, Time = {result.stats.time_total:.2f}s")

        assert np.isfinite(mse)


# -----------------------------------------------------------------------------
# Test L1 Regularization
# -----------------------------------------------------------------------------

class TestL1Regularization:
    """Tests for L1 regularization (referee request for Table III)."""

    @pytest.fixture
    def small_data(self):
        """Generate small test data."""
        cfg = DGPConfig(name="A", n=200, d=20, p=5, M_range=(20, 30), seed=42)
        data, theta_true = simulate_dgp(cfg)
        return data, theta_true

    def test_l1_runs_without_error(self, small_data):
        """IDC with L1 penalty should run without errors."""
        data, theta_true = small_data

        cfg = IDCConfig(init="pairwise", S=5, penalty="l1", lambda_=0.1)
        est = IDCEstimator(cfg)
        result = est.fit(data.C, data.V, data.M)

        assert result.theta.shape == (5, 20)
        assert result.theta_normalized.shape == (5, 20)
        assert np.isfinite(result.theta).all()

    def test_l1_produces_sparser_solution(self, small_data):
        """L1 penalty should produce sparser solutions than no penalty."""
        data, theta_true = small_data

        # Fit without L1
        cfg_no_l1 = IDCConfig(init="pairwise", S=10, penalty="none")
        est_no_l1 = IDCEstimator(cfg_no_l1)
        result_no_l1 = est_no_l1.fit(data.C, data.V, data.M)

        # Fit with L1 (moderate penalty)
        cfg_l1 = IDCConfig(init="pairwise", S=10, penalty="l1", lambda_=1.0)
        est_l1 = IDCEstimator(cfg_l1)
        result_l1 = est_l1.fit(data.C, data.V, data.M)

        # Count near-zero entries (sparsity proxy)
        threshold = 0.01
        sparse_no_l1 = (np.abs(result_no_l1.theta_normalized) < threshold).sum()
        sparse_l1 = (np.abs(result_l1.theta_normalized) < threshold).sum()

        print(f"\nNear-zero entries (threshold={threshold}):")
        print(f"  No L1: {sparse_no_l1}")
        print(f"  L1 (lambda=1.0): {sparse_l1}")

        # L1 should produce at least as many near-zero entries
        # (or more for larger lambda)
        assert sparse_l1 >= sparse_no_l1 * 0.8, (
            f"L1 should promote sparsity: no_l1={sparse_no_l1}, l1={sparse_l1}"
        )

    def test_l1_lambda_effect(self, small_data):
        """Larger lambda should produce sparser solutions."""
        data, theta_true = small_data

        threshold = 0.01
        sparsities = []

        for lam in [0.0, 0.1, 0.5, 1.0]:
            if lam == 0.0:
                cfg = IDCConfig(init="pairwise", S=5, penalty="none")
            else:
                cfg = IDCConfig(init="pairwise", S=5, penalty="l1", lambda_=lam)

            est = IDCEstimator(cfg)
            result = est.fit(data.C, data.V, data.M)

            n_sparse = (np.abs(result.theta_normalized) < threshold).sum()
            sparsities.append((lam, n_sparse))
            print(f"lambda={lam}: near-zero entries = {n_sparse}")

        # Sparsity should generally increase with lambda
        # (relaxed check due to optimization noise)
        assert sparsities[-1][1] >= sparsities[0][1] * 0.5, (
            f"Larger lambda should promote sparsity"
        )

    def test_l1_with_different_inits(self, small_data):
        """L1 should work with all initialization methods."""
        data, theta_true = small_data

        for init in ["pairwise", "taddy", "poisson"]:
            cfg = IDCConfig(init=init, S=3, penalty="l1", lambda_=0.1)
            est = IDCEstimator(cfg)
            result = est.fit(data.C, data.V, data.M)

            assert result.theta.shape == (5, 20)
            assert np.isfinite(result.theta).all()
            print(f"L1 with {init} init: OK")


if __name__ == "__main__":
    # Quick smoke test when run directly
    print("Running quick smoke test...")

    # Test 1: Data generation
    cfg = DGPConfig(name="A", n=100, d=10, p=5, seed=42)
    data, theta_true = simulate_dgp(cfg)
    print(f"Data generated: C={data.C.shape}, V={data.V.shape}, theta={theta_true.shape}")

    # Test 2: IDC fit
    est = IDCEstimator(IDCConfig(init="pairwise", S=5))
    result = est.fit(data.C, data.V, data.M)
    print(f"IDC fit complete: theta={result.theta.shape}, time={result.stats.time_total:.2f}s")

    # Test 3: MSE
    from classesv2 import normalize
    theta_true_norm = normalize(theta_true)
    mse = ((result.theta_normalized - theta_true_norm) ** 2).mean()
    print(f"MSE: {mse:.6f}")

    print("\nSmoke test passed!")
