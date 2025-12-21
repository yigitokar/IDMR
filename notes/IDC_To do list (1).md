# To-Do List
October 24, 2025 (Original)
**Last Updated: December 21, 2025**

---

# Implementation Status

## ✅ COMPLETED

| Item | Description | Date |
|------|-------------|------|
| **Codebase Refactor** | Created `idmr_core` module with clean API wrapping legacy `MDR_v11` | Nov 2024 |
| **SGDEstimator** | Implemented SGD comparison (Adam, SGD, Adagrad optimizers) | Nov 2024 |
| **L1 Regularization** | Added L1 penalty to all optimization steps (init + iterations) | Dec 2024 |
| **Test Suite** | 32 tests passing (DGP, IDC, SGD, L1, paper reproduction) | Dec 2024 |
| **Refactor Verification** | Verified old code = new code (identical MSE) | Dec 2024 |

## 🔄 READY TO RUN

All code is implemented. Just need to run experiments and collect results.

---

# Quick Start (When You Return)

```bash
# 1. Verify everything still works
UV_CACHE_DIR=.uv-cache uv run pytest tests/ -v

# 2. Verify refactor (old code = new code)
UV_CACHE_DIR=.uv-cache uv run python scripts/verify_refactor.py
```

## How to Use the New API

```python
from idmr_core import (
    IDCEstimator, IDCConfig,
    SGDEstimator, SGDConfig,
    DGPConfig, simulate_dgp
)

# Generate data
cfg = DGPConfig(name="A", n=1000, d=250, p=5, M_range=(200, 300), seed=42)
data, theta_true = simulate_dgp(cfg)

# Table I: Large-d IDC
idc = IDCEstimator(IDCConfig(init="pairwise", S=10))
result = idc.fit(data.C, data.V, data.M)

# Table II: SGD comparison
sgd = SGDEstimator(SGDConfig(optimizer="adam", lr=0.01, epochs=1000))
result = sgd.fit(data.C, data.V, data.M)

# Table III: Regularized IDC
idc_l1 = IDCEstimator(IDCConfig(init="pairwise", S=10, penalty="l1", lambda_=0.1))
result = idc_l1.fit(data.C, data.V, data.M)
```

---

# Referee Comments (Original)

Based on the referee report, we need to conduct more simulation study and include an empirical analysis. Here are all the related comments.

1.  While the Monte Carlo simulations validate the finite-sample performance of the IDC estimator, an empirical application involving large choice sets (beyond `d = 150` in the paper) would strengthen the paper's practical relevance. For example, is your IDC estimator feasible on the publicly-available Yelp dataset used in Taddy (2015), where `n = 215,879`, `d = 13,938`, and `p = 11,940`? If so, does it arrive at different insights? Although the authors mentioned the issue of high-dimensional co-variates is being worked on in a companion paper, perhaps they can at least demonstrate their approach using a simplified version of Taddy's Yelp example with fewer covariates. Implementing both the IDC and Taddy's DMR on a common dataset would hopefully showcase the computational and inferential advantages of your approach.

2.  While the paper compares IDC to MLE, it would benefit from an in-depth discussion of how it compares to other modern large-scale multinomial regression methods that are currently deployed in practice, such as stochastic gradient-based methods. In a footnote, the authors mentioned that, unlike stochastic gradient descent, the IDC estimator does not inherently involve any tuning parameter. I feel a more thorough comparison is warranted. Besides, the number of iterations, S, in your algorithm can be viewed as a tuning parameter. Modern SGD variants can also leverage distributed computing frameworks in addition to GPU acceleration (e.g., PyTorch, TensorFlow), potentially offering competitive performance – particularly when the number of individuals n, is very large. Overall, I think it would strengthen the paper if the authors compare IDC with other existing approaches within the Monte Carlo simulation, as well as appealing to theoretical arguments of IDC's provable properties versus SGD's limitations (e.g., sensitivity to learning rates, lack of efficiency guarantees).

3.  In my view, a primary concern is the robustness of the MLE in large MNL models. When dealing with a large number of parameters, an efficient estimator may not perform well with a finite sample size, as discussed in James & Stein (1961).

It might be beneficial to introduce additional bias to reduce variance, such as using shrinkage estimation. While focusing on the computational challenges of traditional MLE is acceptable, it is also important to point out issues related to the large dimensionality of the parameter space. I am not suggesting to reconsider a robust estimator in this context, but I think the authors might want to discuss it somewhere in the paper. By the way, should cases like `d x n` or `d ≫ n` be explicitly ruled out? (By the first comment of the co-editor, we do not need to provide any theoretical result.)

---

# Simulation Experiments to Run

## Table I: Large-d IDC Performance

**Status: Code ready, need to run**

Parameters:
- `n = 1000` (fixed)
- `d ∈ {250, 500, 1000, 2000, 5000}`
- `p = 5` (unchanged)
- `M ∈ [200, 300]` (uniform)
- `S ∈ {10, 20}` (store all intermediate values S=11,...,20)
- `B = 10` repetitions first, then 50 if time permits
- DGP: A and C

| DGP | S  | d = 250 (MSE,Time) | d = 500 (MSE,Time) | d = 1000 (MSE,Time) | d = 2000 (MSE,Time) | d = 5000 (MSE,Time) |
| :-- | :-- | :------------------- | :------------------- | :-------------------- | :-------------------- | :-------------------- |
| A   | 10 | (XX,XX)            | (XX,XX)            | (XX,XX)             | (XX,XX)             | (XX,XX)             |
|     | 20 |                    |                    |                     |                     |                     |
| C   | 10 |                    |                    |                     |                     |                     |
|     | 20 |                    |                    |                     |                     |                     |

## Table II: SGD Comparison

**Status: Code ready, need to run**

SGD methods implemented:
- `M1 = Adam` (common default)
- `M2 = SGD` (vanilla)
- `M3 = Adagrad` (adaptive)

Tuning parameters (κ = learning rate):
- lr ∈ {0.001, 0.01, 0.1}

| DGP | SGD | κ | d = 250 (MSE,Time) | d = 500 (MSE,Time) | d = 1000 (MSE,Time) | d = 2000 (MSE,Time) | d = 5000 (MSE,Time) |
| :-- | :-- | :- | :------------------- | :------------------- | :-------------------- | :-------------------- | :-------------------- |
| A   | Adam  | 0.001 | (XX,XX)            | (XX,XX)            | (XX,XX)             | (XX,XX)             | (XX,XX)             |
|     |       | 0.01  |                    |                    |                     |                     |                     |
|     |       | 0.1   |                    |                    |                     |                     |                     |
|     | SGD   | 0.001 |                    |                    |                     |                     |                     |
|     |       | 0.01  |                    |                    |                     |                     |                     |
|     |       | 0.1   |                    |                    |                     |                     |                     |
| C   | Adam  | 0.001 |                    |                    |                     |                     |                     |
|     |       | ...   |                    |                    |                     |                     |                     |

## Table III: Regularized IDC (L1)

**Status: Code ready, need to run**

Parameters:
- `n = 1000`
- `p ∈ {50, 100, 500, 1000, 2000}`
- `d ∈ {200, 250, 500, 1000, 2000}`
- `λ` (regularization strength) - need to determine reasonable values

| n = 1000 | p = 50 (MSE,Time) | p = 100 (MSE,Time) | p = 500 (MSE,Time) | p = 1000 (MSE,Time) | p = 2000 (MSE,Time) |
| :------- | :------------------ | :------------------- | :------------------- | :-------------------- | :-------------------- |
| d        |                     |                      |                      |                       |                       |
| 200      | (XX,XX)           | (XX,XX)            | (XX,XX)            | (XX,XX)             | (XX,XX)             |
| 250      |                     |                      |                      |                       |                       |
| 500      |                     |                      |                      |                       |                       |
| 1000     |                     |                      |                      |                       |                       |
| 2000     |                     |                      |                      |                       |                       |

---

# Computational Resources

To make the simulation faster, we may need extra computational power. Research fund available for CPU instances (AWS or similar).

**Strategy:**
- Test locally with small d (250-500) first
- Scale to AWS cluster for d ≥ 1000
- Parallelize across (DGP, d, method) combinations

---

# Empirical Analysis (Yelp)

Once the simulation, especially Part 3 where a regularization term is added, is done, we can use the data in Taddy to conduct the empirical analysis. Depending on the result of Part 3, we can use the original setup or reduce `d` and `p`.

Yelp dataset: `n = 215,879`, `d = 13,938`, `p = 11,940`

---

# File Structure Reference

```
IDMR/
├── idmr_core/           # NEW clean API
│   ├── __init__.py      # Exports: IDCEstimator, SGDEstimator, etc.
│   ├── models.py        # IDCEstimator, SGDEstimator, configs
│   ├── simulation.py    # DGP-A, DGP-C generators
│   └── data.py          # TextData wrapper
├── classesv2.py         # Legacy engine (CELL_minQ_kn, etc.)
├── PB.py                # Legacy MDR_v11 class
├── tests/
│   └── test_idmr_core.py  # 32 tests
├── scripts/
│   ├── verify_refactor.py   # Compare old vs new code
│   └── sanity_idc_vs_mle.py # IDC vs MLE comparison
└── notes/
    └── IDC_To do list (1).md  # This file
```
