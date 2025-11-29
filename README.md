# Iterative Distributed Multinomial Regression 
This repository implements the iterative distributed computing estimator described in Iterative Distributed Multinomial Regression. The framework provides a computationally efficient alternative to traditional Maximum Likelihood Estimation (MLE) for multinomial logistic regression models with large choice sets.

## Key Features

- **Iterative Distributed Estimator**: A novel approach that significantly reduces computational time compared to MLE while maintaining asymptotic efficiency when initialized with a consistent estimator
- **Parametric Bootstrap Inference**: Implementation of consistent bootstrap methods for statistical inference
- **Large-Scale Support**: Efficiently handles models with large choice sets through distributed computing
- **Multiple Estimation Methods**:
  - IDMR with Pairwise Binomial initialization
  - Maximum Likelihood Estimation (MLE) baseline
  - Newton-Raphson optimization

## Performance Highlights

- Asymptotically efficient under weak dominance conditions
- Significantly faster computation compared to traditional MLE
- Consistent bootstrap inference for uncertainty quantification


## Installation

Clone this repository:

```bash
git clone https://github.com/yigitokar/IDMR.git
cd IDMR
 ```

### Dependencies

Install required packages:

```bash
pip install numpy pandas matplotlib torch tqdm statsmodels plotly cvxpy scipy
```

For MOSEK solver support (optional but recommended):
1. Obtain a MOSEK license
2. Set the license path in your environment:

```bash
export MOSEK_LICENSE_FILE="/path/to/mosek.lic"
```

## Quickstart (core API)

Run a small synthetic fit using the new core wrapper:

```bash
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
from idmr_core.simulation import DGPConfig, simulate_dgp
from idmr_core.models import IDCConfig, IDCEstimator

cfg = DGPConfig(name="A", n=200, d=10, p=5, M_range=(20, 30), seed=0)
data, _ = simulate_dgp(cfg)

est = IDCEstimator(IDCConfig(init="pairwise", S=5))
res = est.fit(data.C, data.V, data.M)
print("theta_normalized shape:", res.theta_normalized.shape)
print("time_total (s):", res.stats.time_total)
PY
```

Sanity check vs. MLE on a toy problem:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/sanity_idc_vs_mle.py
```
## Usage

### Basic Simulation

Run a basic simulation with default parameters: You can use notebook-runner-DMR.ipynb 

### Custom Data

To use your own dataset:

1. Create a `textData` object with your data:

```python
from classesv2 import textData
#C: count matrix (n x d)
#V: covariate matrix (n x p)
#m: total counts vector (n)
data = textData(C, V, m)
Initialize model
model = MDR_v11(data)
```
2. Fit the model:

```python
#For IDMR with Pairwise Binomial initialization
normalized_theta, theta = model.PARALLEL_PairwiseBinomial_fit(num_epochs=10)
```
### Optimizer Settings

You can modify optimizer settings in several ways:

1. For CVXPY-based optimization (in `PB.py`):

```python
scs_opts = {
'max_iters': 2500, # Maximum iterations
'eps': 1e-3, # Convergence tolerance
'alpha': 1.5, # Relaxation parameter
'scale': 1.0, # Scaling parameter
'normalize': True, # Normalization setting
'rho_x': 1e-3 # Regularization parameter
}
```

2. For Newton-Raphson optimization, modify parameters in `fit_NewtonRaphson` method:




## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

## Citation

If you use this code in your research, please cite:
