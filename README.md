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

bash
git clone 
cd 

### Dependencies

Install required packages:

pip install numpy pandas matplotlib torch tqdm statsmodels plotly cvxpy scipy


For MOSEK solver support (optional but recommended):
1. Obtain a MOSEK license
2. Set the license path in your environment:

bash
export MOSEK_LICENSE_FILE="/path/to/mosek.lic"

## Usage

### Basic Simulation

Run a basic simulation with default parameters: You can use notebook-runner-DMR.ipynb 

### Custom Data

To use your own dataset:

1. Create a `textData` object with your data:

python
from classesv2 import textData
C: count matrix (n x d)
V: covariate matrix (n x p)
m: total counts vector (n)
data = textData(C, V, m)
Initialize model
model = MDR_v11(data)

2. Fit the model:

python
For MLE estimation
normalized_theta, theta = model.fit(num_epochs=10, initial_mu='zero')
For Pairwise Binomial initialization
normalized_theta, theta = model.PARALLEL_PairwiseBinomial_fit(num_epochs=10)

### Optimizer Settings

You can modify optimizer settings in several ways:

1. For CVXPY-based optimization (in `PB.py`):

python
scs_opts = {
'max_iters': 2500, # Maximum iterations
'eps': 1e-3, # Convergence tolerance
'alpha': 1.5, # Relaxation parameter
'scale': 1.0, # Scaling parameter
'normalize': True, # Normalization setting
'rho_x': 1e-3 # Regularization parameter
}


2. For Newton-Raphson optimization, modify parameters in `fit_NewtonRaphson` method:




## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

## Citation

If you use this code in your research, please cite:

