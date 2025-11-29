"""
Tiny wiring check: run IDC (pairwise init, S=10) vs torch MLE on a toy problem.

Usage:
    UV_CACHE_DIR=.uv-cache uv run python scripts/sanity_idc_vs_mle.py
"""

import numpy as np
import torch

from idmr_core.models import IDCConfig, IDCEstimator
from idmr_core.simulation import DGPConfig, simulate_dgp
from PB import MDR_v11
from classesv2 import textData


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    # Small toy design
    sim_cfg = DGPConfig(name="A", n=50, d=4, p=3, M_range=(20, 30), seed=42)
    data, _theta_true = simulate_dgp(sim_cfg)

    # IDC fit (wraps MDR_v11 under the hood)
    idc = IDCEstimator(IDCConfig(init="pairwise", S=10, parallel_by_choice=True))
    idc_res = idc.fit(data.C, data.V, data.M)

    # Torch MLE baseline from MDR_v11
    engine = MDR_v11(textData_obj=textData(data.C, data.V, data.M))
    mle_norm, _mle_theta = engine.fit_MLE(num_iter=2000)

    # Compare normalized parameters
    rmse = np.linalg.norm(idc_res.theta_normalized - mle_norm) / np.sqrt(mle_norm.size)
    print(f"IDC vs MLE normalized theta RMSE: {rmse:.4f}")
    assert rmse < 0.3, "Sanity check failed: IDC and MLE differ more than tolerance."
    print("Sanity check passed.")


if __name__ == "__main__":
    main()
