from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from idmr_core.data import load_yelp_subset
from idmr_core.models import IDCConfig, IDCEstimator


def main() -> int:
    ap = argparse.ArgumentParser(description="Fit IDC on a dense Yelp subset (C.npy/V.npy/M.npy).")
    ap.add_argument(
        "--subset-dir",
        type=Path,
        default=Path("data/processed/yelp_idmr_subset_n5000_d2000_rev_geo"),
        help="Directory produced by scripts/make_yelp_subset_for_idmr.py",
    )
    ap.add_argument("--S", type=int, default=10, help="Number of IDC iterations")
    ap.add_argument(
        "--init",
        type=str,
        default="pairwise",
        choices=["pairwise", "taddy", "poisson"],
        help="Initialization method",
    )
    ap.add_argument("--tol", type=float, default=0.0, help="Optional early-stopping tolerance")
    ap.add_argument("--no-parallel", action="store_true", help="Disable per-choice multiprocessing")
    ap.add_argument("--n-workers", type=int, default=None, help="Number of workers for per-choice multiprocessing")
    ap.add_argument("--l1", action="store_true", help="Enable L1 regularization in the Poisson subproblems")
    ap.add_argument("--lambda", dest="lambda_", type=float, default=0.0, help="L1 strength (only if --l1)")
    ap.add_argument("--log-eta", action="store_true", help="Print max/min(V@theta+mu) each iteration")
    ap.add_argument("--out", type=Path, default=None, help="Output .npz file (default under results/)")
    args = ap.parse_args()

    data = load_yelp_subset(args.subset_dir)

    cfg = IDCConfig(
        init=args.init,
        S=args.S,
        tol=args.tol,
        parallel_by_choice=not args.no_parallel,
        n_workers=args.n_workers,
        store_path=False,
        log_eta=args.log_eta,
        penalty="l1" if args.l1 else "none",
        lambda_=float(args.lambda_),
    )

    est = IDCEstimator(cfg)
    res = est.fit(data.C, data.V, data.M)

    out = args.out
    if out is None:
        Path("results").mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("results") / f"yelp_subset_idc_{args.subset_dir.name}_{args.init}_S{args.S}_{ts}.npz"

    payload = {
        "theta": res.theta,
        "theta_normalized": res.theta_normalized,
        "mu": res.mu,
        "stats_json": json.dumps(asdict(res.stats)),
        "config_json": json.dumps(asdict(cfg)),
        "subset_dir": str(args.subset_dir),
        "n": int(data.n),
        "d": int(data.d),
        "p": int(data.p),
    }

    np.savez_compressed(out, **payload)
    print(f"Wrote: {out}")
    print(f"theta shape: {res.theta.shape}  time_total(s): {res.stats.time_total:.3f}  S_eff: {res.stats.S_effective}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

