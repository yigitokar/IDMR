#!/usr/bin/env python3
"""
Run IDMR experiment grids for Tables I, II, III.

Examples:
  python scripts/run_experiments.py --table 1 --dgp A C --d 250 500 --S 10 20 --B 10
  python scripts/run_experiments.py --table 2 --dgp A C --d 250 500 --optimizer adam sgd --lr 0.01 0.001 --B 10
  python scripts/run_experiments.py --table 3 --dgp A --d 200 250 --p 50 100 --lambda 0 0.01 --B 5
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _set_default_thread_caps() -> None:
    cap = "1"
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(var, cap)


_set_default_thread_caps()

import numpy as np

from classesv2 import normalize
from idmr_core import (
    DGPConfig,
    IDCConfig,
    IDCEstimator,
    SGDConfig,
    SGDEstimator,
    simulate_dgp,
)


DEFAULTS = {
    1: {
        "dgp": ["A", "C"],
        "d": [250, 500, 1000, 2000, 5000],
        "S": [10, 20],
        "p": [5],
        "m_range": (200, 300),
    },
    2: {
        "dgp": ["A", "C"],
        "d": [250, 500, 1000, 2000, 5000],
        "p": [5],
        "m_range": (200, 300),
        "optimizer": ["sgd", "adam"],
        "lr": [0.1, 0.01, 0.001],
    },
    3: {
        "dgp": ["A"],
        "d": [200, 250, 500, 1000, 2000],
        "p": [50, 100, 500, 1000, 2000],
        "S": [10],
        "lambda": [0.0, 0.01, 0.1],
        "m_range": (20, 30),
    },
}


FIELDNAMES = [
    "table",
    "dgp",
    "rep",
    "seed",
    "theta_seed",
    "n",
    "d",
    "p",
    "m_min",
    "m_max",
    "method",
    "solver",
    "init",
    "S",
    "S_effective",
    "optimizer",
    "lr",
    "lambda",
    "epochs",
    "batch_size",
    "device",
    "n_workers",
    "time_total",
    "time_per_iter",
    "time_per_epoch",
    "init_time",
    "final_loss",
    "mse",
    "status",
    "error",
    "timestamp",
]


@dataclass(frozen=True)
class Combo:
    table: int
    dgp: str
    n: int
    d: int
    p: int
    m_range: Tuple[int, int]
    method: str
    init: Optional[str] = None
    S: Optional[int] = None
    optimizer: Optional[str] = None
    lr: Optional[float] = None
    lambda_: Optional[float] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    device: Optional[str] = None
    theta_scale: float = 1.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IDMR experiments.")
    parser.add_argument("--table", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--dgp", nargs="+", choices=["A", "C"], default=None)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--d", nargs="+", type=int, default=None)
    parser.add_argument("--p", nargs="+", type=int, default=None)
    parser.add_argument("--S", nargs="+", type=int, default=None)
    parser.add_argument("--lambda", dest="lambda_vals", nargs="+", type=float, default=None)
    parser.add_argument("--optimizer", nargs="+", choices=["sgd", "adam", "adagrad"], default=None)
    parser.add_argument("--lr", nargs="+", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--init", choices=["pairwise", "taddy", "poisson"], default="pairwise")
    parser.add_argument("--solver", choices=["scs", "mosek", "clarabel"], default=None)
    parser.add_argument("--tol", type=float, default=0.0)
    parser.add_argument("--B", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--theta-seed", type=int, default=1233)
    parser.add_argument("--rep-start", type=int, default=0)
    parser.add_argument("--rep-end", type=int, default=None)
    parser.add_argument("--m-range", nargs=2, type=int, default=None, metavar=("M_MIN", "M_MAX"))
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--partition", type=int, default=0)
    parser.add_argument("--n-partitions", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--local-output-dir", type=str, default=None)
    parser.add_argument("--output-suffix", type=str, default="")
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--theta-scale", type=float, default=None,
                        help="Scale for theta_true ~ N(0, scale^2). Use 'inv_sqrt_p' logic via --theta-scale-inv-sqrt-p.")
    parser.add_argument("--theta-scale-inv-sqrt-p", action="store_true",
                        help="Set theta_scale = 1/sqrt(p) for each config (constant signal strength as p varies).")
    parser.add_argument("--log-eta", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def _apply_defaults(args: argparse.Namespace) -> None:
    defaults = DEFAULTS[args.table]
    if args.dgp is None:
        args.dgp = defaults["dgp"]
    if args.d is None:
        args.d = defaults["d"]
    if args.p is None:
        args.p = defaults["p"]
    if args.S is None and "S" in defaults:
        args.S = defaults["S"]
    if args.lambda_vals is None and "lambda" in defaults:
        args.lambda_vals = defaults["lambda"]
    if args.optimizer is None and "optimizer" in defaults:
        args.optimizer = defaults["optimizer"]
    if args.lr is None and "lr" in defaults:
        args.lr = defaults["lr"]
    if args.m_range is None:
        args.m_range = defaults["m_range"]
    if args.rep_end is None:
        args.rep_end = args.B


def _format_float_for_name(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _build_output_name(combo: Combo, suffix: str) -> str:
    if combo.table == 1:
        name = f"dgp_{combo.dgp}_d_{combo.d}_S_{combo.S}_init_{combo.init}"
    elif combo.table == 2:
        lr_tag = _format_float_for_name(combo.lr or 0.0)
        name = (
            f"dgp_{combo.dgp}_d_{combo.d}_opt_{combo.optimizer}_lr_{lr_tag}_"
            f"ep_{combo.epochs}_bs_{combo.batch_size}"
        )
    else:
        lam_tag = _format_float_for_name(combo.lambda_ or 0.0)
        name = f"dgp_{combo.dgp}_d_{combo.d}_p_{combo.p}_lam_{lam_tag}_S_{combo.S}"
    if suffix:
        name = f"{name}_{suffix}"
    return f"{name}.csv"


def _resolve_paths(output_dir: str, local_output_dir: Optional[str]) -> Tuple[Path, Optional[str]]:
    if output_dir.startswith("s3://"):
        local_dir = Path(local_output_dir or "/tmp/idmr-results")
        return local_dir, output_dir
    return Path(output_dir), None


def _ensure_local_copy(local_path: Path, remote_path: Optional[str]) -> None:
    if remote_path is None:
        return
    if local_path.exists():
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["aws", "s3", "cp", remote_path, str(local_path)]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        if b"NoSuchKey" in result.stderr or b"404" in result.stderr:
            return
        return


def _sync_to_remote(local_path: Path, remote_path: Optional[str]) -> None:
    if remote_path is None:
        return
    cmd = ["aws", "s3", "cp", str(local_path), remote_path]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"Failed to upload {local_path} to {remote_path}: {stderr}")


def _load_existing_seeds(path: Path) -> set[int]:
    if not path.exists():
        return set()
    seeds = set()
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if row.get("status", "ok") == "ok":
                    seeds.add(int(row["seed"]))
            except (KeyError, ValueError):
                continue
    return seeds


def _write_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _log(msg: str) -> None:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now} UTC] {msg}", flush=True)


def _build_combos(args: argparse.Namespace) -> List[Combo]:
    combos: List[Combo] = []
    if args.table == 1:
        for dgp in args.dgp:
            for d in args.d:
                for S in args.S:
                    combos.append(
                        Combo(
                            table=1,
                            dgp=dgp,
                            n=args.n,
                            d=d,
                            p=args.p[0],
                            m_range=tuple(args.m_range),
                            method="idc",
                            init=args.init,
                            S=S,
                        )
                    )
    elif args.table == 2:
        for dgp in args.dgp:
            for d in args.d:
                for optimizer in args.optimizer:
                    for lr in args.lr:
                        combos.append(
                            Combo(
                                table=2,
                                dgp=dgp,
                                n=args.n,
                                d=d,
                                p=args.p[0],
                                m_range=tuple(args.m_range),
                                method="sgd",
                                optimizer=optimizer,
                                lr=lr,
                                epochs=args.epochs,
                                batch_size=args.batch_size,
                                device=args.device,
                            )
                        )
    else:
        for dgp in args.dgp:
            for d in args.d:
                for p in args.p:
                    for lam in args.lambda_vals:
                        for S in args.S:
                            ts = 1.0 / np.sqrt(p) if getattr(args, 'theta_scale_inv_sqrt_p', False) else (args.theta_scale or 1.0)
                            combos.append(
                                Combo(
                                    table=3,
                                    dgp=dgp,
                                    n=args.n,
                                    d=d,
                                    p=p,
                                    m_range=tuple(args.m_range),
                                    method="idc",
                                    init=args.init,
                                    S=S,
                                    lambda_=lam,
                                    theta_scale=ts,
                                )
                            )
    return combos


def _partition_combos(combos: Sequence[Combo], partition: int, n_partitions: int) -> List[Combo]:
    if n_partitions <= 1:
        return list(combos)
    if partition < 0 or partition >= n_partitions:
        raise ValueError(f"partition must be in [0, {n_partitions - 1}]")
    return [c for idx, c in enumerate(combos) if idx % n_partitions == partition]


def _run_idc(
    combo: Combo,
    seed: int,
    theta_seed: int,
    n_workers: Optional[int],
    tol: float,
    parallel_by_choice: bool,
    log_eta: bool,
    solver: Optional[str],
) -> Tuple[Dict[str, Any], float]:
    sim_cfg = DGPConfig(
        name=combo.dgp,
        n=combo.n,
        d=combo.d,
        p=combo.p,
        M_range=combo.m_range,
        seed=seed,
        theta_seed=theta_seed,
        theta_scale=combo.theta_scale,
    )
    data, theta_true = simulate_dgp(sim_cfg)
    theta_true_norm = normalize(theta_true)

    cfg = IDCConfig(
        init=combo.init or "pairwise",
        S=combo.S or 10,
        tol=tol,
        penalty="l1" if (combo.lambda_ is not None and combo.lambda_ > 0) else "none",
        lambda_=combo.lambda_ or 0.0,
        parallel_by_choice=parallel_by_choice,
        n_workers=n_workers,
        log_eta=log_eta,
        poisson_solver=f"cvxpy_{solver}" if solver else "cvxpy_scs",
    )
    est = IDCEstimator(cfg)
    res = est.fit(data.C, data.V, data.M)

    mse = ((res.theta_normalized - theta_true_norm) ** 2).mean()
    row = {
        "method": "idc",
        "solver": cfg.poisson_solver,
        "init": cfg.init,
        "S": cfg.S,
        "S_effective": res.stats.S_effective,
        "time_total": res.stats.time_total,
        "time_per_iter": res.stats.time_per_iter,
        "init_time": res.stats.init_time,
        "mse": mse,
        "m_min": int(data.M.min()),
        "m_max": int(data.M.max()),
    }
    return row, mse


def _run_sgd(
    combo: Combo,
    seed: int,
    theta_seed: int,
) -> Tuple[Dict[str, Any], float]:
    sim_cfg = DGPConfig(
        name=combo.dgp,
        n=combo.n,
        d=combo.d,
        p=combo.p,
        M_range=combo.m_range,
        seed=seed,
        theta_seed=theta_seed,
        theta_scale=combo.theta_scale,
    )
    data, theta_true = simulate_dgp(sim_cfg)
    theta_true_norm = normalize(theta_true)

    cfg = SGDConfig(
        optimizer=combo.optimizer or "adam",
        lr=combo.lr or 0.01,
        batch_size=combo.batch_size or 256,
        epochs=combo.epochs or 50,
        device=combo.device or "cuda",
        seed=seed,
    )
    est = SGDEstimator(cfg)
    res = est.fit(data.C, data.V, data.M)

    mse = ((res.theta_normalized - theta_true_norm) ** 2).mean()
    row = {
        "method": "sgd",
        "optimizer": cfg.optimizer,
        "lr": cfg.lr,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "device": cfg.device,
        "time_total": res.stats.time_total,
        "time_per_epoch": res.stats.time_per_epoch,
        "final_loss": res.stats.final_loss,
        "mse": mse,
        "m_min": int(data.M.min()),
        "m_max": int(data.M.max()),
    }
    return row, mse


def main() -> None:
    args = _parse_args()
    _apply_defaults(args)

    combos = _build_combos(args)
    combos = _partition_combos(combos, args.partition, args.n_partitions)

    if args.dry_run:
        _log(f"Dry run: {len(combos)} combos")
        for combo in combos:
            _log(f"{combo}")
        return

    local_root, remote_root = _resolve_paths(args.output_dir, args.local_output_dir)
    rep_indices = list(range(args.rep_start, args.rep_end))

    for combo in combos:
        file_name = _build_output_name(combo, args.output_suffix)
        local_path = local_root / f"table{combo.table}" / file_name
        remote_path = None
        if remote_root:
            remote_path = f"{remote_root.rstrip('/')}/table{combo.table}/{file_name}"

        _ensure_local_copy(local_path, remote_path)
        existing_seeds = _load_existing_seeds(local_path)

        _log(f"Starting combo: {file_name} (existing seeds: {len(existing_seeds)})")

        for rep in rep_indices:
            seed = args.base_seed + rep
            if seed in existing_seeds:
                continue

            row: Dict[str, Any] = {
                "table": combo.table,
                "dgp": combo.dgp,
                "rep": rep,
                "seed": seed,
                "theta_seed": args.theta_seed,
                "n": combo.n,
                "d": combo.d,
                "p": combo.p,
                "m_min": "",
                "m_max": "",
                "method": combo.method,
                "solver": args.solver or "",
                "init": combo.init or "",
                "S": combo.S or "",
                "S_effective": "",
                "optimizer": combo.optimizer or "",
                "lr": combo.lr or "",
                "lambda": combo.lambda_ if combo.lambda_ is not None else "",
                "epochs": combo.epochs or "",
                "batch_size": combo.batch_size or "",
                "device": combo.device or "",
                "n_workers": args.n_workers or "",
                "time_total": "",
                "time_per_iter": "",
                "time_per_epoch": "",
                "init_time": "",
                "final_loss": "",
                "mse": "",
                "status": "ok",
                "error": "",
                "timestamp": datetime.utcnow().isoformat(),
            }

            try:
                if combo.table in (1, 3):
                    details, _mse = _run_idc(
                        combo=combo,
                        seed=seed,
                        theta_seed=args.theta_seed,
                        n_workers=args.n_workers,
                        tol=args.tol,
                        parallel_by_choice=not args.no_parallel,
                        log_eta=args.log_eta,
                        solver=args.solver,
                    )
                else:
                    details, _mse = _run_sgd(
                        combo=combo,
                        seed=seed,
                        theta_seed=args.theta_seed,
                    )

                row.update(details)

            except Exception as exc:
                row["status"] = "error"
                row["error"] = str(exc)
                _log(f"Error on seed {seed} for {file_name}: {exc}")
                if args.fail_fast:
                    raise

            _write_row(local_path, row)
            try:
                _sync_to_remote(local_path, remote_path)
            except Exception as exc:
                _log(f"Upload failed for {file_name}: {exc}")
                if args.fail_fast:
                    raise

        _log(f"Finished combo: {file_name}")


if __name__ == "__main__":
    main()
