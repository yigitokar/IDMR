from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Create a dense (C,V,M) subset from the processed Yelp (Taddy-style) artifacts for local IDC runs."
    )
    ap.add_argument(
        "--src",
        type=Path,
        default=Path("data/processed/yelp_taddy"),
        help="Directory produced by scripts/process_yelp_taddy.py",
    )
    ap.add_argument("--n", type=int, default=5000, help="Number of reviews to sample")
    ap.add_argument("--d", type=int, default=2000, help="Number of tokens to keep (top by document frequency)")
    ap.add_argument(
        "--covars",
        type=str,
        default="rev",
        help="Which covariates to include: rev, geo, cat, biz (comma-separated). Default: rev",
    )
    ap.add_argument("--add-intercept", action="store_true", help="Prepend an intercept column to V")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for row sampling")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: data/processed/yelp_idmr_subset_n{n}_d{d}_{covars})",
    )
    args = ap.parse_args()

    src = args.src
    C = sparse.load_npz(src / "C.npz").tocsr()
    M = np.load(src / "M.npy")
    vocab = _read_lines(src / "vocab.txt")

    n_total, d_total = C.shape
    if args.n > n_total:
        raise SystemExit(f"--n {args.n} exceeds available rows {n_total}")
    if args.d > d_total:
        raise SystemExit(f"--d {args.d} exceeds available vocab size {d_total}")

    # Pick tokens by document frequency.
    df = np.asarray((C > 0).sum(axis=0)).ravel()
    top_cols = np.argsort(-df)[: args.d]
    top_cols = np.sort(top_cols)  # stable column order
    vocab_sel = [vocab[j] for j in top_cols.tolist()]

    rng = np.random.default_rng(args.seed)
    rows = rng.choice(n_total, size=args.n, replace=False)
    rows.sort()

    C_sub = C[rows][:, top_cols].toarray().astype(np.float64)
    M_sub = M[rows].astype(np.float64)

    covars = [c.strip().lower() for c in args.covars.split(",") if c.strip()]
    V_parts: list[np.ndarray] = []
    col_names: list[str] = []

    if "rev" in covars:
        REV = np.load(src / "REV.npy")
        REV_cols = _read_lines(src / "REV_columns.txt")
        V_parts.append(REV[rows].astype(np.float64))
        col_names.extend([f"rev:{c}" for c in REV_cols])

    if "geo" in covars:
        GEO = sparse.load_npz(src / "GEO.npz").tocsr()
        cities = _read_lines(src / "GEO_cities.txt")
        V_parts.append(GEO[rows].toarray().astype(np.float64))
        col_names.extend([f"geo:{c}" for c in cities])

    if "cat" in covars:
        CAT = sparse.load_npz(src / "CAT.npz").tocsr()
        cats = _read_lines(src / "CAT_categories.txt")
        V_parts.append(CAT[rows].toarray().astype(np.float64))
        col_names.extend([f"cat:{c}" for c in cats])

    if "biz" in covars:
        BIZ = sparse.load_npz(src / "BIZ.npz").tocsr()
        biz_ids = _read_lines(src / "BIZ_business_ids.txt")
        V_parts.append(BIZ[rows].toarray().astype(np.float64))
        col_names.extend([f"biz:{bid}" for bid in biz_ids])

    if not V_parts:
        raise SystemExit("--covars produced empty V; choose from rev,geo,cat,biz")

    V_sub = np.hstack(V_parts)
    if args.add_intercept:
        V_sub = np.hstack([np.ones((V_sub.shape[0], 1), dtype=np.float64), V_sub])
        col_names = ["intercept"] + col_names

    out = args.out
    if out is None:
        cov_tag = "_".join(covars) if covars else "none"
        out = Path(f"data/processed/yelp_idmr_subset_n{args.n}_d{args.d}_{cov_tag}")
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "C.npy", C_sub)
    np.save(out / "V.npy", V_sub)
    np.save(out / "M.npy", M_sub)
    (out / "vocab_selected.txt").write_text("\n".join(vocab_sel) + "\n", encoding="utf-8")
    (out / "V_columns.json").write_text(json.dumps(col_names, indent=2) + "\n", encoding="utf-8")
    np.save(out / "rows.npy", rows.astype(np.int32))

    summary = {
        "src": str(src),
        "out": str(out),
        "n": int(C_sub.shape[0]),
        "d": int(C_sub.shape[1]),
        "p": int(V_sub.shape[1]),
        "covars": covars,
        "add_intercept": bool(args.add_intercept),
        "seed": int(args.seed),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

