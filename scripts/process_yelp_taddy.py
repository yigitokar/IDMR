from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from array import array
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Tokenization (ported from https://github.com/TaddyLab/yelp/blob/master/code/tokenize.py)
# -----------------------------------------------------------------------------

_SYMBOLS = re.compile(r"(\W+)", re.U)
_NUMERIC = re.compile(r"(?<=\s)(\d+|\w\d+|\d+\w)(?=\s)", re.I | re.U)
_SWRD = re.compile(
    r"(?<=\s)(d|re|m|ve|s|n|to|a|the|an|and|or|in|at|with|for|are|is|the|if|of|at|but|and|or)(?=\s)",
    re.I | re.U,
)
_SUFFIX = re.compile(r"(?<=\w)(s|ings*|ives*|ly|led*|i*ed|i*es|ers*)(?=\s)")
_SEPS = re.compile(r"\s+")


def _clean_text(text: str) -> str:
    # Order matters; keep identical to Taddy's tokenize.py.
    text = " " + text.lower() + " "
    text = _SYMBOLS.sub(r" \1 ", text)
    text = _NUMERIC.sub(" ", text)
    text = _SWRD.sub(" ", text)
    text = _SUFFIX.sub("", text)
    text = _SEPS.sub(" ", text)
    return text


def tokenize_taddy(text: str) -> list[str]:
    txt = _clean_text(text)
    # only > 2 letter words (as in tokenize.py)
    return [w for w in txt.split() if len(w) > 2]


# -----------------------------------------------------------------------------
# Raw loaders
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class UserInfo:
    average_stars: float
    review_count: int
    votes_funny: int
    votes_useful: int
    votes_cool: int


@dataclass(frozen=True)
class BusinessInfo:
    # bkey is the integer index used by Taddy's build.R (1..n_biz, order encountered)
    bkey: int
    city: str
    state: str
    stars: float
    review_count: int
    categories: tuple[str, ...]


def _iter_json_lines(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_users(path: Path) -> dict[str, UserInfo]:
    users: dict[str, UserInfo] = {}
    for d in tqdm(_iter_json_lines(path), desc="users", unit="lines"):
        uid = d.get("user_id")
        if not uid:
            continue
        votes = d.get("votes") or {}
        users[str(uid)] = UserInfo(
            average_stars=float(d.get("average_stars", 0.0)),
            review_count=int(d.get("review_count", 0)),
            votes_funny=int(votes.get("funny", 0)),
            votes_useful=int(votes.get("useful", 0)),
            votes_cool=int(votes.get("cool", 0)),
        )
    return users


def load_businesses(path: Path) -> dict[str, BusinessInfo]:
    businesses: dict[str, BusinessInfo] = {}
    bkey = 0
    for d in tqdm(_iter_json_lines(path), desc="businesses", unit="lines"):
        bid = d.get("business_id")
        if not bid:
            continue
        bkey += 1
        cats = d.get("categories") or []
        # categories are a list in this Kaggle dump; keep order stable-ish.
        cat_tuple = tuple(str(c) for c in cats if c)
        businesses[str(bid)] = BusinessInfo(
            bkey=bkey,
            city=str(d.get("city", "")),
            state=str(d.get("state", "")),
            stars=float(d.get("stars", 0.0)),
            review_count=int(d.get("review_count", 0)),
            categories=cat_tuple,
        )
    return businesses


# -----------------------------------------------------------------------------
# Processing
# -----------------------------------------------------------------------------


SNAPSHOT_DATE = date(2013, 1, 19)  # from the Kaggle dump / Taddy build.R


@dataclass
class ReviewMeta:
    review_id: str
    user_id: str
    business_id: str
    date: str
    stars: int
    funny: int
    useful: int
    cool: int
    age_days: int


def _parse_age_days(iso_date: str) -> int:
    try:
        d = date.fromisoformat(iso_date)
        return int((SNAPSHOT_DATE - d).days)
    except Exception:
        return 0


def _save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def build_processed_dataset(
    raw_dir: Path,
    out_dir: Path,
    *,
    min_df: int = 21,  # keep tokens with df >= min_df; Taddy uses >20
    max_docs: int | None = None,
    seed: int = 0,
    cache_pass1: bool = True,
) -> None:
    review_path = raw_dir / "yelp_training_set_review.json"
    user_path = raw_dir / "yelp_training_set_user.json"
    biz_path = raw_dir / "yelp_training_set_business.json"

    for p in (review_path, user_path, biz_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"

    # Pass 0: load user + business tables
    users = load_users(user_path)
    businesses = load_businesses(biz_path)

    # Pass 1: determine included reviews and token document-frequency
    pass1_meta_path = cache_dir / "pass1_review_meta.pkl"
    pass1_df_path = cache_dir / "pass1_token_df.pkl"

    if cache_pass1 and pass1_meta_path.exists() and pass1_df_path.exists():
        meta: list[ReviewMeta] = _load_pickle(pass1_meta_path)
        token_df: dict[str, int] = _load_pickle(pass1_df_path)
    else:
        meta = []
        token_df: dict[str, int] = defaultdict(int)

        n_seen = 0
        for d in tqdm(_iter_json_lines(review_path), desc="reviews pass1", unit="lines"):
            uid = d.get("user_id")
            bid = d.get("business_id")
            if not uid or not bid:
                continue

            uid_s = str(uid)
            bid_s = str(bid)
            if uid_s not in users:
                continue
            if bid_s not in businesses:
                continue

            n_seen += 1
            if max_docs is not None and n_seen > max_docs:
                break

            votes = d.get("votes") or {}
            meta.append(
                ReviewMeta(
                    review_id=str(d.get("review_id", "")),
                    user_id=uid_s,
                    business_id=bid_s,
                    date=str(d.get("date", "")),
                    stars=int(d.get("stars", 0)),
                    funny=int(votes.get("funny", 0)),
                    useful=int(votes.get("useful", 0)),
                    cool=int(votes.get("cool", 0)),
                    age_days=_parse_age_days(str(d.get("date", ""))),
                )
            )

            text = d.get("text") or ""
            toks = tokenize_taddy(text)
            for w in set(toks):
                token_df[w] += 1

        if cache_pass1:
            _save_pickle(meta, pass1_meta_path)
            _save_pickle(dict(token_df), pass1_df_path)

    n0 = len(meta)
    if n0 == 0:
        raise RuntimeError("No reviews left after filtering by known users/businesses.")

    # Vocabulary: keep tokens with df >= min_df (Taddy: df > 20)
    vocab = sorted([w for w, c in token_df.items() if c >= min_df])
    if not vocab:
        raise RuntimeError(f"Empty vocab after df filter (min_df={min_df}).")

    vocab_path = out_dir / "vocab.txt"
    vocab_path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    token_to_col = {w: i for i, w in enumerate(vocab)}

    # Pass 2: build sparse document-term matrix C (CSR) using only the filtered vocab
    indices = array("I")
    data = array("I")
    indptr = array("I", [0])
    m_list: list[int] = []

    # If max_docs was used in pass1, apply the same stop in pass2 by iterating until n0 rows.
    target_rows = n0
    row_idx = 0

    for d in tqdm(_iter_json_lines(review_path), desc="reviews pass2", unit="lines"):
        uid = d.get("user_id")
        bid = d.get("business_id")
        if not uid or not bid:
            continue

        uid_s = str(uid)
        bid_s = str(bid)
        if uid_s not in users:
            continue
        if bid_s not in businesses:
            continue

        if row_idx >= target_rows:
            break

        toks = tokenize_taddy(d.get("text") or "")
        counts: dict[int, int] = {}
        for w in toks:
            j = token_to_col.get(w)
            if j is None:
                continue
            counts[j] = counts.get(j, 0) + 1

        # Store this row (sorted by column id for CSR correctness/perf)
        if counts:
            cols = sorted(counts.keys())
            for j in cols:
                indices.append(j)
                data.append(counts[j])
            m_list.append(int(sum(counts.values())))
        else:
            m_list.append(0)

        indptr.append(len(indices))
        row_idx += 1

    if row_idx != target_rows:
        raise RuntimeError(f"Pass2 row mismatch: expected {target_rows}, got {row_idx}")

    n = target_rows
    d_vocab = len(vocab)

    C = sparse.csr_matrix(
        (
            np.frombuffer(data, dtype=np.uint32),
            np.frombuffer(indices, dtype=np.uint32),
            np.frombuffer(indptr, dtype=np.uint32),
        ),
        shape=(n, d_vocab),
        dtype=np.int32,
    )

    M = np.asarray(m_list, dtype=np.int32)
    keep = M > 0
    if not bool(np.all(keep)):
        C = C[keep]
        M = M[keep]
        meta = [m for m, k in zip(meta, keep) if bool(k)]

    # Build covariates replicating Taddy build.R objects (BIZ, CAT, GEO, REV)
    n_final = len(meta)
    biz_bkeys = np.array([businesses[m.business_id].bkey for m in meta], dtype=np.int32)

    # BIZ: one-hot business id (columns in increasing bkey order, drop unused)
    used_bkeys = np.unique(biz_bkeys)
    used_bkeys.sort()
    bkey_to_col = {int(bkey): i for i, bkey in enumerate(used_bkeys.tolist())}
    biz_cols = np.array([bkey_to_col[int(bkey)] for bkey in biz_bkeys], dtype=np.int32)
    BIZ = sparse.csr_matrix(
        (np.ones(n_final, dtype=np.float32), (np.arange(n_final, dtype=np.int32), biz_cols)),
        shape=(n_final, len(used_bkeys)),
        dtype=np.float32,
    )

    # GEO: one-hot city
    cities = sorted({businesses[m.business_id].city for m in meta})
    city_to_col = {c: i for i, c in enumerate(cities)}
    geo_cols = np.array([city_to_col[businesses[m.business_id].city] for m in meta], dtype=np.int32)
    GEO = sparse.csr_matrix(
        (np.ones(n_final, dtype=np.float32), (np.arange(n_final, dtype=np.int32), geo_cols)),
        shape=(n_final, len(cities)),
        dtype=np.float32,
    )

    # CAT: categories. In Taddy's build.R the filter `colSums(CAT)>5` is applied
    # to the *business x category* matrix (before indexing by reviews), i.e.,
    # keep categories that appear in >5 businesses (not >5 reviews).
    cat_biz_counts: dict[str, int] = defaultdict(int)
    for info in businesses.values():
        for c in set(info.categories):
            if c:
                cat_biz_counts[c] += 1
    cats_keep = sorted([c for c, cnt in cat_biz_counts.items() if cnt > 5])
    cat_to_col = {c: i for i, c in enumerate(cats_keep)}

    cat_indices = array("I")
    cat_data = array("I")
    cat_indptr = array("I", [0])
    for m in meta:
        cols = sorted({cat_to_col[c] for c in businesses[m.business_id].categories if c in cat_to_col})
        for j in cols:
            cat_indices.append(j)
            cat_data.append(1)
        cat_indptr.append(len(cat_indices))

    CAT = sparse.csr_matrix(
        (
            np.frombuffer(cat_data, dtype=np.uint32).astype(np.float32),
            np.frombuffer(cat_indices, dtype=np.uint32),
            np.frombuffer(cat_indptr, dtype=np.uint32),
        ),
        shape=(n_final, len(cats_keep)),
        dtype=np.float32,
    )

    # REV: continuous covariates, then standardize (R: scale())
    age = np.array([m.age_days for m in meta], dtype=np.float64)
    n_zero_age = int((age <= 0).sum())
    if n_zero_age:
        # Avoid division by zero. Taddy's data appears to have age>0, but be defensive.
        age = np.maximum(age, 1.0)

    uid = [m.user_id for m in meta]
    usr_count = np.array([users[u].review_count for u in uid], dtype=np.float64)

    # usr.rank = rank(age) within user (ties=min), as in build.R
    usr_rank = np.zeros(n_final, dtype=np.float64)
    by_user: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for i, (u, a) in enumerate(zip(uid, age)):
        by_user[u].append((float(a), i))
    for u, items in by_user.items():
        items.sort(key=lambda t: t[0])  # ascending age
        last_age = None
        last_rank = 0
        for pos, (a, idx) in enumerate(items, start=1):
            if last_age is None or a != last_age:
                last_rank = pos
                last_age = a
            usr_rank[idx] = float(last_rank)

    sqrt_age = np.sqrt(age)
    biz_stars = np.array([businesses[m.business_id].stars for m in meta], dtype=np.float64)
    biz_count = np.array([businesses[m.business_id].review_count for m in meta], dtype=np.float64)

    rev_funny = np.array([m.funny for m in meta], dtype=np.float64)
    rev_useful = np.array([m.useful for m in meta], dtype=np.float64)
    rev_cool = np.array([m.cool for m in meta], dtype=np.float64)
    rev_stars = np.array([m.stars for m in meta], dtype=np.float64)

    usr_funny = np.array([users[u].votes_funny for u in uid], dtype=np.float64)
    usr_useful = np.array([users[u].votes_useful for u in uid], dtype=np.float64)
    usr_cool = np.array([users[u].votes_cool for u in uid], dtype=np.float64)
    usr_avg_stars = np.array([users[u].average_stars for u in uid], dtype=np.float64)

    # Avoid division by zero in per-user averages (should not happen if review_count>0)
    usr_count_safe = np.maximum(usr_count, 1.0)

    REV = np.column_stack(
        [
            rev_funny / sqrt_age,
            rev_useful / sqrt_age,
            rev_cool / sqrt_age,
            rev_stars - biz_stars,
            usr_funny / usr_count_safe,
            usr_useful / usr_count_safe,
            usr_cool / usr_count_safe,
            usr_avg_stars - 3.75,
            usr_count - usr_rank,
            biz_stars - 3.75,
            biz_count,
        ]
    )
    rev_cols = [
        "funny",
        "useful",
        "cool",
        "stars",
        "usr.funny",
        "usr.useful",
        "usr.cool",
        "usr.stars",
        "usr.count",
        "biz.stars",
        "biz.count",
    ]

    # Standardize (R scale uses sample sd, i.e., ddof=1)
    mu = REV.mean(axis=0)
    sd = REV.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    REV_scaled = ((REV - mu) / sd).astype(np.float32)

    # Write outputs
    sparse.save_npz(out_dir / "C.npz", C)
    np.save(out_dir / "M.npy", M)
    np.save(out_dir / "REV.npy", REV_scaled)

    sparse.save_npz(out_dir / "BIZ.npz", BIZ)
    sparse.save_npz(out_dir / "CAT.npz", CAT)
    sparse.save_npz(out_dir / "GEO.npz", GEO)

    pd.DataFrame([m.__dict__ for m in meta]).to_csv(out_dir / "reviews.csv.gz", index=False)

    # Column label files (for downstream interpretation)
    # BIZ columns correspond to used_bkeys (in increasing bkey order). Save business_ids in that same order.
    # We need reverse map from bkey->business_id; build it once.
    bkey_to_bid: dict[int, str] = {info.bkey: bid for bid, info in businesses.items()}
    biz_ids = [bkey_to_bid[int(bkey)] for bkey in used_bkeys.tolist()]
    (out_dir / "BIZ_business_ids.txt").write_text("\n".join(biz_ids) + "\n", encoding="utf-8")
    (out_dir / "CAT_categories.txt").write_text("\n".join(cats_keep) + "\n", encoding="utf-8")
    (out_dir / "GEO_cities.txt").write_text("\n".join(cities) + "\n", encoding="utf-8")
    (out_dir / "REV_columns.txt").write_text("\n".join(rev_cols) + "\n", encoding="utf-8")

    summary = {
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "n_reviews": int(n_final),
        "d_vocab": int(d_vocab),
        "p_biz": int(BIZ.shape[1]),
        "p_cat": int(CAT.shape[1]),
        "p_geo": int(GEO.shape[1]),
        "p_rev": int(REV_scaled.shape[1]),
        "p_total": int(BIZ.shape[1] + CAT.shape[1] + GEO.shape[1] + REV_scaled.shape[1]),
        "min_df": int(min_df),
        "max_docs": int(max_docs) if max_docs is not None else None,
        "n_zero_age_clamped": int(n_zero_age),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


def main() -> int:
    # Prefer the repo-local default used by scripts/fetch_yelp_recruiting.sh, but
    # also support the common pattern of placing the Kaggle dump at repo root.
    default_raw = Path("data/raw/yelp-recruiting/yelp_training_set")
    if not default_raw.exists():
        alt = Path("yelp-recruiting/yelp_training_set")
        if alt.exists():
            default_raw = alt

    ap = argparse.ArgumentParser(
        description="Process Kaggle yelp-recruiting training dump into Taddy-style matrices (C + covariates)."
    )
    ap.add_argument(
        "--raw-dir",
        type=Path,
        default=default_raw,
        help="Directory containing yelp_training_set_*.json",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/yelp_taddy"),
        help="Output directory for processed artifacts",
    )
    ap.add_argument(
        "--min-df",
        type=int,
        default=21,
        help="Keep tokens with document frequency >= min_df (Taddy uses >20).",
    )
    ap.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional cap on number of reviews processed (for quick tests).",
    )
    ap.add_argument("--seed", type=int, default=0, help="Reserved for future sampling; kept for CLI stability.")
    ap.add_argument(
        "--no-cache-pass1",
        action="store_true",
        help="Disable pass1 caching (token df + review meta).",
    )
    args = ap.parse_args()

    build_processed_dataset(
        args.raw_dir,
        args.out_dir,
        min_df=args.min_df,
        max_docs=args.max_docs,
        seed=args.seed,
        cache_pass1=not args.no_cache_pass1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
