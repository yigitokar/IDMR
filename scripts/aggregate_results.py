#!/usr/bin/env python3
"""
Aggregate per-seed experiment CSVs into table-level summaries.

Examples:
  python scripts/aggregate_results.py --input-dir results --output-dir results/aggregated
  python scripts/aggregate_results.py --input-dir results --table 1 2
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


METRIC_COLS = [
    "mse",
    "time_total",
    "time_per_iter",
    "time_per_epoch",
    "init_time",
    "final_loss",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate IDMR experiment results.")
    parser.add_argument("--input-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="results/aggregated")
    parser.add_argument("--table", nargs="+", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--write-raw", action="store_true")
    parser.add_argument("--write-failures", action="store_true")
    return parser.parse_args()


def _infer_table_from_path(path: Path) -> Optional[int]:
    for part in path.parts:
        match = re.match(r"table(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def _collect_csvs(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.rglob("*.csv") if p.is_file()])


def _group_cols_for_table(table: int, df: pd.DataFrame) -> List[str]:
    if table == 1:
        cols = ["table", "dgp", "n", "d", "p", "S", "init"]
    elif table == 2:
        cols = ["table", "dgp", "n", "d", "p", "optimizer", "lr", "epochs", "batch_size", "device"]
    else:
        cols = ["table", "dgp", "n", "d", "p", "S", "init", "lambda"]
    return [c for c in cols if c in df.columns]


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    df.columns = ["{}_{}".format(a, b) if b else str(a) for a, b in df.columns]
    return df


def _aggregate_table(df: pd.DataFrame, table: int) -> Dict[str, pd.DataFrame]:
    df = df.copy()
    if "status" not in df.columns:
        df["status"] = "ok"
    df_ok = df[df["status"] == "ok"].copy()

    group_cols = _group_cols_for_table(table, df)
    metric_cols = [c for c in METRIC_COLS if c in df.columns]

    if not metric_cols:
        summary = pd.DataFrame()
    else:
        agg_map = {col: ["mean", "std"] for col in metric_cols}
        summary = df_ok.groupby(group_cols, dropna=False).agg(agg_map)
        summary = _flatten_columns(summary).reset_index()

    counts_total = df.groupby(group_cols, dropna=False).size().rename("n_total")
    counts_ok = df_ok.groupby(group_cols, dropna=False).size().rename("n_success")
    counts = pd.concat([counts_total, counts_ok], axis=1).reset_index()
    counts["n_success"] = counts["n_success"].fillna(0).astype(int)
    counts["n_total"] = counts["n_total"].fillna(0).astype(int)
    counts["n_failed"] = counts["n_total"] - counts["n_success"]

    if not summary.empty:
        summary = summary.merge(counts, on=group_cols, how="left")
    else:
        summary = counts

    failures = df[df["status"] != "ok"].copy()

    return {
        "summary": summary,
        "raw": df,
        "failures": failures,
    }


def main() -> None:
    args = _parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = _collect_csvs(input_dir)
    if not csv_paths:
        raise SystemExit(f"No CSV files found under {input_dir}")

    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path)
        if "table" not in df.columns or df["table"].isna().all():
            inferred = _infer_table_from_path(path)
            if inferred is not None:
                df["table"] = inferred
        df["source_file"] = str(path)
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True, sort=False)

    tables = args.table or sorted(set(all_df["table"].dropna().astype(int).tolist()))
    for table in tables:
        table_df = all_df[all_df["table"] == table].copy()
        if table_df.empty:
            continue

        outputs = _aggregate_table(table_df, table)

        summary_path = output_dir / f"table{table}_summary.csv"
        outputs["summary"].to_csv(summary_path, index=False)

        if args.write_raw:
            raw_path = output_dir / f"table{table}_raw.csv"
            outputs["raw"].to_csv(raw_path, index=False)

        if args.write_failures:
            failures_path = output_dir / f"table{table}_failed.csv"
            outputs["failures"].to_csv(failures_path, index=False)

    combined_summary = output_dir / "combined_summary.csv"
    summary_frames = []
    for table in tables:
        path = output_dir / f"table{table}_summary.csv"
        if path.exists():
            summary_frames.append(pd.read_csv(path))
    if summary_frames:
        pd.concat(summary_frames, ignore_index=True, sort=False).to_csv(combined_summary, index=False)


if __name__ == "__main__":
    main()
