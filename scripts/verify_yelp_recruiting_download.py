from __future__ import annotations

import argparse
from pathlib import Path


EXPECTED = [
    "yelp_training_set_review.json",
    "yelp_training_set_business.json",
    "yelp_training_set_user.json",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Sanity-check Kaggle yelp-recruiting dump layout.")
    ap.add_argument("path", type=Path, help="Directory where you unzipped the Kaggle dump")
    args = ap.parse_args()

    root = args.path
    if not root.exists():
        raise SystemExit(f"Path does not exist: {root}")
    if not root.is_dir():
        raise SystemExit(f"Path is not a directory: {root}")

    json_files = sorted(root.rglob("*.json"))
    if not json_files:
        print("No .json files found. Did you unzip the training set archive(s)?")
        return 2

    by_name = {}
    for p in json_files:
        by_name.setdefault(p.name, []).append(p)

    print(f"Found {len(json_files)} JSON files under {root}:")
    for p in json_files[:30]:
        # Keep output short; this script is just a check.
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  - {p.relative_to(root)} ({size_mb:.1f} MB)")
    if len(json_files) > 30:
        print("  ...")

    missing = [name for name in EXPECTED if name not in by_name]
    if missing:
        print()
        print("WARNING: expected training-set JSON files not found:")
        for name in missing:
            print(f"  - {name}")
        print()
        print("If you only unzipped the top-level competition zip, also unzip yelp_training_set.zip.")
        return 1

    print()
    print("OK: training-set JSON files detected:")
    for name in EXPECTED:
        locs = by_name[name]
        # Prefer the shortest relative path if duplicates exist.
        best = min(locs, key=lambda p: len(str(p)))
        print(f"  - {name}: {best.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

