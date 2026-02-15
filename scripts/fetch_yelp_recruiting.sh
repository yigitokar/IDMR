#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-"$ROOT_DIR/data/raw/yelp-recruiting"}"
COMPETITION="yelp-recruiting"

KAGGLE_CMD=()
if command -v kaggle >/dev/null 2>&1; then
  KAGGLE_CMD=(kaggle)
elif [[ -x "$ROOT_DIR/.venv/bin/kaggle" ]]; then
  # Common for this repo: dependencies are installed into ./.venv via uv.
  KAGGLE_CMD=("$ROOT_DIR/.venv/bin/kaggle")
elif command -v uv >/dev/null 2>&1 && uv run kaggle --version >/dev/null 2>&1; then
  # Fallback: kaggle installed in the project env but .venv/bin isn't on PATH.
  KAGGLE_CMD=(uv run kaggle)
else
  cat <<'EOF'
ERROR: `kaggle` CLI not found on PATH and not detected under ./.venv.

Fix (pick one):
  1) Install into this repo's venv:
       uv pip install kaggle
     then re-run this script (it will use ./.venv/bin/kaggle)

  2) Activate your venv and re-run:
       source .venv/bin/activate
       ./scripts/fetch_yelp_recruiting.sh

  3) Install system-wide:
       python3 -m pip install --upgrade kaggle
EOF
  exit 1
fi

if [[ ! -f "$HOME/.kaggle/kaggle.json" ]]; then
  cat <<EOF
ERROR: missing Kaggle API token at: $HOME/.kaggle/kaggle.json

Setup:
  1) Download kaggle.json from Kaggle account settings (API token).
  2) Run:
       mkdir -p ~/.kaggle
       chmod 700 ~/.kaggle
       chmod 600 ~/.kaggle/kaggle.json
  3) Make sure you have joined/accepted the competition rules for "$COMPETITION".
EOF
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "Downloading Kaggle competition \"$COMPETITION\" into:"
echo "  $OUT_DIR"
"${KAGGLE_CMD[@]}" competitions download -c "$COMPETITION" -p "$OUT_DIR" --force

# Kaggle typically downloads ${COMPETITION}.zip which contains yelp_training_set.zip, etc.
TOP_ZIP="$OUT_DIR/${COMPETITION}.zip"
if [[ -f "$TOP_ZIP" ]]; then
  echo "Unzipping: $(basename "$TOP_ZIP")"
  unzip -o "$TOP_ZIP" -d "$OUT_DIR" >/dev/null
fi

for z in "$OUT_DIR"/yelp_*set*.zip "$OUT_DIR"/yelp_*_set*.zip "$OUT_DIR"/yelp_*_set.zip; do
  [[ -f "$z" ]] || continue
  echo "Unzipping: $(basename "$z")"
  unzip -o "$z" -d "$OUT_DIR" >/dev/null
done

echo
echo "Done. Quick check:"
python3 "$ROOT_DIR/scripts/verify_yelp_recruiting_download.py" "$OUT_DIR"
