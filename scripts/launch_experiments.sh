#!/usr/bin/env bash
# Launch IDMR Experiments
# Usage: ./launch_experiments.sh <table> <dgp> [additional_args...]
#
# Examples:
#   ./launch_experiments.sh 1 A
#   ./launch_experiments.sh 1 C
#   ./launch_experiments.sh 3 A
#   ./launch_experiments.sh 2 A --device cpu

set -euo pipefail

# Check required tools
for cmd in uv; do
    if ! command -v "$cmd" &> /dev/null; then
        echo "Error: $cmd not found. Run ./scripts/aws_setup.sh first."
        exit 1
    fi
done

TABLE="${1:-}"
DGP="${2:-}"

if [ -z "$TABLE" ] || [ -z "$DGP" ]; then
    echo "Usage: $0 <table> <dgp> [additional_args...]"
    echo "  table: 1, 2, or 3"
    echo "  dgp: A or C"
    exit 1
fi

shift 2

REPO_DIR="${REPO_DIR:-$HOME/IDMR}"
S3_BUCKET="${S3_BUCKET:-idmr-experiments-2025}"
N_WORKERS="${N_WORKERS:-12}"
SESSION_NAME="${SESSION_NAME:-exp}"

cd "$REPO_DIR"

if [ ! -f "scripts/run_experiments.py" ]; then
    echo "Error: scripts/run_experiments.py not found in $REPO_DIR."
    exit 1
fi

has_flag() {
    local flag="$1"
    shift
    for arg in "$@"; do
        if [ "$arg" = "$flag" ]; then
            return 0
        fi
    done
    return 1
}

# Build command as array (preserves quoting)
CMD=(uv run python scripts/run_experiments.py)

case "$TABLE" in
    1)
        CMD+=(
            --table 1
            --dgp "$DGP"
            --d 250 500 1000 2000 5000
            --S 10 20
            --B 50
            --n-workers "$N_WORKERS"
            --output-dir "s3://${S3_BUCKET}/table1/"
        )
        ;;
    2)
        CMD+=(
            --table 2
            --dgp "$DGP"
            --d 250 500 1000 2000 5000
            --optimizer sgd adam
            --lr 0.1 0.01 0.001
            --epochs 50
            --batch-size 256
            --B 50
            --output-dir "s3://${S3_BUCKET}/table2/"
        )
        if ! has_flag --device "$@"; then
            CMD+=(--device "${DEVICE:-cuda}")
        fi
        ;;
    3)
        CMD+=(
            --table 3
            --dgp "$DGP"
            --d 200 250 500 1000 2000
            --p 50 100 500 1000 2000
            --lambda 0 0.01 0.1
            --S 10
            --B 50
            --n-workers "$N_WORKERS"
            --output-dir "s3://${S3_BUCKET}/table3/"
        )
        ;;
    *)
        echo "Error: Unknown table: $TABLE (must be 1, 2, or 3)"
        exit 1
        ;;
esac

# Append any extra args from command line
CMD+=("$@")

echo "=== Launching Table $TABLE, DGP $DGP ==="
printf 'Command:'
printf ' %q' "${CMD[@]}"
echo ""
echo ""

# Run in tmux if session exists
if command -v tmux &> /dev/null && tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Sending to tmux session '$SESSION_NAME'..."
    cmd_str="$(printf '%q ' "${CMD[@]}")"
    tmux send-keys -t "$SESSION_NAME" "$cmd_str" Enter
    echo "Experiment started. Attach with: tmux attach -t $SESSION_NAME"
else
    echo "No tmux session found. Running directly..."
    "${CMD[@]}"
fi
