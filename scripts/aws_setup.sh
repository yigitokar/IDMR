#!/usr/bin/env bash
# AWS EC2 Instance Setup Script for IDMR Experiments (Ubuntu 22.04+)
set -euo pipefail

echo "=== IDMR AWS Setup ==="

# Install base dependencies if missing (Ubuntu via apt-get)
INSTALL_DEPS="${INSTALL_DEPS:-1}"
missing_pkgs=()
for pkg_cmd in curl git tmux aws; do
    if ! command -v "$pkg_cmd" &> /dev/null; then
        case "$pkg_cmd" in
            aws) missing_pkgs+=("awscli") ;;
            *) missing_pkgs+=("$pkg_cmd") ;;
        esac
    fi
done

if [ "${#missing_pkgs[@]}" -gt 0 ]; then
    if [ "$INSTALL_DEPS" = "1" ] && command -v apt-get &> /dev/null; then
        echo "Installing missing packages: ${missing_pkgs[*]}"
        sudo apt-get update -y
        sudo apt-get install -y "${missing_pkgs[@]}"
    else
        echo "Missing packages: ${missing_pkgs[*]}"
        echo "Install them first or set INSTALL_DEPS=1 on Ubuntu."
        exit 1
    fi
fi

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if [ -f "$HOME/.bashrc" ]; then
        if ! grep -q ".local/bin" "$HOME/.bashrc"; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
        fi
        if ! grep -q ".cargo/bin" "$HOME/.bashrc"; then
            echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> "$HOME/.bashrc"
        fi
    fi
fi

if ! command -v uv &> /dev/null; then
    echo "uv not found in PATH after install."
    exit 1
fi

# Clone or update repository
REPO_URL="${IDMR_REPO_URL:-https://github.com/yigitokar/IDMR.git}"
REPO_DIR="${REPO_DIR:-$HOME/IDMR}"

if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning repository..."
    git clone "$REPO_URL" "$REPO_DIR"
else
    echo "Repository exists, pulling latest..."
    git -C "$REPO_DIR" pull
fi

cd "$REPO_DIR"

# Sync dependencies
echo "Syncing dependencies..."
uv sync

# Create tmux session if available
SESSION_NAME="${SESSION_NAME:-exp}"
if command -v tmux &> /dev/null; then
    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "Creating tmux session: $SESSION_NAME"
        tmux new-session -d -s "$SESSION_NAME"
    else
        echo "tmux session '$SESSION_NAME' already exists"
    fi
else
    echo "tmux not found; skipping session creation."
fi

# Configure AWS region (only if not already set)
if command -v aws &> /dev/null; then
    AWS_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-2}}"
    CURRENT_REGION="$(aws configure get region || true)"
    if [ -z "$CURRENT_REGION" ] || [ "${FORCE_AWS_REGION:-0}" = "1" ]; then
        aws configure set default.region "$AWS_REGION"
    else
        echo "AWS region already set to '$CURRENT_REGION'."
    fi
fi

# Create local results directory
mkdir -p /tmp/idmr-results

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "  1. Attach to tmux: tmux attach -t $SESSION_NAME"
echo "  2. Run experiments: ./scripts/launch_experiments.sh <table> <dgp>"
