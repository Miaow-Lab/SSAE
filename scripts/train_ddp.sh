#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CONFIG_PATH="${1:-configs/train.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" train.py --config "$CONFIG_PATH" "$@"
