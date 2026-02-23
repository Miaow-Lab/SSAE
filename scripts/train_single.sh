#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${1:-configs/train.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

python train.py --config "$CONFIG_PATH" "$@"
