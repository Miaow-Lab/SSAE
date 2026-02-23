#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$PYTHONPATH:$PWD"
else
  export PYTHONPATH="$PWD"
fi

CONFIG_PATH="${1:-configs/experiment.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

python experiment/probing_exp.py --config "$CONFIG_PATH" "$@"
