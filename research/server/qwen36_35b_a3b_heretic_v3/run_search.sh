#!/usr/bin/env bash
set -Eeuo pipefail

BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY="$(cd -- "$BUNDLE_DIR/../../.." && pwd)"
MODEL_DIR="${MODEL_DIR:-/workspace/models/Qwen3.6-35B-A3B}"
DATA_ROOT="${DATA_ROOT:-/workspace/heretic-data/qwen36-v3}"
RUN_ROOT="${RUN_ROOT:-/workspace/heretic-runs/qwen36-35b-a3b-v3}"
TARGET_TRIALS="${TARGET_TRIALS:-600}"
EXPLORATION_TRIALS="${EXPLORATION_TRIALS:-120}"
DEVICES="${DEVICES:-0,1}"

PYTHON="$REPOSITORY/.venv/bin/python"
HERETIC="$REPOSITORY/.venv/bin/hereticMOE"
CONTINUE_ARGS=()
if [[ -f "$RUN_ROOT/shared_tpe/config.toml" ]]; then
  CONTINUE_ARGS+=(--continue-shared-only)
fi

export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

exec "$PYTHON" "$REPOSITORY/research/scripts/run_adaptive_search.py" \
  --base-config "$BUNDLE_DIR/qwen36_sparse_geometry.toml" \
  --model "$MODEL_DIR" \
  --data-root "$DATA_ROOT" \
  --run-root "$RUN_ROOT" \
  --heretic "$HERETIC" \
  --devices "$DEVICES" \
  --exploration-trials "$EXPLORATION_TRIALS" \
  --target-trials "$TARGET_TRIALS" \
  --finalist-top-n 6 \
  --finalist-selection-policy feasible_diverse \
  --recheck-ppl-chunks 64 \
  --recheck-ppl-window 1024 \
  --max-ppl-drift 0.005 \
  --max-keywords 2 \
  --recheck-only \
  "${CONTINUE_ARGS[@]}"
