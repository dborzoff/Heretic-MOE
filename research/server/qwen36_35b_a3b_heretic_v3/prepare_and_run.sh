#!/usr/bin/env bash
set -Eeuo pipefail

BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY="$(cd -- "$BUNDLE_DIR/../../.." && pwd)"
MODEL_ID="Qwen/Qwen3.6-35B-A3B"
MODEL_REVISION="995ad96eacd98c81ed38be0c5b274b04031597b0"
MODEL_DIR="${MODEL_DIR:-/workspace/models/Qwen3.6-35B-A3B}"
DATA_ROOT="${DATA_ROOT:-/workspace/heretic-data/qwen36-v3}"
RUN_ROOT="${RUN_ROOT:-/workspace/heretic-runs/qwen36-35b-a3b-v3}"
HF_HOME="${HF_HOME:-/workspace/hf-cache}"
UV_VERSION="${UV_VERSION:-0.7.14}"

: "${HF_TOKEN:?Set HF_TOKEN without printing it before starting this script}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This entrypoint requires Linux." >&2
  exit 2
fi
if [[ -n "$(git -C "$REPOSITORY" status --porcelain --untracked-files=no)" ]]; then
  echo "Refusing to run from a modified Heretic-MOE checkout." >&2
  exit 2
fi

mkdir -p "$MODEL_DIR" "$DATA_ROOT" "$RUN_ROOT" "$HF_HOME"
export HF_HOME HF_TOKEN HF_XET_HIGH_PERFORMANCE=1 PYTHONUNBUFFERED=1

if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
  SUDO=()
elif command -v sudo >/dev/null 2>&1; then
  SUDO=(sudo)
else
  echo "Root or sudo is required to install system prerequisites." >&2
  exit 2
fi

"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y ca-certificates curl git python3

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
UV_BIN="$(command -v uv)"
echo "[BOOT] uv=$($UV_BIN --version) git=$(git -C "$REPOSITORY" rev-parse HEAD)"

setup_environment() {
  cd -- "$REPOSITORY"
  "$UV_BIN" sync --frozen
}

download_model() {
  "$UV_BIN" tool run --from "huggingface_hub==1.24.0" hf download \
    "$MODEL_ID" \
    --revision "$MODEL_REVISION" \
    --local-dir "$MODEL_DIR"
  cat >"$MODEL_DIR/.heretic_download.json" <<EOF
{"model_id":"$MODEL_ID","revision":"$MODEL_REVISION"}
EOF
}

run_prefixed() {
  local label="$1"
  shift
  set -o pipefail
  "$@" 2>&1 | sed -u "s/^/[$label] /"
}

run_prefixed ENV setup_environment &
environment_pid=$!
run_prefixed MODEL download_model &
model_pid=$!

status=0
wait "$environment_pid" || status=1
wait "$model_pid" || status=1
if [[ "$status" -ne 0 ]]; then
  echo "Parallel preparation failed; inspect the prefixed output above." >&2
  exit "$status"
fi

"$REPOSITORY/.venv/bin/python" "$BUNDLE_DIR/verify_ready.py" \
  --model "$MODEL_DIR" \
  --data "$DATA_ROOT" \
  --repository "$REPOSITORY" \
  --output "$RUN_ROOT/preflight.json" \
  --expected-gpus 2 \
  --min-vram-gib 90

export MODEL_DIR DATA_ROOT RUN_ROOT HF_HOME
exec "$BUNDLE_DIR/run_search.sh"
