#!/usr/bin/env bash
set -euo pipefail

trial=${1:?usage: run_remote_export_trial.sh TRIAL_NUMBER VARIANT}
variant=${2:?usage: run_remote_export_trial.sh TRIAL_NUMBER VARIANT}
[[ "$trial" =~ ^[0-9]+$ ]] || { echo "invalid trial=$trial" >&2; exit 2; }
[[ "$variant" =~ ^[a-z0-9][a-z0-9-]*$ ]] || {
  echo "invalid variant=$variant" >&2
  exit 2
}

ROOT=/workspace
RUN_CONFIG="$ROOT/heretic-moe/research/runs/adaptive_search_v2/qwen3vl32b_rental_hybrid_600_v1"
EXPORT_ROOT="$ROOT/exports/qwen3vl32b_hybrid_600_v1"
JOB_ROOT="$ROOT/export_jobs/qwen3vl32b_hybrid_600_v1"
output="$EXPORT_ROOT/$variant"
job="$JOB_ROOT/$variant"
log="$ROOT/logs/qwen_export_${variant}.log"

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

if [[ -e "$output" ]]; then
  echo "Refusing to overwrite existing export: $output" >&2
  exit 1
fi
mkdir -p "$EXPORT_ROOT" "$job" "$ROOT/logs"

/opt/conda/bin/python - "$RUN_CONFIG/config.toml" "$job/config.toml" "$trial" "$output" <<'PY'
import pathlib
import sys

source, target, trial, output = sys.argv[1:]
text = pathlib.Path(source).read_text(encoding="utf-8")
text = text.replace("optimization_only = true", "optimization_only = false", 1)
runtime = (
    f"restore_trial_number = {int(trial)}\n"
    'model_action = "save"\n'
    f'save_directory = "{output}"\n'
    'export_strategy = "merge"\n'
)
marker = "[[scorers]]"
if marker not in text:
    raise RuntimeError("Could not find insertion point in base config")
text = text.replace(marker, runtime + "\n" + marker, 1)
pathlib.Path(target).write_text(text, encoding="utf-8", newline="\n")
PY

cd "$job"
echo "OPENAI CODEX | EXPORT TRIAL | trial=$trial variant=$variant"
echo "started=$(date --iso-8601=seconds)"
/opt/conda/bin/heretic 2>&1 | tee "$log"
test -f "$output/config.json"
echo "finished=$(date --iso-8601=seconds) trial=$trial variant=$variant status=PASS"
