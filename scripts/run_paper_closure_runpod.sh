#!/usr/bin/env bash
set -euo pipefail

# One resumable job for the remaining workshop-paper causal gaps. Environment variables
# override every expensive setting; defaults are the full paper run.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

models_string="${MODELS:-gemma-3-4b qwen3-4b}"
stages="${STAGES:-all}"
judge_model="${JUDGE_MODEL:-openrouter/anthropic/claude-haiku-4.5}"
batch_size="${BATCH_SIZE:-8}"
max_new_tokens="${MAX_NEW_TOKENS:-1024}"
n_transplant="${N_TRANSPLANT:-100}"
min_class_n="${MIN_CLASS_N:-10}"
scope_limit="${SCOPE_LIMIT_PER_CLASS:-}"
restoration_limit="${RESTORATION_LIMIT_PER_CLASS:-}"
classes="${CLASSES:-auto}"
output_root="${OUTPUT_ROOT:-results/paper_closure}"

mkdir -p logs output
log_path="logs/paper_closure_runpod.log"
exec > >(tee -a "$log_path") 2>&1

echo "[closure-runpod] started $(date -Iseconds)"
echo "[closure-runpod] models: $models_string"
echo "[closure-runpod] stages: $stages"

if [[ "$stages" != "analyze" && -z "${OPENROUTER_API_KEY:-}" && "$judge_model" == openrouter/* ]]; then
  echo "OPENROUTER_API_KEY is required for generation stages."
  exit 2
fi

if [[ "${SMOKE:-0}" == "1" ]]; then
  output_root="${OUTPUT_ROOT:-results/paper_closure_smoke}"
  max_new_tokens=64
  n_transplant=4
  min_class_n=1
  classes="fiction,refusal_suppression_long"
  scope_limit=2
  restoration_limit=2
  echo "[closure-runpod] SMOKE=1: two classes, two scope/restoration prompts, four transplant prompts"
fi

mkdir -p "$output_root"
echo "[closure-runpod] output: $output_root"

extra_args=()
if [[ -n "$scope_limit" ]]; then
  extra_args+=(--scope-limit-per-class "$scope_limit")
fi
if [[ -n "$restoration_limit" ]]; then
  extra_args+=(--restoration-limit-per-class "$restoration_limit")
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  extra_args+=(--overwrite)
fi

git rev-parse HEAD > "$output_root/GIT_COMMIT"
bundle_path="output/paper_closure_runpod_results.tar.gz"

bundle_on_exit() {
  status=$?
  trap - EXIT
  tar -czf "$bundle_path" "$output_root" "$log_path" || true
  if [[ $status -ne 0 ]]; then
    echo "[closure-runpod] interrupted/failed with status $status; partial bundle: $bundle_path"
  fi
  exit "$status"
}
trap bundle_on_exit EXIT

for model in $models_string; do
  if [[ ! -f "results/h1/$model/candidates.pt" ]]; then
    echo "[closure-runpod] preparing H1 candidates for $model"
    h1_args=(--model "$model" --n 64 --n-val 32 --dtype float32)
    if [[ "$model" == "gemma-3-4b" ]]; then
      h1_args+=(--reconcile)
    fi
    uv run python experiments/h1_direction/run.py "${h1_args[@]}"
  fi

  echo "[closure-runpod] running $model"
  uv run python experiments/paper_closure/run.py \
    --model "$model" \
    --stage "$stages" \
    --out "$output_root" \
    --classes "$classes" \
    --min-class-n "$min_class_n" \
    --n-transplant "$n_transplant" \
    --max-new-tokens "$max_new_tokens" \
    --batch-size "$batch_size" \
    --judge-model "$judge_model" \
    "${extra_args[@]}"
done

trap - EXIT
tar -czf "$bundle_path" "$output_root" "$log_path"
echo "[closure-runpod] finished $(date -Iseconds)"
echo "[closure-runpod] bundle: $bundle_path"
