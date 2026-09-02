#!/usr/bin/env bash
set -euo pipefail

# One restartable RunPod job for the complete Arditi-gap experiment.  Re-running this
# command resumes completed grid cells and generation arms.  Override any setting through
# environment variables; the defaults are the paper run.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

models_string="${MODELS:-gemma-3-4b qwen3-4b}"
judge_model="${JUDGE_MODEL:-openrouter/anthropic/claude-haiku-4.5}"
batch_size="${BATCH_SIZE:-8}"
proxy_n="${PROXY_N:-32}"
generation_n="${GENERATION_N:-100}"
max_new_tokens="${MAX_NEW_TOKENS:-512}"
proxy_start="${PROXY_START:-128}"
eval_start="${EVAL_START:-160}"
output_root="${OUTPUT_ROOT:-results/arditi_gap}"

mkdir -p logs output
log_path="logs/arditi_gap_runpod.log"
exec > >(tee -a "$log_path") 2>&1

echo "[runpod] started $(date -Iseconds)"
echo "[runpod] models: $models_string"

judge_args=()
if [[ "${SKIP_LLM_JUDGE:-0}" == "1" ]]; then
  judge_args+=(--skip-llm-judge)
elif [[ -z "${OPENROUTER_API_KEY:-}" && "$judge_model" == openrouter/* ]]; then
  echo "OPENROUTER_API_KEY is required for the default judge."
  echo "Set SKIP_LLM_JUDGE=1 only if you intentionally want lexical refusal scores alone."
  exit 2
fi

smoke_args=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  output_root="${OUTPUT_ROOT:-results/arditi_gap_smoke}"
  proxy_n=8
  generation_n=4
  max_new_tokens=64
  smoke_args+=(--limit-cells 2)
  echo "[runpod] SMOKE=1: two cells, four full generations per population"
fi
echo "[runpod] output: $output_root"

for model in $models_string; do
  echo "[runpod] preparing H1 candidates for $model"
  if [[ ! -f "results/h1/$model/candidates.pt" ]]; then
    h1_args=(--model "$model" --n 64 --n-val 32 --dtype float32)
    if [[ "$model" == "gemma-3-4b" ]]; then
      h1_args+=(--reconcile)
    fi
    uv run python experiments/h1_direction/run.py "${h1_args[@]}"
  else
    echo "[runpod] results/h1/$model/candidates.pt exists; reusing it"
  fi

  echo "[runpod] running full Arditi gap experiment for $model"
  uv run python experiments/arditi_gap/run.py \
    --model "$model" \
    --stage all \
    --out "$output_root" \
    --proxy-start "$proxy_start" \
    --eval-start "$eval_start" \
    --n-proxy "$proxy_n" \
    --n-generation "$generation_n" \
    --batch-size "$batch_size" \
    --max-new-tokens "$max_new_tokens" \
    --judge-model "$judge_model" \
    "${judge_args[@]}" \
    "${smoke_args[@]}"
done

bundle_path="output/arditi_gap_runpod_results.tar.gz"
tar -czf "$bundle_path" "$output_root" logs/arditi_gap_runpod.log
echo "[runpod] finished $(date -Iseconds)"
echo "[runpod] bundle: $bundle_path"
