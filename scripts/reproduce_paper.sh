#!/usr/bin/env bash
# End-to-end GPU reproduction for the anonymous workshop artifact.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODELS="${MODELS:-gemma-3-4b qwen3-4b}"
STAGES="${STAGES:-h1,h2,f1,f1_categories,arditi,h3,closure,export,figures}"
RUN_ROOT="${RUN_ROOT:-reproduced}"
RESULTS_ROOT="$RUN_ROOT/results"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
JUDGE_MODEL="${JUDGE_MODEL:-openrouter/anthropic/claude-haiku-4.5}"

has_stage() {
  [[ ",$STAGES," == *",$1,"* ]]
}

run_python() {
  uv run python "$@"
}

announce() {
  printf '\n[reproduce] %s\n' "$1"
}

mkdir -p "$RESULTS_ROOT"

for model in $MODELS; do
  if has_stage h1; then
    announce "$model: fit and validate refusal directions"
    run_python experiments/h1_direction/run.py \
      --model "$model" \
      --out "$RESULTS_ROOT/h1"
  fi

  if has_stage h2; then
    announce "$model: generate and judge paired jailbreaks"
    run_python experiments/h2_jailbreak/run.py \
      --model "$model" \
      --dataset data/jailbreaks/jailbreak_dataset.json \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --batch-size "$BATCH_SIZE" \
      --judge-model "$JUDGE_MODEL" \
      --out "$RESULTS_ROOT/h2"
  fi

  if has_stage f1; then
    announce "$model: measure across-layer refusal-direction geometry"
    run_python experiments/f1_geometry/run.py \
      --model "$model" \
      --h2 "$RESULTS_ROOT/h2" \
      --out "$RESULTS_ROOT/f1_geometry"
  fi

  if has_stage f1_categories; then
    announce "$model: summarize natural trajectories by jailbreak class"
    run_python experiments/f1_categories/run.py \
      --model "$model" \
      --f1 "$RESULTS_ROOT/f1_geometry" \
      --h2 "$RESULTS_ROOT/h2" \
      --plot-min-n 18 \
      --out "$RESULTS_ROOT/f1_categories"
  fi

  if has_stage arditi; then
    announce "$model: run the complete layer-by-position causal screen"
    run_python experiments/arditi_gap/run.py \
      --model "$model" \
      --stage all \
      --h1 "$RESULTS_ROOT/h1" \
      --out "$RESULTS_ROOT/arditi_gap" \
      --n-proxy 32 \
      --n-generation 100 \
      --max-new-tokens 512 \
      --batch-size "$BATCH_SIZE" \
      --judge-model "$JUDGE_MODEL"
  fi

  if has_stage h3; then
    announce "$model: run sustained removal and strong addition"
    run_python experiments/h3_causal/run.py \
      --model "$model" \
      --arm all \
      --h1 "$RESULTS_ROOT/h1" \
      --h2 "$RESULTS_ROOT/h2" \
      --coeff-sweep 0.25,0.5,1 \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --batch-size "$BATCH_SIZE" \
      --judge-model "$JUDGE_MODEL" \
      --out "$RESULTS_ROOT/h3_causal"
  fi

  if has_stage closure; then
    announce "$model: run scope and measured-state counterfactuals"
    run_python experiments/paper_closure/run.py \
      --model "$model" \
      --stage all \
      --h1 "$RESULTS_ROOT/h1" \
      --h2 "$RESULTS_ROOT/h2" \
      --f1-projections "$RESULTS_ROOT/f1_geometry" \
      --out "$RESULTS_ROOT/paper_closure" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --batch-size "$BATCH_SIZE" \
      --judge-model "$JUDGE_MODEL"
  fi
done

if has_stage export; then
  announce "export text-free per-example labels"
  run_python scripts/export_public_labels.py \
    --source-root "$RUN_ROOT" \
    --output-root "$RESULTS_ROOT/public_labels"
fi

if has_stage figures; then
  announce "rebuild figures from the included public aggregate artifacts"
  run_python paper/workshop_2026/build_figures.py
fi

announce "complete; fresh experimental results are under $RESULTS_ROOT"
