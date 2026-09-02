# Arditi-gap RunPod experiment

This experiment closes the two foundational gaps that cannot be filled from the committed
artifacts alone:

1. a behavioral layer x post-instruction-position sweep, rather than the existing AUC grid;
2. activation addition on held-out harmless prompts, the sufficiency half of Arditi et al.

It also distinguishes two ablation scopes that must not be conflated:

- **Arditi all-write ablation:** removes the candidate direction from every decoder-block
  input and every attention/MLP output. This matches the released Arditi hooks.
- **Source-local block-output ablation:** removes the candidate at its source layer across
  the full token-position axis. This matches the intervention family used by the existing
  H3 results and localizes the causal site.

## RunPod setup

Use a CUDA GPU with at least 48 GB of memory. The paper run was validated on a 48 GB
RTX PRO 6000 partition; A100 80 GB and H100 GPUs also provide ample headroom. From a
fresh clone:

```bash
uv sync --extra steering --extra judge --extra dev
export HF_TOKEN=...            # Gemma access
export OPENROUTER_API_KEY=...  # harmful-completion refusal judge
```

Run a two-cell end-to-end smoke test first. It writes to a separate output directory and
does not contaminate the paper run:

```bash
SMOKE=1 bash scripts/run_arditi_gap_runpod.sh
```

Then launch the complete two-model job:

```bash
bash scripts/run_arditi_gap_runpod.sh
```

The command is restartable. It regenerates the gitignored H1 candidate tensors when absent,
checkpoints every heatmap cell, and checkpoints every generation arm. Re-running it skips
completed work. A load-bearing configuration change fails rather than mixing incompatible
cells; use a new `OUTPUT_ROOT` for a genuinely different run.

Useful overrides:

```bash
BATCH_SIZE=16 bash scripts/run_arditi_gap_runpod.sh
MODELS="gemma-3-4b" bash scripts/run_arditi_gap_runpod.sh
SKIP_LLM_JUDGE=1 bash scripts/run_arditi_gap_runpod.sh  # lexical metrics only
```

## Design

### Data independence

- H1 direction fit: first 64 harmful and 64 harmless training prompts.
- H1 representational validation: the committed validation split.
- Causal proxy selection: training indices `[128:160]` (32 per population).
- Full-generation validation: training indices `[160:260]` (100 per population).

The proxy-selection and generation sets are disjoint from each other and from the direction
fit. The committed harmful corpus has exactly enough records for the default 100-example
generation slice; changing `EVAL_START` upward requires reducing `GENERATION_N`.

### Complete causal grid

For every H1 candidate `(source layer, template-relative source position)`, the run records:

- harmful next-token refusal log-odds after source-local block-output ablation;
- harmful next-token refusal log-odds after Arditi all-write ablation;
- harmless next-token refusal log-odds after adding one difference-in-means vector at the
  source layer, across all token positions;
- harmless-prompt KL divergence under both ablation scopes;
- standard errors and per-prompt values for every cell.

The refusal proxy is Arditi's efficient metric: log-odds of the summed next-token
probability assigned to characteristic refusal-start tokens. Token ids are derived from
refusal phrases using each model's tokenizer and are written to the report. Baseline prompts
are filtered as in Arditi when the derived token set yields enough examples; otherwise the
filter is disabled loudly and the reason is recorded.

The exhaustive grid evaluates the real direction. Matched random-direction controls are run
with full generations at the selected sites; running them at every cell would approximately
double the largest stage without changing direction selection.

### Full-generation validation

Three sites are validated and deduplicated if they coincide:

- the Arditi-style all-write selection;
- the best source-local causal site;
- the registry site used by existing experiments.

For each site, 512-token greedy generations cover real and matched-random local ablation,
real and matched-random all-write ablation, and real and matched-random activation addition.
Harmful prompts receive both the repository's LLM refusal judgment and Arditi-style lexical
refusal scoring. Harmless addition uses lexical refusal scoring because the harmful-payload
judge is not designed to classify refusals of benign requests. Every arm reports incoherence
and token-cap counts.

## Outputs

```text
results/arditi_gap/<model>/
  run_config.json
  proxy_cells/*.json
  proxy_grid_report.json
  causal_heatmaps.png
  generations/*.json.gz
  generation_validation_report.json
  generation_validation.png

output/arditi_gap_runpod_results.tar.gz
logs/arditi_gap_runpod.log
```

The final tarball is the file to download from RunPod. The H1 tensors are deliberately not
included because they are large, gitignored, and reproducible from the committed corpus.

## Acceptance checks before quoting results

1. `n_cells` is 102 for Gemma and 108 for Qwen.
2. Inspect the recorded refusal token strings. They should correspond to openings such as
   `I` and `As`; a bizarre token set invalidates the proxy grid.
3. The 32-prompt baseline harmful and harmless refusal metrics should separate. If the
   automatic filter was disabled, explain why before using the selected site.
4. The selected site's addition arm should outperform its matched random addition without
   a large incoherence increase.
5. Treat all-write ablation as uninterpretable if the real and random arms both become
   incoherent. The source-local result remains separately interpretable.
6. Do not call the proxy grid a compliance heatmap. It is a causal refusal-log-odds heatmap;
   the selected cells receive the full behavioral validation.

After downloading the bundle, copy `results/arditi_gap/` into the repository and rebuild the
foundations report. The generated heatmap and validation figures are already publication
drafts, but their captions must be written from the observed results rather than templated
in advance.
