# Workshop paper closure run

This experiment packages the remaining paper-changing causal gaps into one restartable
RunPod job for Gemma-3-4B and Qwen3-4B.

## What it runs

### 1. Temporal and positional scope

All nine arms use the same 350 matched harmful controls (50 per dataset class) and the
same generation/judge configuration:

1. untreated baseline;
2. causal-layer anchor-token removal and matched orthogonal-shift sham;
3. causal-layer removal at every prompt position and matched sham;
4. causal-layer removal at every prompt and generated position and matched sham; and
5. sustained removal at the pre-existing off-layer site and matched sham.

The sham reads the shift required along the real refusal direction but writes that exact
shift along a seeded orthogonal direction. It therefore matches intervention magnitude
more tightly than independently ablating a random component.

### 2. Per-class profile transplant

For every non-prefill class with at least `MIN_CLASS_N` successful attacks and refusing
controls, the run computes a full per-layer class profile from the committed complement-
basis projections. It then applies the successful-attack profile to 100 held-out harmful
prompts and compares it with:

- that class's refusing-control profile (`null`);
- a matched orthogonal-shift sham; and
- shared untreated and zero-profile reference arms.

### 3. Exact harmful-profile restoration

For each eligible class, the run starts from H2 attacks that actually succeeded and clamps
their prompt-side, full per-layer profile to the mean of refusing controls from that class.
It compares against an untreated rerun, an attack-profile null clamp, and a matched sham.

Prefill is intentionally excluded from both profile stages. Its successful state is
measured after an assistant prefix, so a negative token position does not denote the same
prompt-side location as it does for the other classes.

## Statistics and outputs

Every comparison is paired by prompt. The report includes compliance rates, paired
differences, deterministic paired-bootstrap 95% intervals, discordant counts, exact
McNemar/sign-test p-values, and Holm-adjusted p-values within each comparison family.

Outputs for each model:

```text
results/paper_closure/<model>/
  scope/*.json.gz
  transplant/*.json.gz
  restoration/*.json.gz
  paper_closure_report.json
  paper_closure_summary.csv
```

Generation artifacts are compressed and each arm is resumable. Re-running the launcher
reuses an existing arm only when its ordered `pair_id` population matches exactly.
The launcher also writes a partial results bundle if a model or judge call fails, so a
late interruption does not strand completed arms on the pod.

## RunPod launch

Smoke test:

```bash
SMOKE=1 bash scripts/run_paper_closure_runpod.sh
```

Full paper run:

```bash
bash scripts/run_paper_closure_runpod.sh
```

The default full run uses 1024 generated tokens, 50 scope examples per class, 100 held-out
transplant prompts, and every eligible successful attack for restoration. It is deliberately
large: roughly ten thousand generations across both models before early stopping. Plan for
an overnight 4090-class run and monitor OpenRouter quota.

Useful overrides:

```bash
MODELS="gemma-3-4b" STAGES="scope" bash scripts/run_paper_closure_runpod.sh
N_TRANSPLANT=64 MIN_CLASS_N=15 bash scripts/run_paper_closure_runpod.sh
OVERWRITE=1 bash scripts/run_paper_closure_runpod.sh
```

`OPENROUTER_API_KEY` is required for generation stages. `STAGES=analyze` reconstructs
statistics from finished compressed arms without loading model weights or contacting the
judge.

## Interpretation guardrails

- The scope stage answers anchor-only versus prompt-wide versus sustained intervention.
- The off-layer arm is a specificity check, not a search over layers.
- Profile targets are population means, not prompt-matched counterfactual values.
- Restoration targets are fitted on the same H2 class population being interrogated;
  inference is causal for that population but does not establish target generalization.
- A null profile or sham that becomes incoherent invalidates the corresponding comparison.
- Attribution-graph exploration is not part of this batch. It remains a human inspection
  task for generating hypotheses from the class contrast revealed here.
