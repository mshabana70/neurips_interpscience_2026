# Reproducing the experiments

## 1. Environment

Use Python 3.12 on Linux with CUDA. Install the locked environment:

```bash
uv sync --extra steering --extra judge --extra plots --extra dev
```

Provide Hugging Face access to Gemma and an OpenRouter-compatible judge key:

```bash
export HF_TOKEN=...
export OPENROUTER_API_KEY=...
```

No credential is read from a repository file. Generation is greedy unless a runner explicitly records a different decoding policy. Each JSON report records the model alias, key arguments, and available package/environment provenance.

The source runs used the Hugging Face `main` revisions and did not retain the resolved immutable snapshot hashes. This prevents a guarantee of bit-for-bit model identity if an upstream repository changes. Preserve the resolved model and tokenizer snapshot hashes when repeating the experiment; the checked-in aggregate reports remain the record of the paper runs.

## 2. CPU-only artifact reproduction

The following command validates and rebuilds all paper figures from public inputs:

```bash
uv run python scripts/check_artifact.py --rebuild-figures
```

To build only the figures:

```bash
uv run python paper/workshop_2026/build_figures.py
```

To compile the paper after building figures:

```bash
cd paper/workshop_2026
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## 3. Full GPU reproduction

Run all stages for both models:

```bash
bash scripts/reproduce_paper.sh
```

By default, outputs go to `reproduced/`, which is ignored because intermediate files contain full model completions. Useful overrides are:

```bash
MODELS=qwen3-4b bash scripts/reproduce_paper.sh
STAGES=h1,h2,f1 bash scripts/reproduce_paper.sh
RUN_ROOT=/workspace/reproduced bash scripts/reproduce_paper.sh
BATCH_SIZE=4 bash scripts/reproduce_paper.sh
```

The valid stage names are `h1`, `h2`, `f1`, `f1_categories`, `arditi`, `h3`, `closure`, `export`, and `figures`. Dependencies are not inferred when a subset is requested, so run their prerequisites first or retain prior outputs under the same `RUN_ROOT`.

### Stage dependencies

```text
h1 ---------------------> arditi
 |                         |
 +-----------> h3 <--------h2
 |                         |
 +-----------> closure <---f1 <--- h2
                              \
                               +--> f1_categories
```

H2 writes `generations.json` before calling the external judge. If judging fails after generation, resume without loading the language model:

```bash
uv run python experiments/h2_jailbreak/run.py \
  --model gemma-3-4b \
  --out reproduced/h2 \
  --judge-existing
```

The Arditi-gap and closure runners checkpoint individual conditions. Rerunning the same command reuses complete condition files unless `--force` or `--overwrite` is supplied.

## 4. Independent experiment commands

Every runner documents its full command line:

```bash
uv run python experiments/h1_direction/run.py --help
uv run python experiments/h2_jailbreak/run.py --help
uv run python experiments/f1_geometry/run.py --help
uv run python experiments/f1_categories/run.py --help
uv run python experiments/arditi_gap/run.py --help
uv run python experiments/h3_causal/run.py --help
uv run python experiments/paper_closure/run.py --help
```

The master script is intentionally plain shell and prints each stage before execution, making it suitable for RunPod logs and easy to modify for a scheduler.

## 5. Verification

After a full run, export safe per-example labels and inspect aggregate reports:

```bash
uv run python scripts/export_public_labels.py \
  --source-root reproduced \
  --output-root reproduced/public_labels
uv run pytest
```

Do not publish `reproduced/h2/*/judged.json` or the raw per-condition generation files. They include full model completions. The exporter removes every prompt, prefill, and response field recursively.
