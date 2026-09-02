# Refusal Directions as Causal Controls for Jailbreaks

This anonymous repository is the artifact for our workshop submission. It contains the code, public input data, aggregate results, label-only behavioral records, paper source, and figure builder needed to inspect and reproduce the paper.

Our experiments separate three questions that are easy to conflate:

1. **Natural-state test:** Do successful jailbreaks naturally move the model to a shared low-refusal state?
2. **Removal test:** Does removing the refusal direction causally weaken refusal?
3. **Measured-state test:** Does transferring the refusal-direction value measured on one prompt reproduce that prompt's behavior?

The strongest result is the second. Sustained removal of the refusal direction increases compliance on harmful requests in both models and is much stronger than matched random-direction controls. The first and third tests do not support a universal scalar mediation account: successful jailbreak classes have different activation trajectories, and class-average measured-state transfers do not consistently reproduce jailbreak or refusal behavior.

## Start here

For a CPU-only integrity check and exact regeneration of every paper figure from the included public artifacts:

```bash
uv sync --extra plots --extra dev
uv run python scripts/check_artifact.py --rebuild-figures
```

The checker validates required files, expected model coverage, the absence of response text from label-only exports, and the paper's figure dependencies. It then runs the figure builder. Expected final output:

```text
Artifact checks passed.
Rebuilt 16 paper figures.
```

The paper source and generated figures are in [`paper/workshop_2026/`](paper/workshop_2026/). See [`docs/ARTIFACT_MANIFEST.md`](docs/ARTIFACT_MANIFEST.md) for the exact mapping from paper tests and figures to scripts and data.

## Repository map

```text
data/
  refusal_direction/       Public harmful/harmless train and validation splits
  jailbreaks/              Seven-class paired jailbreak dataset
experiments/
  h1_direction/            Fit and validate refusal directions
  h2_jailbreak/            Generate and judge attacks plus matched controls
  f1_geometry/             Measure across-layer refusal-direction trajectories
  f1_categories/           Summarize trajectories by jailbreak class
  arditi_gap/              Full layer-by-position causal screen and validation
  h3_causal/               Removal and strong addition experiments
  paper_closure/            Scope comparison and measured-state counterfactuals
results/
  <experiment>/<model>/     Aggregate reports used by the paper
  public_labels/            Per-example conditions and labels, without text
paper/workshop_2026/        Anonymous LaTeX source, figures, and figure builder
scripts/                    Reproduction, export, and artifact-check entry points
src/jbct/                   Shared model, direction, intervention, and judge code
test/                       Unit tests for the experiment-critical implementation
```

## Models and compute

The paper evaluates:

- `google/gemma-3-4b-it`, referred to as `gemma-3-4b`
- `Qwen/Qwen3-4B`, referred to as `qwen3-4b`

The original runs recorded the repository identifiers but resolved the Hugging Face `main` revisions rather than immutable model commit hashes. This is a reproducibility limitation, stated in the paper as well. The included reports and label exports freeze the reported observations, but a future upstream model update could change a fresh run. Reviewers reproducing the artifact should record the resolved snapshot hashes in their run log.

Full reproduction requires Linux, a CUDA-capable GPU, access to the two Hugging Face checkpoints, and an OpenRouter-compatible judge key. The experiments were designed to fit on a 48 GB GPU. An A100 80 GB or H100 also provides ample memory. The complete layer-by-position sweep is the longest stage.

Set credentials without committing them:

```bash
export HF_TOKEN=...
export OPENROUTER_API_KEY=...
```

Install all experiment dependencies:

```bash
uv sync --extra steering --extra judge --extra plots --extra dev
```

Run the complete pipeline into the ignored `reproduced/` directory:

```bash
bash scripts/reproduce_paper.sh
```

You can reproduce one model or one stage:

```bash
MODELS=gemma-3-4b bash scripts/reproduce_paper.sh
STAGES=h1,h2,f1,h3 bash scripts/reproduce_paper.sh
```

The stages are ordered because later experiments consume full judged generations and fitted directions from earlier stages. See [`docs/REPRODUCING.md`](docs/REPRODUCING.md) for commands, expected outputs, restart behavior, and estimated scope.

## Experiment terminology

- **Anchor position:** one token position near the end of the formatted request where a refusal direction is estimated or applied. Negative indices count backward from the final request token.
- **Request-wide removal:** remove the direction at every token position while the model reads the request, then stop intervening during generation.
- **Sustained removal:** remove the direction while reading the request and again at every generated token.
- **Comparison direction:** an equal-norm random direction orthogonal to the refusal direction. It controls for an activation edit of the same size.
- **Comparison layer:** a layer selected in advance where the same edit is not expected to have the primary causal effect.
- **Measured-state transfer:** set the refusal-direction component to a value measured from successful attacks or refusing matched controls, rather than adding a large arbitrary amount.
- **Strong addition:** add a chosen multiple of the fitted direction during generation. This tests whether the direction is a useful causal control, not whether the jailbreak naturally made an equal-size change.

The seven jailbreak transformations are defined in the paper and in [`data/README.md`](data/README.md). In particular, **prompted prefill** means the template begins the assistant answer with compliant-looking text; it does not denote an intervention performed by our code.

## Public data policy

The paired jailbreak prompts and refusal-direction corpora are included because they are required inputs. Raw model completions are not distributed: they can contain operationally harmful content, are unnecessary for regenerating the reported statistics, and create an avoidable release risk. Instead, [`results/public_labels/`](results/public_labels/) contains the condition, example identifier, judge label, coherence label, and token-count metadata for each behavioral trial. Aggregate JSON reports contain the values consumed by the paper figures and tables.

Fresh runs write full completions under `reproduced/`, which is ignored by Git. To create a shareable label-only export from a local run, use:

```bash
uv run python scripts/export_public_labels.py \
  --source-root reproduced \
  --output-root reproduced/public_labels
```

## Tests

Run the CPU test suite with:

```bash
uv run pytest
```

GPU tests are marked `gpu` and are excluded by default. Run them explicitly on a configured GPU with `uv run pytest -m gpu`.

## Anonymity

This branch is an orphan snapshot with no parent commits. It contains no author names, private machine paths, remote URLs, or raw completion files. Please preserve that property when creating the public review repository. Git hosting metadata, repository ownership, and release links must also be anonymized outside this codebase.
