"""H1 primitive: diff-in-means refusal direction (Arditi 2024), per layer.

Two entry points:
- ``compute_directions``: one direction per layer at the model's canonical position
  (``spec.direction_position``). This is the Refusal-Lens "single position" behavior
  (Gemma used pos=-2, the "model" token — NOT the last token).
- ``compute_direction_candidates``: the Arditi grid — a direction at each
  (layer, post-instruction position) in ``positions`` (default ``spec.direction_positions``),
  yielding |positions|x|layers| candidate vectors for a downstream selection step.

Load-bearing (MENTEE_NOTE Bug 3): read the post-block residual (captured by HookManager as
the raw layer output) at the chosen prompt position; accumulate means in float64. Bug 3 was
extracting r at the residual stream but *applying* it post-RMSNorm pre-MLP — a ~1700x
magnitude mismatch across a non-linear normalization. Whatever consumes these directions
(H3 steering, H4 attribution `measurement_layer`) must inject at the same point they were
read from: the block output.

**Last-layer convention.** These are RAW residuals at every layer. Refusal-Lens read
`hidden_states[L+1]`, which HF post-RMSNorms for the FINAL layer only — which is why its
Gemma L33 separation reads 287 while L32 reads 20,873 (`cos(L32,L33)=+0.83`: a magnitude
collapse, not a rotation). Our L33 is the un-normed value, so it is NOT comparable to that
287. Middle layers — including every verified Gemma layer (causal 15, separation 32) — are
identical under both conventions.
"""

from __future__ import annotations

import torch

from .models import HookManager, count_leading_bos, get_layers, tokenize_prompt
from .registry import ModelSpec


def _mean_resids_at_positions(
    model,
    tokenizer,
    spec: ModelSpec,
    prompts: list[str],
    positions: list[int],
    trace: dict | None = None,
) -> dict[tuple[int, int], torch.Tensor]:
    """Mean residual at every (layer, position), in ONE forward pass per prompt.

    Positions are negative indices; callers must ensure each prompt is long enough
    (chat-formatted prompts always carry the post-instruction template suffix).

    If ``trace`` is given, records the (layer, position) indices ACTUALLY read — the
    R6a hook that proves reality matches the spec, not just the recorded metadata.
    """
    n_layers = len(get_layers(model, spec))
    keys = [(layer, pos) for layer in range(n_layers) for pos in positions]
    sums: dict[tuple[int, int], torch.Tensor | None] = dict.fromkeys(keys)
    read: set[tuple[int, int]] = set()
    n_bos = 0
    for text in prompts:
        ids = tokenize_prompt(tokenizer, text, device=model.device)
        n_bos = max(n_bos, count_leading_bos(tokenizer, ids["input_ids"]))
        with HookManager(model, spec) as hm:
            model(**ids)
        for layer in range(n_layers):
            for pos in positions:
                v = hm.residuals[layer][0, pos].to(torch.float64)  # raw residual, float64
                key = (layer, pos)
                sums[key] = v.clone() if sums[key] is None else sums[key] + v
                if trace is not None:
                    read.add(key)
    if trace is not None:
        trace["layers"] = sorted({layer for (layer, _pos) in read})
        positions_read = sorted({pos for (_layer, pos) in read})
        trace["positions"] = positions_read
        trace["position"] = positions_read[0] if len(positions_read) == 1 else None
        # Provenance for the two things that silently change a direction's numbers:
        # the model dtype it was read in, and whether the input carried a duplicate BOS.
        trace["dtype"] = str(getattr(model, "dtype", None))
        trace["n_leading_bos"] = n_bos
    k = len(prompts)
    return {key: sums[key] / k for key in keys}


def compute_directions(
    model,
    tokenizer,
    spec: ModelSpec,
    harmful: list[str],
    harmless: list[str],
    trace: dict | None = None,
) -> dict[int, torch.Tensor]:
    """Per-layer ``r = mean(harmful) - mean(harmless)`` at ``spec.direction_position``.

    Pass ``trace`` (a dict) to record the (layer, position) actually read (R6a).
    """
    pos = spec.direction_position.value
    hf = _mean_resids_at_positions(model, tokenizer, spec, harmful, [pos], trace=trace)
    hl = _mean_resids_at_positions(model, tokenizer, spec, harmless, [pos])
    n_layers = len(get_layers(model, spec))
    return {layer: hf[(layer, pos)] - hl[(layer, pos)] for layer in range(n_layers)}


def drop_dims(r, dims):
    """Zero the given residual dimensions and renormalize.

    For Gemma-3 the L15 diff-in-means direction is **90% dimension 443** (81% of its squared
    norm on one coordinate, and rank-1 at every layer) — that is the model's massive-activation
    outlier, not refusal. Two consequences measured directly from the reference directions:

      - Ablating the raw direction at every layer zeroes that coordinate across the whole
        residual stream and collapses the model to a single repeated token.
      - Layer geometry is nonsense while it is present: cos(L10, L15) = -0.816 raw, but
        **+0.401** with 443 removed. A near-antiparallel cosine between nearby layers was the
        outlier's sign flipping, not the refusal direction rotating.

    Refusal-Lens reached the same conclusion from the other end (the "complement" is the real
    circuit; the outlier is inert). Prefer the complement for anything causal.
    """
    import torch

    out = r.clone()
    for d in dims:
        out[..., d] = 0.0
    n = torch.linalg.vector_norm(out)
    return out / n if n > 0 else out


def detect_outlier_dims(acts, factor: float = 20.0) -> tuple[int, ...]:
    """Dimensions whose mean |activation| exceeds `factor` x the median dimension.

    Model-agnostic replacement for hardcoding: pass a (n, d_model) tensor of residuals.
    Gemma's 443 is found this way; other models' outliers are simply not measured yet, which
    is why `ModelSpec.outlier_dims` defaults to an EMPTY tuple rather than to "none".
    """
    import torch

    mag = acts.abs().to(torch.float64).mean(dim=0)
    med = float(mag.median())
    if med <= 0:
        return ()
    return tuple(int(i) for i in torch.nonzero(mag > factor * med).flatten())


def normalize_directions(
    directions: dict[int, torch.Tensor],
) -> tuple[dict[int, torch.Tensor], dict[int, float]]:
    """Split each raw ``r`` into a unit vector and its separation ``||r||``.

    Refusal-Lens stored the direction already normalized plus a separate ``separation``
    float, and picked its "best layer" by max separation (this is where Gemma's
    ``separation_layer=32`` came from, at 20,873 vs 3,101 at L15). We keep the RAW
    difference as the primitive because it carries both — but callers that assume unit
    norm must go through here rather than assuming, and anything reporting separation
    must say which convention (and dtype) produced it.
    """
    unit, separation = {}, {}
    for layer, r in directions.items():
        norm = float(torch.linalg.vector_norm(r))
        separation[layer] = norm
        unit[layer] = r / norm if norm > 0 else r.clone()
    return unit, separation


def compute_direction_candidates(
    model,
    tokenizer,
    spec: ModelSpec,
    harmful: list[str],
    harmless: list[str],
    positions: list[int] | None = None,
) -> dict[tuple[int, int], torch.Tensor]:
    """Arditi grid: ``r`` at each (layer, position). Returns ``{(layer, pos): r}``.

    ``positions=None`` uses ``spec.direction_positions`` (the post-instruction candidate set).
    Selecting the single best (layer, position) from these candidates is a downstream step.
    """
    if positions is None:
        positions = list(spec.direction_positions)
    # NOTE: Arditi keeps each position a SEPARATE candidate. Refusal-Lens'
    # collect_resid_acts_multipos instead averaged activations ACROSS positions before
    # differencing — a different estimator. Don't conflate the two sets of numbers.
    hf = _mean_resids_at_positions(model, tokenizer, spec, harmful, positions)
    hl = _mean_resids_at_positions(model, tokenizer, spec, harmless, positions)
    return {key: hf[key] - hl[key] for key in hf}
