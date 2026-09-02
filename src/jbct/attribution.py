"""H4 — attribution graphs targeting the refusal direction.

This module decides *what* gets attributed. The heavy lifting is circuit-tracer's; everything
here exists because three specific ways of getting it wrong have already cost us a run.

**1. The cotangent must be the COMPLEMENT direction.** Gemma's raw L15 diff-in-means direction
carries 0.81 of its squared norm on dimension 443, the attention-sink outlier. Attributing
toward it would yield a meticulous attribution graph *of the attention sink*. `refusal_cotangent`
drops `spec.outlier_dims` and records that it did, so a graph can never silently be built on
the raw vector.

**2. The measurement hook must match the extraction site (MENTEE_NOTE Bug 3).** The direction is
read from the residual stream (block output). circuit-tracer's default injects the cotangent at
the transcoder's `feature_input_hook` — post-RMSNorm, pre-MLP — which is ~1700x away in
magnitude across a non-linear normalization. `MEASUREMENT_HOOK` is not a parameter here; it is
a constant, because there is no correct alternative.

**3. Positions resolve by landmark, never by negative index.** A prefill appends to the
assistant turn, so Gemma's `model` token moves from -2 to -20 while its ABSOLUTE index is
unchanged. Attributing a prefill prompt at a fixed -2 would target a token in the middle of the
prefill string.

One plan per (prompt, anchor): each anchor is a different question — `role_marker` reads the
decision state before any prefill, `last` reads it at the point generation continues from — so
they get their own graphs rather than being averaged into one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .directions import drop_dims
from .positions import resolve_anchors
from .registry import ModelSpec

# Bug 3: attribute against the residual stream, the same basis the direction was read from.
MEASUREMENT_HOOK = "hook_resid_post"

# Anchors worth a graph, per family. Gemma gets 2, Qwen 3 — Qwen's `enable_thinking=False`
# injects an empty think block, so `role_marker` and `last` are 5+ tokens apart even with no
# prefill, and `user_end` is a genuinely distinct third question (the end of the USER turn,
# before any assistant scaffolding).
DEFAULT_ANCHORS = {
    "gemma": ("role_marker", "last"),
    "qwen": ("role_marker", "user_end", "last"),
    "llama": ("role_marker", "last"),
}


@dataclass
class AttributionPlan:
    """Everything needed to run one graph, plus everything needed to trust it later."""

    model: str
    layer: int
    anchor: str
    position: int  # ABSOLUTE token index, prefill-safe
    prompt: str  # the full templated string, prefill included
    direction: object  # unit cotangent (complement)
    measurement_hook: str = MEASUREMENT_HOOK
    metadata: dict = field(default_factory=dict)


def refusal_cotangent(spec: ModelSpec, r_raw, keep_outliers: bool = False):
    """Unit cotangent for attribution, with the outlier dims removed. Returns (vec, provenance).

    `keep_outliers=True` exists ONLY to reproduce the failure deliberately; it is never the
    right choice for a graph you intend to interpret.
    """
    import torch

    unit_raw = r_raw / torch.linalg.vector_norm(r_raw)
    energy = (
        float(sum(unit_raw[d] ** 2 for d in spec.outlier_dims)) if spec.outlier_dims else 0.0
    )
    dims = () if keep_outliers else spec.outlier_dims
    vec = drop_dims(unit_raw, dims) if dims else unit_raw
    return vec, {
        "outlier_dims": list(spec.outlier_dims),
        "outlier_dims_dropped": list(dims),
        "outlier_energy_of_raw_direction": round(energy, 4),
        "warning": (
            "outliers KEPT — this graph traces structural dimensions, not refusal"
            if keep_outliers and spec.outlier_dims
            else None
        ),
    }


def anchors_for(spec: ModelSpec, requested=None) -> tuple[str, ...]:
    return tuple(requested) if requested else DEFAULT_ANCHORS.get(spec.family, ("role_marker", "last"))


def plan_attributions(
    spec: ModelSpec,
    tokenizer,
    text: str,
    r_raw,
    layer: int,
    prefill: str | None = None,
    anchors=None,
    keep_outliers: bool = False,
) -> list[AttributionPlan]:
    """One plan per anchor for a single prompt."""
    from .generate import build_input

    vec, prov = refusal_cotangent(spec, r_raw, keep_outliers=keep_outliers)
    resolved = resolve_anchors(spec, tokenizer, text, prefill)
    prompt = build_input(spec, tokenizer, text, prefill)

    plans = []
    for name in anchors_for(spec, anchors):
        if name not in resolved:
            continue
        plans.append(
            AttributionPlan(
                model=spec.name,
                layer=layer,
                anchor=name,
                position=resolved[name],
                prompt=prompt,
                direction=vec,
                metadata={
                    "has_prefill": prefill is not None,
                    "anchors_resolved": dict(resolved),
                    # The negative index is recorded for HUMAN reading only. It is not what we
                    # attribute at, and it moves when a prefill is present.
                    "position_negative_equivalent": resolved[name] - (resolved["last"] + 1),
                    "direction": prov,
                    "causal_layer_provenance": [
                        spec.causal_layer.value,
                        spec.causal_layer.provenance,
                    ],
                },
            )
        )
    return plans


def align_to_model(vec, model):
    """Put the cotangent on the model's device and dtype.

    Directions live on CPU in float64 — that is how `candidates.pt` is loaded (`map_location=
    "cpu"`) and how the means are accumulated. circuit-tracer injects the vector straight into
    a CUDA autograd graph, so an unaligned cotangent surfaces as
    `RuntimeError: Expected all tensors to be on the same device` from inside ct's `_inject`,
    naming ct's tensors rather than ours.

    Same failure mode as the H1xH2 chain's `project_scalar`: align BOTH sides at the boundary,
    never rely on promotion, and never assume a tensor loaded from disk carries no device.
    """
    p = next(model.parameters())
    return vec.detach().to(device=p.device, dtype=p.dtype)


def to_target(plan: AttributionPlan, model=None, label: str | None = None):
    """circuit-tracer `CustomTarget` for this plan's cotangent.

    Pass `model` so the vector is aligned to it; omitting it is only for inspection.
    """
    from circuit_tracer.attribution.targets import CustomTarget

    vec = align_to_model(plan.direction, model) if model is not None else plan.direction
    return CustomTarget(
        token_str=label or f"refusal@L{plan.layer}:{plan.anchor}",
        prob=1.0,
        vec=vec,
    )


def run_attribution(plan: AttributionPlan, model, **kwargs):
    """Execute one plan. `measurement_hook` is fixed — see module docstring (Bug 3)."""
    from circuit_tracer import attribute

    return attribute(
        plan.prompt,
        model,
        attribution_targets=[to_target(plan, model)],
        measurement_layer=plan.layer,
        measurement_position=plan.position,
        measurement_hook=plan.measurement_hook,
        **kwargs,
    )
