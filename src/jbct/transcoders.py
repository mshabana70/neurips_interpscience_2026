"""H4: load a circuit-tracer ReplacementModel (model + transcoders) for a registry model.

Replaces the P0 scaffold. Everything model-specific lives in `ModelSpec.transcoder`, so this
module has no per-model branches — adding a model is a registry entry, not a code change.

Two behaviours of circuit-tracer are worth knowing before reading any graph produced here:

**Weight processing is disabled.** `from_pretrained_and_transcoders` passes
`fold_ln=False, center_writing_weights=False, center_unembed=False`, which is why
TransformerLens' residual stream matches HuggingFace's exactly (measured: cosine
`1.00000000` on all three models — `scripts/check_activation_parity.py`). That is what makes
a `jbct`-extracted direction a valid cotangent.

**Gemma-3-it zeroes its first four token positions.** `_configure_replacement_model` sets
`zero_positions = slice(0, 4)` for any `gemma-3*-it` model, versus `slice(0, 1)` elsewhere —
i.e. `<bos><start_of_turn>user\\n` contribute nothing to a Gemma graph. Our anchors sit well
past that (`user_end` at index 8, `role_marker` at 11), so they are unaffected; but a graph
will show no attribution from those tokens and that is circuit-tracer's design, not a bug in
ours. `expected_zero_positions()` exposes it so a report can state it rather than surprise
someone.
"""

from __future__ import annotations

from .registry import ModelSpec


def expected_zero_positions(spec: ModelSpec) -> int:
    """How many leading positions circuit-tracer zeroes for this model.

    Mirrors `ReplacementModel._configure_replacement_model`. Kept here so a graph's metadata
    can record it; if circuit-tracer changes the rule, `test_transcoders.py` fails rather than
    the discrepancy going unnoticed.
    """
    hf_id = spec.hf_id.lower()
    return 4 if "gemma-3" in hf_id and hf_id.endswith("-it") else 1


def parse_hf_ref(hf_ref: str) -> tuple[str, str | None, str | None]:
    """Mirror circuit-tracer's `HfUri.from_str`: (repo_id, file_path, revision).

    Reimplemented rather than imported so it can be validated WITHOUT circuit-tracer present,
    and so a malformed ref fails here with a readable message instead of as a 404 from deep
    inside `hf_hub_download`.
    """
    path_part, _, revision = hf_ref.partition("@")
    parts = path_part.split("/")
    if len(parts) < 2:
        raise ValueError(f"hf_ref {hf_ref!r} must be at least 'org/repo'")
    if any(p == "" for p in parts):
        raise ValueError(
            f"hf_ref {hf_ref!r} has an empty path component — a subfolder uses a SINGLE "
            f"slash ('org/repo/sub/folder'); a double slash 404s on resolve/main//sub/..."
        )
    repo_id = "/".join(parts[:2])
    file_path = "/".join(parts[2:]) if len(parts) > 2 else None
    return repo_id, file_path, (revision or None)


def load_replacement_model(spec: ModelSpec, device: str = "cuda", dtype=None,
                           lazy_encoder: bool = False):
    """The circuit-tracer model for `spec`. Raises if the registry has no transcoder.

    `lazy_encoder=True` keeps the encoder weights on disk and reads them per access. For
    Gemma that is 34 x 16384 x 2560 — **2.9 GB at bf16** — which is the difference between
    fitting and not on a 16 GB card. `ReplacementModel.from_pretrained` gives no way to pass
    it through (it forwards **kwargs to HookedTransformer, not to the transcoder loader), so
    the two stages are called separately here.
    """
    # Validate BEFORE importing circuit-tracer. It is an optional H4 dependency, so importing
    # first would surface "ModuleNotFoundError: circuit_tracer" on an H1-H3 checkout when the
    # real problem is that this model has no transcoder — the wrong error entirely.
    if spec.transcoder is None:
        raise ValueError(
            f"{spec.name} has no transcoder in the registry; H4 needs one. "
            f"H1-H3 are transcoder-free and do not require this."
        )
    if spec.transcoder.backend != "transformerlens":
        raise NotImplementedError(
            f"{spec.name} uses the {spec.transcoder.backend!r} backend; only "
            f"'transformerlens' is wired. nnsight/adapter models are Tier-2 follow-ups."
        )

    parse_hf_ref(spec.transcoder.hf_ref)  # fail readably on a malformed ref, not as a 404

    import torch
    from circuit_tracer.replacement_model import ReplacementModel

    resolved = torch.float32 if dtype is None else dtype
    if not lazy_encoder:
        return ReplacementModel.from_pretrained(
            spec.hf_id, spec.transcoder.hf_ref, device=device, dtype=resolved,
        )

    from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

    transcoders, _cfg = load_transcoder_from_hub(
        spec.transcoder.hf_ref, device=torch.device(device), dtype=resolved,
        lazy_encoder=True, lazy_decoder=True,
    )
    return ReplacementModel.from_pretrained_and_transcoders(
        spec.hf_id, transcoders, device=device, dtype=resolved,
    )


def verify_hooks(spec: ModelSpec, model) -> dict:
    """Check the loaded model's hooks against what the registry claims.

    The registry copies both hook names out of each repo's `config.yaml`. They are NOT
    interchangeable between models (Qwen writes to `mlp.hook_out`, Gemma to `hook_mlp_out`),
    so a silent mismatch would mean the transcoders are reading or writing somewhere other
    than where we recorded — which no downstream graph would reveal.
    """
    declared_in = spec.transcoder.feature_input_hook
    declared_out = spec.transcoder.feature_output_hook
    actual_in = getattr(model, "feature_input_hook", None)
    actual_out = getattr(model, "original_feature_output_hook", None)
    return {
        "feature_input_hook": {"declared": declared_in, "actual": actual_in,
                               "match": declared_in == actual_in},
        "feature_output_hook": {"declared": declared_out, "actual": actual_out,
                                "match": declared_out == actual_out},
        "zero_positions_expected": expected_zero_positions(spec),
        "zero_positions_actual": getattr(getattr(model, "zero_positions", None), "stop", None),
        "scan": getattr(model, "scan", None),
    }
