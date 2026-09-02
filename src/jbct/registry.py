"""The one config surface: per-model physics + provenance.

Every physics field is a ``Physics(value, provenance)`` so a placeholder can never be
mistaken for a verified value (north-star §4.1). H1-H3 do not need transcoders, so
``transcoder`` is ``None`` in P0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Physics:
    value: Any
    provenance: str  # "verified" | "placeholder"

    def __post_init__(self) -> None:
        if self.provenance not in ("verified", "placeholder"):
            msg = f"bad provenance: {self.provenance!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class Delegation:
    """Why a registry model cannot be GPU-validated locally, and who owns validating it."""

    owner: str  # who runs the validation
    reason: str  # why the RTX 5080 can't


@dataclass(frozen=True)
class TranscoderSpec:
    """Where H4's transcoders come from, and what they hook.

    `hf_ref` is the string circuit-tracer's `load_transcoder_from_hub` resolves. `HfUri.from_str`
    splits on `/`, takes the FIRST TWO components as the repo id and the remainder as the
    subfolder, with an optional `@revision` suffix — so a subfolder is a plain SINGLE slash
    (`org/repo/sub/folder`). A double slash yields an empty path component and a 404 on
    `resolve/main//sub/...`. (`scan` is formatted with `//` internally; that is a display
    string, not the input format.)

    The two hook names are copied from each repo's own `config.yaml` and recorded here so a
    mismatch is visible rather than latent — they are NOT interchangeable between models
    (Qwen writes to `mlp.hook_out`, Gemma to `hook_mlp_out`).
    """

    hf_ref: str
    repo: str
    revision: str
    type: str  # "plt" | "clt" | "skip"
    backend: str  # "nnsight" | "transformerlens"
    source: str  # "native" | "adapter"
    # From the repo config.yaml. `feature_input_hook` is the post-RMSNorm pre-MLP point that
    # circuit-tracer injects the cotangent at BY DEFAULT — i.e. the origin of Bug 3. We
    # override it with attribution.MEASUREMENT_HOOK; this field documents what we override.
    feature_input_hook: str
    feature_output_hook: str


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_id: str
    hf_revision: str
    family: str
    n_layers: int
    d_model: int
    causal_layer: Physics
    separation_layer: Physics
    direction_position: Physics  # the selected/canonical position (provenance-tracked)
    direction_positions: tuple  # Arditi candidate search set (post-instruction region); config, not physics
    chat_template_kwargs: dict
    # Template landmarks, used to resolve read positions by NAME rather than by a negative
    # index. Load-bearing for prefill: appending an assistant-turn prefill shifts every
    # negative index (Gemma's `model` token moves from -2 to -20), while the landmark's
    # absolute index is unchanged.
    role_marker: str  # token that opens the assistant turn
    user_end_marker: str  # token that closes the user turn
    # Massive-activation ("outlier") residual dimensions. These carry structural signal
    # (attention-sink style) of enormous magnitude, and a diff-in-means direction loads onto
    # them heavily without them meaning anything about refusal. Ablating such a direction is
    # a lobotomy, not an intervention. Empty tuple = not yet measured, NOT "none exist".
    outlier_dims: tuple
    hook_resolution: str
    transcoder: TranscoderSpec | None
    tier: int


MODELS: dict[str, ModelSpec] = {
    "llama-3.2-1b": ModelSpec(
        name="llama-3.2-1b",
        hf_id="meta-llama/Llama-3.2-1B-Instruct",
        hf_revision="main",  # TODO(env task): pin to a commit SHA once confirmed
        family="llama",
        n_layers=16,
        d_model=2048,
        causal_layer=Physics(None, "placeholder"),
        separation_layer=Physics(None, "placeholder"),
        direction_position=Physics(-1, "placeholder"),
        direction_positions=(-5, -4, -3, -2, -1),  # post-instruction region (verify vs Llama template)
        chat_template_kwargs={},
        role_marker="assistant",
        user_end_marker="<|eot_id|>",
        outlier_dims=(),
        hook_resolution="decoder_layer_output",
        transcoder=None,
        tier=1,
    ),
    "gemma-3-4b": ModelSpec(
        name="gemma-3-4b",
        hf_id="google/gemma-3-4b-it",
        hf_revision="main",
        family="gemma",
        n_layers=34,
        d_model=2560,
        causal_layer=Physics(15, "verified"),  # verified by the causal sweep
        separation_layer=Physics(32, "verified"),
        direction_position=Physics(-2, "verified"),  # "model" token, Gemma-3 template
        direction_positions=(-5, -3, -2),  # template-anchored (old TARGET_POSITIONS_MULTI)
        chat_template_kwargs={},
        role_marker="model",
        user_end_marker="<end_of_turn>",
        # MEASURED 2026-07-28 (scripts/measure_outlier_dims.py, L15, 64 prompts): max/median
        # activation ratio 1562x. The L15 refusal direction puts **0.81 of its squared norm**
        # on these — it is mostly structure, so causal work MUST use the complement.
        outlier_dims=(443, 634),
        hook_resolution="gemma3_next_input_layernorm",
        transcoder=TranscoderSpec(
            hf_ref="mwhanna/gemma-scope-2-4b-it/transcoder_all/width_16k_l0_big",
            repo="mwhanna/gemma-scope-2-4b-it",
            revision="main",
            type="plt",
            backend="transformerlens",
            source="native",
            feature_input_hook="mlp.hook_in",
            feature_output_hook="hook_mlp_out",  # NB: differs from Qwen's
        ),
        tier=1,
    ),
    # The local fast-iteration slice model (replaces Llama-3.2-1B as the M1 GPU gate:
    # ungated on HF, ~1.2 GB bf16). n_layers/d_model verified against the published config.json.
    "qwen3-0.6b": ModelSpec(
        name="qwen3-0.6b",
        hf_id="Qwen/Qwen3-0.6B",
        hf_revision="main",  # TODO(env task): pin to a commit SHA once confirmed
        family="qwen",
        n_layers=28,
        d_model=1024,
        causal_layer=Physics(None, "placeholder"),
        separation_layer=Physics(None, "placeholder"),
        direction_position=Physics(-1, "placeholder"),
        direction_positions=(-5, -3, -1),
        chat_template_kwargs={"enable_thinking": False},  # load-bearing (R4)
        role_marker="assistant",
        user_end_marker="<|im_end|>",
        outlier_dims=(),
        hook_resolution="decoder_layer_output",
        transcoder=None,
        tier=1,
    ),
    "qwen3-4b": ModelSpec(
        name="qwen3-4b",
        hf_id="Qwen/Qwen3-4B",
        hf_revision="main",
        family="qwen",
        n_layers=36,
        d_model=2560,  # confirm on load
        causal_layer=Physics(18, "placeholder"),  # was placeholder in old repo; re-verify
        separation_layer=Physics(None, "placeholder"),
        direction_position=Physics(-1, "placeholder"),
        direction_positions=(-5, -3, -1),
        chat_template_kwargs={"enable_thinking": False},  # load-bearing (R4)
        role_marker="assistant",
        user_end_marker="<|im_end|>",
        # MEASURED 2026-07-28 (L14/18/22): outliers DO exist at 39-50x the median activation,
        # but the refusal direction puts only **0.017-0.022** of its squared norm on them —
        # versus Gemma's 0.81. Present but irrelevant to the direction, which is why ablating
        # Qwen's raw direction never collapsed the model the way Gemma's did.
        outlier_dims=(0, 3, 4),
        hook_resolution="decoder_layer_output",
        transcoder=TranscoderSpec(
            hf_ref="mwhanna/qwen3-4b-transcoders",
            repo="mwhanna/qwen3-4b-transcoders",
            revision="main",
            type="plt",
            backend="transformerlens",
            source="native",
            feature_input_hook="mlp.hook_in",
            feature_output_hook="mlp.hook_out",  # NB: differs from Gemma's
        ),
        tier=1,
    ),
    "qwen3-14b": ModelSpec(
        name="qwen3-14b",
        hf_id="Qwen/Qwen3-14B",
        hf_revision="main",
        family="qwen",
        n_layers=40,
        d_model=5120,  # confirm on load
        causal_layer=Physics(None, "placeholder"),
        separation_layer=Physics(None, "placeholder"),
        direction_position=Physics(-1, "placeholder"),
        direction_positions=(-5, -3, -1),
        chat_template_kwargs={"enable_thinking": False},
        role_marker="assistant",
        user_end_marker="<|im_end|>",
        outlier_dims=(),
        hook_resolution="decoder_layer_output",
        transcoder=None,
        tier=1,
    ),
}


# --- Where each model actually gets GPU-validated -----------------------------------
# Code supports every model in MODELS identically; *validation* is what splits them.
# The partition below is total and disjoint (enforced by test/test_registry.py) so a
# green local suite can never be mistaken for evidence that a delegated model works.

LOCAL_GPU_MODELS: tuple[str, ...] = ("qwen3-0.6b", "gemma-3-4b", "qwen3-4b")

DELEGATED_GPU_VALIDATION: dict[str, Delegation] = {
    "llama-3.2-1b": Delegation(
        owner="anonymous",
        reason="meta-llama/Llama-3.2-1B-Instruct is gated; no local HF access as of 2026-07-28. "
        "Run `scripts/validate_model_gpu.py --model llama-3.2-1b` and report the JSON.",
    ),
    "qwen3-14b": Delegation(
        owner="RunPod (P2 scale-out)",
        reason="~28 GB of bf16 weights exceeds the 16 GB RTX 5080. Quantizing would perturb "
        "the activations the direction is computed from, so validate at full precision off-box.",
    ),
}


def get_model_spec(name: str) -> ModelSpec:
    return MODELS[name]
