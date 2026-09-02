"""Model loading, chat-template abstraction, residual-layer access."""

from __future__ import annotations

from .registry import ModelSpec


def format_prompt(spec: ModelSpec, tokenizer, text: str) -> str:
    """Apply the model's chat template with spec-driven kwargs (e.g. Qwen enable_thinking)."""
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **spec.chat_template_kwargs
    )


# Decoder-block locations, most-specific first. ORDER IS LOAD-BEARING: a multimodal
# wrapper also exposes `vision_tower.vision_model.encoder.layers`, so a generic
# "first module with .layers" walk can silently return the VISION stack and every
# downstream direction would be computed on image features. We only ever look at
# explicitly language-side paths, and never recurse.
_LAYER_PATHS: tuple[tuple[str, ...], ...] = (
    ("model", "language_model"),  # Gemma-3 @ transformers 5.x: Gemma3ForConditionalGeneration
    ("language_model", "model"),  # Gemma-3 @ transformers 4.x: .language_model is a CausalLM
    ("language_model",),  # multimodal wrappers exposing the text model directly
    ("model",),  # Llama / Qwen / Gemma-2 causal LMs
    (),  # already a bare decoder
)


def get_layers(model, spec: ModelSpec | None = None) -> list:
    """Return the list of decoder blocks, arch-agnostically.

    Pass ``spec`` to assert the count matches the registry — a stale ``n_layers`` then
    fails loudly here instead of silently producing a short direction set.
    """
    for path in _LAYER_PATHS:
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "layers"):
            layers = list(obj.layers)
            if spec is not None and len(layers) != spec.n_layers:
                raise ValueError(
                    f"{spec.name}: registry says n_layers={spec.n_layers} but "
                    f"{type(model).__name__} exposes {len(layers)} decoder blocks "
                    f"at {'.'.join(('model', *path, 'layers'))}"
                )
            return layers
    raise AttributeError(
        f"cannot locate decoder layers on {type(model).__name__}; "
        f"tried {['.'.join(p) or '<self>' for p in _LAYER_PATHS]}"
    )


def get_d_model(model) -> int:
    """Residual width, arch-agnostically.

    Multimodal configs (Gemma-3) nest the text hyperparameters under ``text_config``;
    reading ``config.hidden_size`` on those returns the *wrapper's* value or nothing.
    """
    cfg = model.config
    return int(getattr(cfg, "text_config", cfg).hidden_size)


def tokenize_prompt(tokenizer, text: str, device=None):
    """Tokenize an ALREADY chat-templated string, without re-adding special tokens.

    ``add_special_tokens=False`` is load-bearing. ``format_prompt`` returns a string that
    already contains the template's own BOS (``<bos>`` for Gemma, ``<|begin_of_text|>``
    for Llama); tokenizing it with the default ``add_special_tokens=True`` prepends a
    SECOND one. Refusal-Lens' Stage 01 had exactly this double-BOS and measured its effect
    (Gemma cosine 0.999983 at L15) — negligible, because a token added at the *front*
    doesn't shift negative indices, so the read position is unchanged. Negligible is not
    the same as correct, and it was never measured on Qwen, so we emit the right input.
    """
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return enc.to(device) if device is not None else enc


def count_leading_bos(tokenizer, input_ids) -> int:
    """How many BOS tokens the sequence opens with (expected: 0 or 1, never 2)."""
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is None:
        return 0
    row = input_ids[0] if input_ids.dim() > 1 else input_ids
    n = 0
    for tok in row.tolist():
        if tok != bos:
            break
        n += 1
    return n


class HookManager:
    """Capture each decoder layer's output residual (== ``hidden_states[L+1]``).

    P0 captures layer *outputs* uniformly across architectures; the arch-specific
    *injection* point (Gemma's next-layer input_layernorm, carried in
    ``ModelSpec.hook_resolution``) is for later H3/H4 and is not exercised here.

    Note: this is the RAW residual. ``output_hidden_states[L+1]`` equals it for every
    middle layer, but HF applies the final RMSNorm to the last entry
    (``hidden_states[n] == norm(last output)``). We always measure at a middle layer,
    so the raw value here is the correct residual-stream reading.
    """

    def __init__(self, model, spec: ModelSpec):
        self.model = model
        self.spec = spec
        self.residuals: dict[int, object] = {}
        self._handles: list = []

    def _make_hook(self, idx: int):
        def hook(_module, _inp, output):
            resid = output[0] if isinstance(output, tuple) else output
            self.residuals[idx] = resid.detach()

        return hook

    def __enter__(self):
        for i, layer in enumerate(get_layers(self.model)):
            self._handles.append(layer.register_forward_hook(self._make_hook(i)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


def load_model(spec: ModelSpec, device: str = "cuda", dtype=None):
    """Load model + tokenizer at the pinned revision on `device`.

    ``dtype`` defaults to bf16, which is right for plumbing smoke tests but is NOT the
    methodology behind Refusal-Lens' verified physics: Stage 01 loaded **float32**
    ("float32 for direction computation") and accumulated in float64.
    Scientific direction extraction should pass ``dtype=torch.float32`` explicitly and
    record it — bf16 carries ~8 mantissa bits against residual norms of ~30,000, so
    directions from the two dtypes must never be compared numerically. Note fp32
    Gemma-3-4B is ~16 GB and will not fit the 5080 alongside activations: full-precision
    extraction is RunPod work (see the delegation split in `registry`).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(spec.hf_id, revision=spec.hf_revision)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id,
        revision=spec.hf_revision,
        dtype=torch.bfloat16 if dtype is None else dtype,  # transformers 5.x: `dtype`
        device_map=device,
    )
    model.eval()
    return model, tok
