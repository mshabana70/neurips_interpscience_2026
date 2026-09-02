"""GPU smoke: end-to-end direction extraction, per model. Run with `-m gpu`.

Qwen3-0.6B is the M1 slice gate (ungated, ~1.2 GB) — it replaced Llama-3.2-1B, whose HF
repo we cannot access. Delegated models still appear here as explicit SKIPs carrying their
owner and reason, so `pytest -m gpu -v` reports what was *not* validated locally instead of
quietly omitting it.
"""

import pytest

from jbct.directions import compute_directions
from jbct.models import format_prompt, load_model
from jbct.registry import DELEGATED_GPU_VALIDATION, MODELS, get_model_spec

SLICE_MODEL = "qwen3-0.6b"


@pytest.mark.gpu
@pytest.mark.parametrize("model_name", sorted(MODELS))
def test_slice_end_to_end(model_name, toy_prompts):
    delegation = DELEGATED_GPU_VALIDATION.get(model_name)
    if delegation:
        pytest.skip(f"delegated to {delegation.owner}: {delegation.reason}")

    spec = get_model_spec(model_name)
    model, tok = load_model(spec, device="cuda")
    harmful, harmless = toy_prompts
    harmful = [format_prompt(spec, tok, t) for t in harmful]
    harmless = [format_prompt(spec, tok, t) for t in harmless]

    trace: dict = {}
    dirs = compute_directions(model, tok, spec, harmful, harmless, trace=trace)

    # (a) ran on CUDA
    assert next(model.parameters()).is_cuda
    # (b) per-layer d_model directions, and the registry shape is not stale
    assert set(dirs) == set(range(spec.n_layers))
    assert dirs[0].shape == (spec.d_model,)
    # (c) read where the spec claims (R6a)
    assert trace["positions"] == [spec.direction_position.value]
    # (d) non-trivial separation at SOME middle layer (mechanism sanity, not a verified layer)
    norms = {layer: dirs[layer].norm().item() for layer in dirs}
    mid = range(spec.n_layers // 3, 2 * spec.n_layers // 3)
    assert max(norms[layer] for layer in mid) > 0, f"all-zero directions: {norms}"
