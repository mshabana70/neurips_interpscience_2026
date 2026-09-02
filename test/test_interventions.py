"""CPU tests for the H3 causal interventions."""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jbct.interventions import ActivationAddition, ArditiAblation, DirectionalAblation
from jbct.models import HookManager, get_layers, tokenize_prompt
from test.conftest import TINY_LLAMA


@pytest.fixture(scope="module")
def tiny():
    return AutoModelForCausalLM.from_pretrained(TINY_LLAMA), AutoTokenizer.from_pretrained(TINY_LLAMA)


def _resid(model, tok, spec, hook=None):
    ids = tokenize_prompt(tok, "hello world")
    ctx = hook if hook is not None else _Null()
    with ctx, HookManager(model, spec) as hm:
        model(**ids)
    return {k: v.clone() for k, v in hm.residuals.items()}


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_ablation_removes_the_component_along_the_direction(tiny_spec, tiny):
    model, tok = tiny
    r = torch.zeros(tiny_spec.d_model, dtype=torch.float64)
    r[0] = 1.0  # a basis direction makes the expected result exact
    with DirectionalAblation(model, tiny_spec, r) as abl:
        after = _resid(model, tok, tiny_spec)
    assert abl.n_calls > 0
    for layer in after:
        # the component along r must be ~0 at every layer after ablation
        assert after[layer][..., 0].abs().max() < 1e-2


def test_ablation_leaves_orthogonal_components_alone(tiny_spec, tiny):
    model, tok = tiny
    base = _resid(model, tok, tiny_spec)
    r = torch.zeros(tiny_spec.d_model, dtype=torch.float64)
    r[0] = 1.0
    with DirectionalAblation(model, tiny_spec, r):
        after = _resid(model, tok, tiny_spec)
    # layer 0 is edited from a clean input, so its non-r coordinates must be untouched
    assert torch.allclose(base[0][..., 1:], after[0][..., 1:], atol=1e-3)


def test_ablation_normalizes_an_unnormalized_direction(tiny_spec, tiny):
    """A caller passing raw r must not scale the intervention by ||r|| (which varies ~7x)."""
    model, tok = tiny
    unit, scaled = torch.zeros(tiny_spec.d_model), torch.zeros(tiny_spec.d_model)
    unit[0], scaled[0] = 1.0, 500.0
    with DirectionalAblation(model, tiny_spec, unit):
        a = _resid(model, tok, tiny_spec)
    with DirectionalAblation(model, tiny_spec, scaled):
        b = _resid(model, tok, tiny_spec)
    assert torch.allclose(a[0], b[0], atol=1e-4)


def test_exact_arditi_ablation_projects_every_residual_writer(tiny_spec, tiny):
    model, tok = tiny
    r = torch.zeros(tiny_spec.d_model, dtype=torch.float64)
    r[0] = 1.0
    with ArditiAblation(model, tiny_spec, r) as ablation:
        after = _resid(model, tok, tiny_spec)
    assert ablation.n_calls >= 3 * len(get_layers(model, tiny_spec))
    for layer in after:
        assert after[layer][..., 0].abs().max() < 1e-2


def test_exact_arditi_ablation_can_be_restricted_to_one_layer(tiny_spec, tiny):
    model, tok = tiny
    r = torch.zeros(tiny_spec.d_model, dtype=torch.float64)
    r[0] = 1.0
    with ArditiAblation(model, tiny_spec, r, layers=[0]) as ablation:
        _resid(model, tok, tiny_spec)
    assert ablation.layers == [0]
    assert ablation.n_calls == 3


def test_addition_moves_the_residual_along_the_direction(tiny_spec, tiny):
    model, tok = tiny
    base = _resid(model, tok, tiny_spec)
    r = torch.zeros(tiny_spec.d_model)
    r[0] = 1.0
    with ActivationAddition(model, tiny_spec, r, layer=0, coeff=5.0, absolute=True):
        after = _resid(model, tok, tiny_spec)
    delta = (after[0][..., 0] - base[0][..., 0]).abs().mean()
    assert delta == pytest.approx(5.0, abs=0.5)


def test_addition_touches_only_its_own_layer(tiny_spec, tiny):
    model, tok = tiny
    r = torch.zeros(tiny_spec.d_model)
    r[0] = 1.0
    add = ActivationAddition(model, tiny_spec, r, layer=0, coeff=1.0, absolute=True)
    assert add.layers == [0]
    n_layers = len(get_layers(model, tiny_spec))
    with DirectionalAblation(model, tiny_spec, r) as abl:
        _resid(model, tok, tiny_spec)
    assert len(abl.layers) == n_layers  # ablation defaults to EVERY layer


def test_hooks_are_removed_on_exit(tiny_spec, tiny):
    model, tok = tiny
    base = _resid(model, tok, tiny_spec)
    r = torch.zeros(tiny_spec.d_model)
    r[0] = 1.0
    with DirectionalAblation(model, tiny_spec, r):
        pass
    after = _resid(model, tok, tiny_spec)
    assert torch.allclose(base[0], after[0])  # no residue from the context manager


def test_fires_on_every_generation_step_not_just_the_prompt(tiny_spec, tiny):
    """Ablating only the prompt would let the model re-introduce the direction while writing."""
    model, tok = tiny
    ids = tokenize_prompt(tok, "hello")
    r = torch.zeros(tiny_spec.d_model)
    r[0] = 1.0
    with DirectionalAblation(model, tiny_spec, r) as abl:
        model.generate(**ids, do_sample=False, max_new_tokens=4, pad_token_id=tok.eos_token_id)
    n_layers = len(get_layers(model, tiny_spec))
    assert abl.n_calls > n_layers  # more than a single forward pass


def test_prompt_only_ablation_edits_prompt_but_not_cached_generation(tiny_spec, tiny):
    model, _ = tiny
    r = torch.zeros(tiny_spec.d_model, dtype=torch.float64)
    r[0] = 1.0
    abl = DirectionalAblation(model, tiny_spec, r, layers=[0], prompt_only=True)
    prompt = torch.randn(2, 4, tiny_spec.d_model)
    generation = torch.randn(2, 1, tiny_spec.d_model)

    prompt_after = abl._edit(prompt)
    generation_after = abl._edit(generation)

    assert prompt_after[..., 0].abs().max() < 1e-6
    assert torch.equal(generation_after, generation)
    assert abl.n_edits == 1


def test_rejects_an_out_of_range_layer(tiny_spec, tiny):
    model, _ = tiny
    r = torch.zeros(tiny_spec.d_model)
    with pytest.raises(ValueError, match="out of range"):
        ActivationAddition(model, tiny_spec, r, layer=999, coeff=1.0)
