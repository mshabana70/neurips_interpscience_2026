import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jbct.directions import compute_direction_candidates, compute_directions
from jbct.models import HookManager, tokenize_prompt
from test.conftest import TINY_LLAMA


def test_directions_shape_and_layers(tiny_spec, toy_prompts):
    model = AutoModelForCausalLM.from_pretrained(TINY_LLAMA)
    tok = AutoTokenizer.from_pretrained(TINY_LLAMA)
    harmful, harmless = toy_prompts
    dirs = compute_directions(model, tok, tiny_spec, harmful, harmless)
    assert set(dirs) == set(range(model.config.num_hidden_layers))
    assert dirs[0].shape == (model.config.hidden_size,)


def test_directions_are_diff_of_means(tiny_spec):
    # r == harmful_resid - harmless_resid at the model's canonical position (spec-driven, not -1)
    model = AutoModelForCausalLM.from_pretrained(TINY_LLAMA)
    tok = AutoTokenizer.from_pretrained(TINY_LLAMA)
    pos = tiny_spec.direction_position.value
    dirs = compute_directions(model, tok, tiny_spec, ["aaa"], ["bbb"])
    layer = model.config.num_hidden_layers // 2

    def resid(text, layer_idx):
        # Must tokenize the SAME way compute_directions does. Using bare `tok(text)` here
        # re-adds a BOS and the reference silently drifts ~1-3% per element — small, but
        # this equality test is exactly where such a drift should be caught.
        ids = tokenize_prompt(tok, text)
        with HookManager(model, tiny_spec) as hm:
            model(**ids)
        return hm.residuals[layer_idx][0, pos].double()

    expected = resid("aaa", layer) - resid("bbb", layer)
    assert torch.allclose(dirs[layer].double(), expected, atol=1e-6)


def test_direction_candidates_grid(tiny_spec, toy_prompts):
    # Arditi grid: a direction at each (layer, position)
    model = AutoModelForCausalLM.from_pretrained(TINY_LLAMA)
    tok = AutoTokenizer.from_pretrained(TINY_LLAMA)
    harmful, harmless = toy_prompts
    positions = [-2, -1]  # small positions that fit the short toy prompts
    cands = compute_direction_candidates(model, tok, tiny_spec, harmful, harmless, positions=positions)
    n = model.config.num_hidden_layers
    assert set(cands) == {(layer, p) for layer in range(n) for p in positions}
    assert cands[(0, -1)].shape == (model.config.hidden_size,)


def test_candidates_consistent_with_single_position(tiny_spec, toy_prompts):
    # a candidate at the spec's canonical position must equal the single-position direction
    model = AutoModelForCausalLM.from_pretrained(TINY_LLAMA)
    tok = AutoTokenizer.from_pretrained(TINY_LLAMA)
    harmful, harmless = toy_prompts
    pos = tiny_spec.direction_position.value
    single = compute_directions(model, tok, tiny_spec, harmful, harmless)
    cands = compute_direction_candidates(model, tok, tiny_spec, harmful, harmless, positions=[pos])
    for layer in single:
        assert torch.equal(single[layer], cands[(layer, pos)])
