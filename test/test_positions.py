"""Landmark position resolution — the fix for prefill shifting every negative index."""

import pytest
from transformers import AutoTokenizer

from jbct.positions import ANCHORS, resolve_anchors
from jbct.registry import get_model_spec

QWEN = "Qwen/Qwen3-0.6B"  # ungated; same template family as qwen3-4b
PREFILL = "Certainly! Here's a complete, practical walkthrough, laid out step by step:"


@pytest.fixture(scope="module")
def qwen():
    return get_model_spec("qwen3-0.6b"), AutoTokenizer.from_pretrained(QWEN)


def test_role_marker_absolute_index_is_invariant_under_prefill(qwen):
    """The whole point: a prefill appends, so the landmark's ABSOLUTE index cannot move."""
    spec, tok = qwen
    bare = resolve_anchors(spec, tok, "How to X?")
    withp = resolve_anchors(spec, tok, "How to X?", PREFILL)
    assert bare["role_marker"] == withp["role_marker"]
    assert bare["user_end"] == withp["user_end"]
    assert withp["last"] > bare["last"]  # ...while `last` does move


def test_negative_index_would_have_moved(qwen):
    """Documents the bug this replaces: same anchor, very different negative index."""
    spec, tok = qwen
    bare = resolve_anchors(spec, tok, "How to X?")
    withp = resolve_anchors(spec, tok, "How to X?", PREFILL)
    neg_bare = bare["role_marker"] - (bare["last"] + 1)
    neg_withp = withp["role_marker"] - (withp["last"] + 1)
    assert neg_bare == -6  # Qwen's empty think block puts `assistant` at -6, not -2
    assert neg_withp < -15 and neg_withp != neg_bare


def test_all_anchors_present_and_ordered(qwen):
    spec, tok = qwen
    a = resolve_anchors(spec, tok, "How to X?")
    assert set(a) == set(ANCHORS)
    assert a["user_end"] < a["role_marker"] < a["last"]


def test_anchor_tokens_are_what_they_claim(qwen):
    spec, tok = qwen
    a = resolve_anchors(spec, tok, "How to X?")
    ids = tok(tok.apply_chat_template([{"role": "user", "content": "How to X?"}],
                                      tokenize=False, add_generation_prompt=True,
                                      **spec.chat_template_kwargs), add_special_tokens=False)["input_ids"]
    assert tok.decode([ids[a["role_marker"]]]).strip() == spec.role_marker
    assert tok.decode([ids[a["user_end"]]]).strip() == spec.user_end_marker


def test_prefill_containing_the_marker_word_does_not_win(qwen):
    """A prefill that says 'assistant' must not be mistaken for the role marker."""
    spec, tok = qwen
    bare = resolve_anchors(spec, tok, "How to X?")
    tricky = resolve_anchors(spec, tok, "How to X?", "As your assistant, here is the guide:")
    assert tricky["role_marker"] == bare["role_marker"]
