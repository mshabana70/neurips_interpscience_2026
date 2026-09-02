"""Resolve read positions by template LANDMARK rather than by negative index.

Negative indices silently break the moment an assistant-turn prefill is appended: Gemma's
`model` token sits at -2 on a bare prompt and at **-20** behind a prefill seed, and the shift
depends on the prefill's token count. Anything comparing an attack against its control at a
fixed negative index is then reading two different kinds of token.

Landmarks are stable because a prefill appends at the END, so the marker's ABSOLUTE index is
unchanged. This module returns absolute indices for named anchors:

    role_marker  the token opening the assistant turn ("model" / "assistant")
    user_end     the token closing the user turn ("<end_of_turn>" / "<|im_end|>")
    last         the final token, where generation continues from

Note Qwen's `enable_thinking=False` injects an empty `<think></think>` block, so `last` sits
several tokens after `role_marker` even without a prefill — the two anchors are genuinely
different questions, not the same one measured twice.
"""

from __future__ import annotations

from .models import format_prompt
from .registry import ModelSpec

ANCHORS = ("role_marker", "user_end", "last")


def _last_index_of(tokens: list[str], marker: str, limit: int) -> int | None:
    """Index of the last `marker` token strictly before `limit`."""
    for i in range(min(limit, len(tokens)) - 1, -1, -1):
        if tokens[i].strip() == marker:
            return i
    return None


def resolve_anchors(
    spec: ModelSpec, tokenizer, text: str, prefill: str | None = None
) -> dict[str, int]:
    """Absolute token indices of each anchor in the full (prompt + prefill) sequence.

    Raises if appending the prefill re-tokenizes the prompt differently — in that case the
    prompt's own token indices are no longer valid and silently proceeding would compare
    mismatched positions.
    """
    prompt = format_prompt(spec, tokenizer, text)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full = prompt + prefill if prefill else prompt
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]

    if full_ids[: len(prompt_ids)] != list(prompt_ids):
        raise ValueError(
            f"{spec.name}: appending the prefill changed the prompt's tokenization at the "
            "join, so prompt-relative anchors are invalid for this prefill"
        )

    tokens = [tokenizer.decode([i]) for i in full_ids]
    anchors: dict[str, int] = {"last": len(full_ids) - 1}
    for name, marker in (
        ("role_marker", spec.role_marker),
        ("user_end", spec.user_end_marker),
    ):
        # Search only the templated region: a prefill string could itself contain the word
        # "assistant" and would otherwise win the search.
        idx = _last_index_of(tokens, marker, limit=len(prompt_ids))
        if idx is not None:
            anchors[name] = idx
    return anchors
