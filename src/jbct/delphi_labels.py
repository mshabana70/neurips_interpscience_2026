"""Delphi-style feature explanations, without Delphi's dependency tree.

**What this is.** A port of the *prompt methodology* from EleutherAI's Delphi
(https://github.com/EleutherAI/delphi, `delphi/explainers/default/`): show the model each
activating example with the firing tokens wrapped in `<<...>>`, followed by per-token
activation strengths as `("token", N)` tuples, and require the answer after `[EXPLANATION]:`.

**What this is NOT.** It is not the Delphi library. `eai-delphi` hard-depends on
`vllm>=0.10.2`, `bitsandbytes`, `faiss-cpu`, `sentence_transformers` and `eai-sparsify` —
vllm pins torch/transformers aggressively and would very likely break the
`transformers==4.57.3` + circuit-tracer environment we gated carefully
(docs/reorg/2026-07-28-transformers-downgrade-for-h4.md). The prompt design is a few hundred
bytes of string formatting and carries none of that.

**Why it beats the heuristic.** `heuristic_label` reports a bag of peak tokens plus top
logits: no context, no gradation. Most of the signal about what a feature detects is in the
CONTEXT it fires in and in which tokens fire *hardest* — both of which this preserves.

**What it still cannot do.** Delphi's real advantage is SCORING explanations (detection,
fuzzing), and every classifier scorer builds from `record.not_active` — non-activating
examples. The transcoder repos ship only top-activating examples, so we have no negatives and
`DetectionScorer` would silently score against an empty negative set. Explanations here are
therefore unvalidated, exactly like the plain LLM labels. Phase 2 is collecting negatives.
"""

from __future__ import annotations

import re

MAX_ACT_SCALE = 10  # Delphi presents activations on a 0-10 scale
DEFAULT_N_EXAMPLES = 8

# Bumped whenever `parse_explanation` changes what it accepts. Cached LLM labels record it,
# so tightening the parser retires the labels the looser one let through instead of leaving
# them on the dashboard forever (the same versioning the heuristic labels already use).
PARSE_VERSION = 2
MIN_FALLBACK_WORDS = 3
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s")

SYSTEM_PROMPT = (
    "You are a mechanistic-interpretability researcher. You are shown text examples where a "
    "neural network feature activates. Tokens the feature fires on are wrapped in <<double "
    "angle brackets>>, followed by that example's activation strengths as (token, strength) "
    "pairs on a 0-10 scale.\n\n"
    "Describe the pattern the feature detects. Be specific and concise — at most 12 words. "
    "Do not list alternatives or hedge; state the single best description. If the examples "
    "share only a surface form (a punctuation mark, a template token), say so plainly.\n\n"
    "End your reply with the description on its own line prefixed by [EXPLANATION]:"
)

# One worked example, in Delphi's format, so the model copies the output shape.
FEW_SHOT = (
    "Example 1: and he was <<over the moon>> to find\n"
    'Activations: ("over", 5), (" the", 6), (" moon", 9)\n'
    "Example 2: we were <<on cloud nine>> after hearing\n"
    'Activations: ("on", 4), (" cloud", 7), (" nine", 8)\n'
    "\n[EXPLANATION]: idioms expressing strong happiness"
)


def normalize_activations(acts: list[float], scale: int = MAX_ACT_SCALE) -> list[int]:
    """Scale to 0..`scale` integers relative to the example's own maximum.

    Per-example rather than global: the prompt asks which tokens fire hardest *within this
    context*, and raw magnitudes vary by orders of magnitude across features.
    """
    peak = max((a for a in acts), default=0.0)
    if peak <= 0:
        return [0] * len(acts)
    return [max(0, round(a / peak * scale)) for a in acts]


def highlight(tokens: list[str], norm: list[int], threshold: int = 1) -> str:
    """Wrap contiguous runs of activating tokens in `<<...>>` (Delphi's markup).

    Runs are merged so a multi-token phrase reads as one span rather than
    `<<over>><< the>><< moon>>`.
    """
    out, i = [], 0
    n = min(len(tokens), len(norm))
    while i < n:
        if norm[i] >= threshold:
            j = i
            while j < n and norm[j] >= threshold:
                j += 1
            out.append("<<" + "".join(tokens[i:j]) + ">>")
            i = j
        else:
            out.append(tokens[i])
            i += 1
    return "".join(out)


def activation_pairs(tokens: list[str], norm: list[int], threshold: int = 1) -> str:
    """`("tok", 5), (" the", 6)` for the activating tokens only."""
    parts = [
        f'("{tokens[i]}", {norm[i]})'
        for i in range(min(len(tokens), len(norm)))
        if norm[i] >= threshold
    ]
    return ", ".join(parts)


def build_prompt(record, n_examples: int = DEFAULT_N_EXAMPLES) -> list[dict]:
    """OpenAI-style messages asking for one feature's explanation."""
    blocks = []
    for k, (tokens, acts) in enumerate(record.examples[:n_examples], start=1):
        norm = normalize_activations(acts)
        pairs = activation_pairs(tokens, norm)
        if not pairs:  # nothing crossed the threshold — the example carries no signal
            continue
        blocks.append(f"Example {k}: {highlight(tokens, norm)}\nActivations: {pairs}")

    body = "\n".join(blocks) if blocks else "(no activating examples available)"
    logits = ", ".join(f'"{t}"' for t in record.top_logits[:10])
    tail = f"\n\nTokens this feature promotes in the output: {logits}" if logits else ""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Here is a worked example.\n\n{FEW_SHOT}"},
        {"role": "assistant", "content": "Understood. Send the feature's examples."},
        {"role": "user", "content": f"{body}{tail}"},
    ]


def parse_explanation(text: str) -> str:
    """Take what follows `[EXPLANATION]:`, else the last line that reads like an answer.

    Returns `""` when the reply carries no usable explanation, which the caller treats as a
    miss and leaves the deterministic heuristic label in place.

    The old fallback took the last non-empty line unconditionally, and 4 of 40 Gemma labels
    came out as `- Ref`, `- Tokens that`, `-`: the model wrote a bulleted analysis, hit the
    token cap mid-bullet, and never reached the marker. A truncated bullet is not a worse
    label than the heuristic — it is a *misleading* one, because it reads like a finding.
    So a fallback line must not look like an enumeration item and must be a real phrase.
    """
    if not text:
        return ""
    marker = "[EXPLANATION]:"
    if marker in text:
        return text.split(marker)[-1].strip().strip('"').strip()

    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    # ONLY the last line is a candidate. Scanning upward for the first line that passes is
    # how "Let me analyse:" gets promoted over the bullets beneath it — if the reply ends
    # mid-enumeration, no line in it is the conclusion.
    last = lines[-1].strip('"').strip()
    if _BULLET.match(last):  # ended mid-enumeration: `- Ref`, `2. Tokens that`
        return ""
    if last.endswith(":"):  # a heading or preamble, not an explanation
        return ""
    return last if len(last.split()) >= MIN_FALLBACK_WORDS else ""
