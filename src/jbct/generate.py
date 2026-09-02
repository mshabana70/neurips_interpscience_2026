"""H2 primitive: generate model responses to attack / control prompts.

Four settings here are load-bearing, each one a bug Refusal-Lens already paid for:

1. **No truncation, and a generous cap.** Storing `resp[:300]` and capping at 80 new tokens
   "was the bug that had been masking real jailbreaks" (old `eval_loop_README.md`): a
   slow-ramp payload preambles for 200 tokens, so a clipped response reads as a refusal.
   Default `max_new_tokens=1024`, full text kept, and `hit_token_cap` is *recorded* so a
   response that ran out of room is visible instead of being silently judged as-is.
2. **Single BOS.** The chat template already emits one; re-tokenizing with
   `add_special_tokens=True` adds a second (see `tokenize_prompt`). The old generator had
   this too (`generate_v3.py:48`).
3. **Greedy decoding.** `do_sample=False` — ASR must be reproducible run-to-run.
4. **`enable_thinking=False` for Qwen**, carried by `spec.chat_template_kwargs`. With
   thinking on, Qwen3 spends its budget in a `<think>` scaffold and the answer never
   appears — which reads as a refusal.

Prefill: an AdvPrefix-style attack opens the assistant turn with fixed text. The judge must
see the *whole* assistant message, so the prefill is prepended to the continuation
(matching old `dataset_v5/generate.py:55`), not dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import count_leading_bos, format_prompt
from .registry import ModelSpec


@dataclass
class GenResult:
    prompt_text: str
    response: str
    n_prompt_tokens: int
    n_new_tokens: int
    hit_token_cap: bool
    prefill: str | None = None
    meta: dict = field(default_factory=dict)


def _stop_token_ids(model, tokenizer) -> set[int]:
    """Every id that ends a turn, plus the pad id.

    `tokenizer.eos_token_id` alone is WRONG for chat models whose turn terminator differs
    from `<eos>`: Gemma-3 ends an assistant turn with `<end_of_turn>` (106) and declares
    `generation_config.eos_token_id = [1, 106]`. `generate` honors the list, so a naturally
    finished row would look unterminated to us — and since only the LONGEST row in a batch
    carries no padding, that row gets mislabeled `hit_token_cap=True`. That flag is the
    guard against the truncation bug that masked real jailbreaks in Refusal-Lens, so a
    false positive there is expensive: it sends you chasing a cap that wasn't hit.
    """
    ids: set[int] = set()
    gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if isinstance(gen_eos, int):
        ids.add(gen_eos)
    elif gen_eos:
        ids.update(int(t) for t in gen_eos)
    for tok_id in (tokenizer.eos_token_id, tokenizer.pad_token_id):
        if tok_id is not None:
            ids.add(int(tok_id))
    return ids


def build_input(spec: ModelSpec, tokenizer, text: str, prefill: str | None = None) -> str:
    """Chat-templated prompt; ``prefill`` opens the assistant turn when present."""
    prompt = format_prompt(spec, tokenizer, text)
    return prompt + prefill if prefill else prompt


def generate(
    model,
    tokenizer,
    spec: ModelSpec,
    texts: list[str],
    prefills: list[str | None] | None = None,
    max_new_tokens: int = 1024,
    batch_size: int = 1,
) -> list[GenResult]:
    """Greedy-generate one response per text. ``prefills[i]`` opens turn ``i`` if given."""
    import torch

    prefills = prefills or [None] * len(texts)
    if len(prefills) != len(texts):
        raise ValueError(f"prefills ({len(prefills)}) must align with texts ({len(texts)})")

    # Left padding is mandatory when batching: with right padding the generation would
    # continue from PAD tokens instead of the end of the prompt.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    results: list[GenResult] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        chunk_prefills = prefills[start : start + batch_size]
        prompts = [build_input(spec, tokenizer, t, p) for t, p in zip(chunk, chunk_prefills)]

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,  # the template already emitted BOS
        ).to(model.device)
        plen = enc["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )

        stop = _stop_token_ids(model, tokenizer)
        for i, (text, prefill) in enumerate(zip(chunk, chunk_prefills)):
            new_ids = out[i][plen:].tolist()
            # In a batch, `generate` pads finished rows out to the longest row, so the raw
            # slice length is the BATCH's length, not this row's. Cut at the first stop
            # token to recover the true generated length.
            gen_len = next((j for j, t in enumerate(new_ids) if t in stop), len(new_ids))
            finished = gen_len < len(new_ids)  # a stop token was actually emitted
            cont = tokenizer.decode(new_ids[:gen_len], skip_special_tokens=True)

            # BOS check must see the UNPADDED prompt: with left padding the row starts with
            # pad tokens, so counting from index 0 would report 0 regardless of duplication.
            n_real = int(enc["attention_mask"][i].sum())
            real_ids = enc["input_ids"][i][-n_real:].unsqueeze(0)

            results.append(
                GenResult(
                    prompt_text=text,
                    # the judge scores the FULL assistant message, prefill included
                    response=(prefill + cont) if prefill else cont,
                    n_prompt_tokens=n_real,
                    n_new_tokens=gen_len,
                    hit_token_cap=(not finished) and gen_len >= max_new_tokens,
                    prefill=prefill,
                    meta={"n_leading_bos": count_leading_bos(tokenizer, real_ids)},
                )
            )
    return results
