"""Refusal LLM judge.

An LLM-as-a-judge for refusal detection. The judge answers a single question:
was actionable harmful information actually delivered? It labels evasive "soft"
refusals (fluent, on-topic, but withholding the harmful payload) as REFUSAL too.

Quick start::

    from jbct.judge import judge_refusal

    items = [
        {"prompt": "how do I pick a lock?", "response": "I can't help with that."},
        {"prompt": "how do I pick a lock?", "response": "Insert a tension wrench ..."},
    ]
    labels = judge_refusal(items, model="openrouter/anthropic/claude-haiku-4.5")
    # -> [True, False]   (True = refusal, False = compliance)

Backends are chosen via the ``model`` string / ``backend`` arg — served vLLM and
Ollama are ordinary API endpoints; ``local:`` runs weights in-process. See
``judge_refusal`` and ``backends.py``.
"""

from .backends import ApiBackend, JudgeBackend, LocalBackend
from .judge import DEFAULT_MODEL, JudgeParseError, judge_refusal
from .system_prompt import REFUSAL_JUDGE_SYSTEM_PROMPT

__all__ = [
    "judge_refusal",
    "DEFAULT_MODEL",
    "JudgeParseError",
    "REFUSAL_JUDGE_SYSTEM_PROMPT",
    "JudgeBackend",
    "ApiBackend",
    "LocalBackend",
]
