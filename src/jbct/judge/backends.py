"""Inference backends for the refusal LLM judge.

There are only two execution modes, and the judge logic in ``judge.py`` is
written once on top of this thin interface:

- ``ApiBackend``   -> any OpenAI-compatible HTTP endpoint, via LiteLLM. This
  covers OpenRouter, OpenAI, Anthropic, *and* locally-served engines (a vLLM
  ``--api-server`` or an Ollama daemon are just API services). Pick the provider
  with the ``model`` string (``"openrouter/..."``, ``"gpt-4o-mini"``,
  ``"anthropic/claude-opus-4-8"``, ``"ollama/llama3"``) and, for a local server,
  ``api_base``.
- ``LocalBackend``  -> weights loaded in-process on your GPU with no server, via
  vLLM offline (``vllm.LLM``) or HF ``transformers``. Use this only when you
  want to avoid standing up an HTTP server.

Heavy dependencies (litellm / vllm / torch / transformers) are imported lazily
inside each backend so that ``import jbct.judge`` stays cheap.
"""

from __future__ import annotations

from typing import Protocol

Messages = list[dict]  # OpenAI-style chat messages: [{"role", "content"}, ...]


class JudgeBackend(Protocol):
    """A backend turns a batch of chat conversations into raw text completions."""

    def generate(self, conversations: list[Messages]) -> list[str]:
        """Return one raw completion string per conversation (order preserved)."""
        ...


class ApiBackend:
    """OpenAI-compatible API backend (OpenRouter / OpenAI / Anthropic / served
    vLLM / Ollama) via LiteLLM.

    Concurrency is handled by LiteLLM's ``batch_completion`` thread pool; the
    judge chunks work so the pool size equals the configured batch size.
    """

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        api_base: str | None = None,
        api_key: str | None = None,
        **completion_kwargs,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_base = api_base
        self.api_key = api_key
        # The judge is a one-word classification task: a reasoning trace is pure
        # waste and (on Ollama) consumes the token budget before the label is
        # emitted, leaving `content` empty. Disable thinking for Ollama models
        # unless the caller explicitly set `think`.
        if model.startswith(("ollama/", "ollama_chat/")) and "think" not in completion_kwargs:
            completion_kwargs = {**completion_kwargs, "think": False}
        self.completion_kwargs = completion_kwargs

    def generate(self, conversations: list[Messages]) -> list[str]:
        import litellm

        responses = litellm.batch_completion(
            model=self.model,
            messages=conversations,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_base=self.api_base,
            api_key=self.api_key,
            **self.completion_kwargs,
        )

        out: list[str] = []
        for i, r in enumerate(responses):
            # batch_completion returns the raised exception in-place on failure.
            # Do NOT swallow it — a dead backend must surface, not be mislabeled.
            if isinstance(r, Exception):
                raise RuntimeError(
                    f"judge backend call failed for item {i} "
                    f"(model={self.model!r}, api_base={self.api_base!r}): {r}"
                ) from r
            try:
                out.append(r.choices[0].message.content or "")
            except (AttributeError, IndexError, KeyError) as e:
                raise RuntimeError(
                    f"judge backend returned an unreadable response for item {i}: {r!r}"
                ) from e
        return out


class LocalBackend:
    """In-process GPU backend (no HTTP server).

    ``engine="vllm"`` uses vLLM offline batched generation (fast).
    ``engine="transformers"`` uses HF ``transformers`` (already a repo dep, but
    slower). Both apply the model's own chat template.
    """

    def __init__(
        self,
        model: str,
        *,
        engine: str = "vllm",
        temperature: float = 0.0,
        max_tokens: int = 512,
        **engine_kwargs,
    ):
        self.model = model
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.engine_kwargs = engine_kwargs
        self._llm = None          # vllm.LLM
        self._hf_model = None      # transformers model
        self._tokenizer = None

    # -- vLLM offline ----------------------------------------------------------
    def _ensure_vllm(self):
        if self._llm is None:
            from vllm import LLM

            self._llm = LLM(model=self.model, **self.engine_kwargs)

    def _generate_vllm(self, conversations: list[Messages]) -> list[str]:
        from vllm import SamplingParams

        self._ensure_vllm()
        sp = SamplingParams(temperature=self.temperature, max_tokens=self.max_tokens)
        outputs = self._llm.chat(conversations, sampling_params=sp, use_tqdm=False)
        return [o.outputs[0].text for o in outputs]

    # -- HF transformers -------------------------------------------------------
    def _ensure_hf(self):
        if self._hf_model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.padding_side = "left"  # for decoder-only batched gen
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                self.model,
                torch_dtype=getattr(torch, "bfloat16", None),
                device_map="auto",
                **self.engine_kwargs,
            )

    def _generate_hf(self, conversations: list[Messages]) -> list[str]:
        import torch

        self._ensure_hf()
        tok = self._tokenizer
        prompts = [
            tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
            for c in conversations
        ]
        enc = tok(prompts, return_tensors="pt", padding=True).to(self._hf_model.device)
        with torch.no_grad():
            gen = self._hf_model.generate(
                **enc,
                max_new_tokens=self.max_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=tok.pad_token_id,
            )
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        return tok.batch_decode(new_tokens, skip_special_tokens=True)

    def generate(self, conversations: list[Messages]) -> list[str]:
        if self.engine == "vllm":
            return self._generate_vllm(conversations)
        if self.engine == "transformers":
            return self._generate_hf(conversations)
        raise ValueError(f"unknown local engine: {self.engine!r}")
