import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jbct.models import format_prompt, get_layers
from jbct.registry import get_model_spec
from test.conftest import TINY_LLAMA


def test_format_prompt_applies_chat_template():
    tok = AutoTokenizer.from_pretrained(TINY_LLAMA)
    # tiny-random tokenizers may lack a chat template; set a deterministic one for the unit test
    tok.chat_template = (
        "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}"
    )
    spec = get_model_spec("llama-3.2-1b")
    out = format_prompt(spec, tok, "Hello")
    assert isinstance(out, str)
    assert "Hello" in out and "<|user|>" in out and out.endswith("<|assistant|>")


def test_get_layers_returns_all_blocks():
    model = AutoModelForCausalLM.from_pretrained(TINY_LLAMA)
    layers = get_layers(model)
    assert len(layers) == model.config.num_hidden_layers


def test_hookmanager_captures_all_layer_residuals():
    from jbct.models import HookManager

    model = AutoModelForCausalLM.from_pretrained(TINY_LLAMA)
    tok = AutoTokenizer.from_pretrained(TINY_LLAMA)
    ids = tok("Hello world", return_tensors="pt")
    spec = get_model_spec("llama-3.2-1b")
    with HookManager(model, spec) as hm:
        model(**ids)
    n = model.config.num_hidden_layers
    assert set(hm.residuals) == set(range(n))
    assert hm.residuals[0].shape[-1] == model.config.hidden_size


def test_hookmanager_matches_hidden_states():
    # HookManager captures each layer's RAW output residual. output_hidden_states[L+1] equals
    # that for every MIDDLE layer, but HF applies the final RMSNorm before appending the LAST
    # entry (hidden_states[n] == norm(last raw output)). Since direction extraction only ever
    # measures at a middle layer, the raw residual (what HookManager captures) is the correct,
    # un-normed value; the last-layer divergence is expected and asserted below.
    from jbct.models import HookManager

    model = AutoModelForCausalLM.from_pretrained(TINY_LLAMA)
    tok = AutoTokenizer.from_pretrained(TINY_LLAMA)
    ids = tok("Hello world", return_tensors="pt")
    with HookManager(model, get_model_spec("llama-3.2-1b")) as hm:
        out = model(**ids, output_hidden_states=True)
    n = model.config.num_hidden_layers
    for L in range(n - 1):  # middle layers: hook == hidden_states[L+1]
        assert torch.allclose(hm.residuals[L], out.hidden_states[L + 1])
    # last layer: hidden_states[n] is normed, so the raw hook output must differ
    assert not torch.allclose(hm.residuals[n - 1], out.hidden_states[n])
