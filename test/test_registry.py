import pytest

from jbct.registry import (
    DELEGATED_GPU_VALIDATION,
    LOCAL_GPU_MODELS,
    MODELS,
    Physics,
    get_model_spec,
)


def test_llama_present_and_shaped():
    s = get_model_spec("llama-3.2-1b")
    assert s.hf_id == "meta-llama/Llama-3.2-1B-Instruct"
    assert s.family == "llama"
    assert s.n_layers == 16 and s.d_model == 2048
    assert s.tier == 1


def test_llama_physics_is_placeholder():
    s = get_model_spec("llama-3.2-1b")
    assert isinstance(s.causal_layer, Physics)
    assert s.causal_layer.provenance == "placeholder"


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        get_model_spec("gpt-5")


def test_registry_is_populated():
    assert "llama-3.2-1b" in MODELS


@pytest.mark.parametrize(
    "name,family,layers",
    [
        ("gemma-3-4b", "gemma", 34),
        ("qwen3-4b", "qwen", 36),
        ("qwen3-14b", "qwen", 40),
        ("qwen3-0.6b", "qwen", 28),
    ],
)
def test_fanout_models_present(name, family, layers):
    s = get_model_spec(name)
    assert s.family == family and s.n_layers == layers and s.tier == 1


def test_gemma_physics_verified_qwen_placeholder():
    assert get_model_spec("gemma-3-4b").causal_layer.provenance == "verified"
    assert get_model_spec("gemma-3-4b").causal_layer.value == 15
    assert get_model_spec("qwen3-14b").causal_layer.provenance == "placeholder"


def test_tiny_qwen_is_the_local_slice_model():
    """Qwen3-0.6B replaces Llama-3.2-1B as the fast local GPU slice target (ungated)."""
    s = get_model_spec("qwen3-0.6b")
    assert s.hf_id == "Qwen/Qwen3-0.6B"
    assert s.d_model == 1024  # verified against the published config.json
    assert s.chat_template_kwargs == {"enable_thinking": False}  # R4, load-bearing
    assert "qwen3-0.6b" in LOCAL_GPU_MODELS


def test_gpu_validation_partition_is_total_and_disjoint():
    """Every registry model is either locally validatable or explicitly delegated — no gaps."""
    local, delegated = set(LOCAL_GPU_MODELS), set(DELEGATED_GPU_VALIDATION)
    assert local | delegated == set(MODELS)
    assert not (local & delegated)


def test_untestable_models_are_delegated_with_an_owner_and_reason():
    """A green local suite must never imply Llama-3.2-1B or Qwen3-14B were GPU-validated."""
    assert set(DELEGATED_GPU_VALIDATION) == {"llama-3.2-1b", "qwen3-14b"}
    for d in DELEGATED_GPU_VALIDATION.values():
        assert d.owner and d.reason


def test_outlier_dims_record_what_was_measured():
    """Empty tuple means NOT MEASURED. Once measured, the result must be recorded."""
    gemma, qwen = get_model_spec("gemma-3-4b"), get_model_spec("qwen3-4b")
    assert gemma.outlier_dims == (443, 634)  # 0.81 of the L15 direction sits here
    assert qwen.outlier_dims == (0, 3, 4)    # exist, but only ~0.02 of the direction
    # unmeasured models stay empty rather than being assumed clean
    assert get_model_spec("qwen3-14b").outlier_dims == ()
