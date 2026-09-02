import pytest

from jbct.provenance import env_versions


def test_records_the_versions_that_can_move_a_number():
    env = env_versions()
    for key in ("python", "platform", "torch", "transformers", "cuda"):
        assert key in env
    assert env["torch"] and env["transformers"]  # both are installed here


def test_missing_optional_deps_are_recorded_as_none_not_raised():
    """transformer_lens/nnsight are H4-only; their absence is data, not a crash."""
    env = env_versions()
    for key in ("transformer_lens", "nnsight", "huggingface_hub"):
        assert key in env  # present as a key even when the import fails


def test_transformers_is_capped_only_where_circuit_tracer_is_installed():
    """The cap binds the H4 environment, not every dev box.

    circuit-tracer pins transformers<=4.57.3. Asserting that unconditionally would fail on
    any machine running only H1-H3, so the check keys off whether TransformerLens is actually
    present — i.e. whether this IS an H4 environment.
    """
    env = env_versions()
    if env["transformer_lens"] is None:
        pytest.skip("not an H4 environment (transformer_lens absent); cap does not apply")
    major, minor = (int(x) for x in env["transformers"].split(".")[:2])
    assert (major, minor) <= (4, 57), (
        f"transformers {env['transformers']} exceeds circuit-tracer's 4.57.3 cap"
    )
