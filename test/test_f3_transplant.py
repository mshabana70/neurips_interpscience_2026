"""CPU tests for the multi-layer refusal-profile transplant.

The hazard this file guards is the one that already cost F3 two rounds: a target measured
against one vector being imposed on another. Here it multiplies by 34 layers, because each
layer has its OWN direction and its OWN target and a single mis-pairing is silent.
"""

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from f3_transplant.run import (  # noqa: E402
    PROFILES,
    mediation_fraction,
    profile_targets,
    require_layers,
)


# --------------------------------------------------------------------- fixtures


@pytest.fixture
def multi_layer_env(monkeypatch):
    """A four-block stand-in: enough layers for a per-layer profile to be wrong."""
    import torch

    import jbct.interventions as I

    class Blk(torch.nn.Module):
        def forward(self, x):
            return x

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Blk() for _ in range(4)])

    monkeypatch.setattr(I, "get_layers", lambda m, s: list(m.layers))
    from jbct.registry import get_model_spec

    return M(), replace(get_model_spec("gemma-3-4b"), n_layers=4, outlier_dims=())


def _dirs_and_resid():
    """Per-layer directions that point along DIFFERENT axes.

    If the implementation reuses one layer's direction everywhere, the projections land on
    the wrong coordinate and the assertions below fail — which is the point.
    """
    import torch

    d = 8
    dirs = {}
    for layer in range(4):
        v = torch.zeros(d, dtype=torch.float64)
        v[layer] = float(layer + 2)      # distinct axis, deliberately not unit norm
        dirs[layer] = v
    h = torch.zeros(2, 5, d, dtype=torch.float64)
    for layer in range(4):
        h[..., layer] = 10.0 + layer     # a distinct starting projection per axis
    h[..., 7] = 3.0                      # an orthogonal component that must survive
    return dirs, h


# ------------------------------------------------------------------ ProfileClamp


def test_every_layer_lands_on_its_own_target(multi_layer_env):
    """The core property. Each layer clamps ITS projection onto ITS direction to ITS target."""
    from jbct.interventions import ProfileClamp

    model, spec = multi_layer_env
    dirs, h = _dirs_and_resid()
    targets = {0: -1.0, 1: 2.5, 2: 0.0, 3: 7.25}
    clamp = ProfileClamp(model, spec, dirs, targets)
    for layer in range(4):
        out = clamp._edit_at(h, layer)
        rhat = dirs[layer] / dirs[layer].norm()
        assert float(out[0, -1] @ rhat) == pytest.approx(targets[layer])


def test_using_one_layers_direction_everywhere_would_fail_this(multi_layer_env):
    """Guards the specific mis-pairing: layer 3's clamp must not move layer 0's coordinate."""
    import torch

    from jbct.interventions import ProfileClamp

    model, spec = multi_layer_env
    dirs, h = _dirs_and_resid()
    clamp = ProfileClamp(model, spec, dirs, {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0})
    out = clamp._edit_at(h, 3)
    # only coordinate 3 may move
    for coord in (0, 1, 2, 7):
        assert torch.allclose(out[..., coord], h[..., coord]), f"coord {coord} moved"


def test_orthogonal_content_survives(multi_layer_env):
    import torch

    from jbct.interventions import ProfileClamp

    model, spec = multi_layer_env
    dirs, h = _dirs_and_resid()
    clamp = ProfileClamp(model, spec, dirs, dict.fromkeys(range(4), -2.0))
    for layer in range(4):
        assert torch.allclose(clamp._edit_at(h, layer)[..., 7], h[..., 7])


def test_all_zero_targets_reproduce_directional_ablation(multi_layer_env):
    """The zero arm must be exactly ablation, or the ceiling it establishes is not the
    same quantity F2 measured."""
    import torch

    from jbct.interventions import DirectionalAblation, ProfileClamp

    model, spec = multi_layer_env
    dirs, h = _dirs_and_resid()
    prof = ProfileClamp(model, spec, dirs, dict.fromkeys(range(4), 0.0))
    for layer in range(4):
        abl = DirectionalAblation(model, spec, dirs[layer], layers=[layer])._edit(h)
        assert torch.allclose(prof._edit_at(h, layer), abl, atol=1e-12)


def test_directions_and_targets_must_cover_the_same_layers(multi_layer_env):
    from jbct.interventions import ProfileClamp

    model, spec = multi_layer_env
    dirs, _ = _dirs_and_resid()
    with pytest.raises(ValueError, match="same layers"):
        ProfileClamp(model, spec, dirs, {0: 1.0, 1: 2.0})


def test_a_layer_outside_the_model_is_rejected(multi_layer_env):
    import torch

    from jbct.interventions import ProfileClamp

    model, spec = multi_layer_env
    dirs, _ = _dirs_and_resid()
    dirs[99] = torch.ones(8, dtype=torch.float64)
    with pytest.raises(ValueError, match="out of range"):
        ProfileClamp(model, spec, dirs, dict.fromkeys(list(range(4)) + [99], 0.0))


def test_position_scoping_touches_only_the_named_token(multi_layer_env):
    """Same lesson as ProjectionClamp: the target is measured at ONE token."""
    import torch

    from jbct.interventions import ProfileClamp

    model, spec = multi_layer_env
    dirs, h = _dirs_and_resid()
    h = h.clone()
    h[:, 0, 0] = 99.0                              # a position with a very different value
    clamp = ProfileClamp(model, spec, dirs, dict.fromkeys(range(4), -1.0), position=-2)
    out = clamp._edit_at(h, 0)
    assert torch.allclose(out[:, 0, :], h[:, 0, :])          # untouched
    rhat = dirs[0] / dirs[0].norm()
    assert float(out[0, -2] @ rhat) == pytest.approx(-1.0)   # clamped


def test_on_a_generation_step_the_single_token_is_clamped(multi_layer_env):
    """During generation the sequence is length 1; the constraint must stay live."""
    from jbct.interventions import ProfileClamp

    model, spec = multi_layer_env
    dirs, h = _dirs_and_resid()
    step = h[:, -1:, :].clone()
    clamp = ProfileClamp(model, spec, dirs, dict.fromkeys(range(4), -1.0), position=-2)
    out = clamp._edit_at(step, 1)
    rhat = dirs[1] / dirs[1].norm()
    assert float(out[0, -1] @ rhat) == pytest.approx(-1.0)


def test_the_sham_moves_by_the_same_amount_without_changing_the_projection(multi_layer_env):
    """Exactly matched: shift computed from r_hat, written along u_hat. The arms differ in
    direction alone, so the refusal projection itself must be left where it started."""
    import torch

    from jbct.interventions import ProfileClamp

    model, spec = multi_layer_env
    dirs, h = _dirs_and_resid()
    shams = {}
    for layer in range(4):
        u = torch.zeros(8, dtype=torch.float64)
        u[7] = 1.0                                   # orthogonal to every real direction
        shams[layer] = u
    targets = dict.fromkeys(range(4), -1.0)
    real_rec: dict = {}
    sham_rec: dict = {}
    real = ProfileClamp(model, spec, dirs, targets, position=-1, record=real_rec)
    sham = ProfileClamp(model, spec, dirs, targets, position=-1, apply_along=shams,
                        record=sham_rec)
    for layer in range(4):
        r_out = real._edit_at(h, layer)
        s_out = sham._edit_at(h, layer)
        rhat = dirs[layer] / dirs[layer].norm()
        # the real arm lands on target; the sham leaves the projection untouched
        assert float(r_out[0, -1] @ rhat) == pytest.approx(targets[layer])
        assert float(s_out[0, -1] @ rhat) == pytest.approx(float(h[0, -1] @ rhat))
        # ...but both moved by the same distance
        assert (r_out[0, -1] - h[0, -1]).norm() == pytest.approx(
            (s_out[0, -1] - h[0, -1]).norm())
    assert real_rec.keys() == sham_rec.keys() == set(range(4))
    for layer in range(4):
        assert real_rec[layer] == pytest.approx(sham_rec[layer])


def test_the_record_holds_one_shift_per_prompt_per_layer(multi_layer_env):
    """`mean_shift_applied` was once a single sample under a plural name. Not again, and
    now it has to stay per-layer as well as per-prompt."""
    from jbct.interventions import ProfileClamp

    model, spec = multi_layer_env
    dirs, h = _dirs_and_resid()
    h = h.clone()
    h[0, -1, 0] = 4.0                       # two prompts at DIFFERENT projections
    h[1, -1, 0] = 6.0
    rec: dict = {}
    ProfileClamp(model, spec, dirs, dict.fromkeys(range(4), 0.0), position=-1,
                 record=rec)._edit_at(h, 0)
    assert len(rec[0]) == 2
    assert rec[0][0] == pytest.approx(-4.0)
    assert rec[0][1] == pytest.approx(-6.0)


# ------------------------------------------------------------------ target profiles


def _f1_report():
    """Two layers of a per_site_complement block, shaped like the real artifact."""
    return {
        "per_site_complement": {
            "0@-2": {"harmful": {"mean": 100.0, "n": 32},
                     "attack-COMPLY": {"mean": 40.0, "n": 196}},
            "1@-2": {"harmful": {"mean": 200.0, "n": 32},
                     "attack-COMPLY": {"mean": 20.0, "n": 196}},
        }
    }


def test_transplant_targets_are_the_per_layer_jailbroken_means():
    got = profile_targets(_f1_report(), position=-2, layers=[0, 1], profile="transplant")
    assert got == {0: pytest.approx(40.0), 1: pytest.approx(20.0)}


def test_null_targets_are_the_per_layer_harmful_means():
    got = profile_targets(_f1_report(), position=-2, layers=[0, 1], profile="null")
    assert got == {0: pytest.approx(100.0), 1: pytest.approx(200.0)}


def test_zero_targets_are_zero_everywhere():
    got = profile_targets(_f1_report(), position=-2, layers=[0, 1], profile="zero")
    assert got == {0: 0.0, 1: 0.0}


def test_every_profile_name_is_buildable():
    """PROFILES is what the runner iterates; a name without a builder would fail at run time
    after the model is already loaded."""
    for name in PROFILES:
        got = profile_targets(_f1_report(), position=-2, layers=[0, 1], profile=name)
        assert set(got) == {0, 1}


def test_an_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="unknown profile"):
        profile_targets(_f1_report(), position=-2, layers=[0], profile="wishful")


def test_a_layer_missing_from_f1_fails_loudly_rather_than_being_skipped():
    """Silently dropping a layer would make the transplant a partial profile under a name
    that claims otherwise."""
    with pytest.raises(SystemExit, match="2@-2"):
        require_layers(_f1_report(), position=-2, layers=[0, 1, 2])


def test_require_layers_passes_when_every_layer_is_present():
    require_layers(_f1_report(), position=-2, layers=[0, 1])


def test_a_position_f1_never_measured_fails_loudly():
    with pytest.raises(SystemExit, match="0@-7"):
        require_layers(_f1_report(), position=-7, layers=[0])


# ---------------------------------------------------------------------- reporting


def test_mediation_fraction_is_the_share_of_the_achievable_effect():
    """(transplant - null) / (zero - null): 1.0 means the jailbroken profile does everything
    ablation does, 0.0 means it does nothing."""
    assert mediation_fraction(null=0.0, transplant=0.4, zero=0.8) == pytest.approx(0.5)
    assert mediation_fraction(null=0.1, transplant=0.8, zero=0.8) == pytest.approx(1.0)
    assert mediation_fraction(null=0.1, transplant=0.1, zero=0.8) == pytest.approx(0.0)


def test_mediation_fraction_is_none_when_the_ceiling_is_degenerate():
    """If the zero arm achieved nothing over the null there is no scale to report against,
    and a ratio would be a division by ~0 dressed up as a result."""
    assert mediation_fraction(null=0.5, transplant=0.6, zero=0.5) is None


def test_mediation_fraction_can_exceed_one_and_is_not_clipped():
    """A transplant beating ablation is a real (surprising) outcome, not something to hide."""
    assert mediation_fraction(null=0.0, transplant=0.9, zero=0.6) == pytest.approx(1.5)


# ------------------------------------------------------- hook wiring (integration)
#
# The tests above call `_edit_at(h, layer)` directly, which proves the arithmetic but
# NOT that layer L's hook is the one registered on block L. A transposed mapping would
# pass every test above and silently clamp each layer to another layer's target.


def test_each_block_receives_its_own_layers_target(multi_layer_env):
    """Run a real forward pass through the context manager and read what each block emitted."""
    import torch

    from jbct.interventions import ProfileClamp

    model, spec = multi_layer_env
    dirs, h = _dirs_and_resid()
    targets = {0: -1.0, 1: 2.5, 2: 4.0, 3: 7.25}

    seen = {}
    with ProfileClamp(model, spec, dirs, targets) as clamp:
        # Registered INSIDE the context: forward hooks fire in registration order, so an
        # observer added first would capture the residual before the clamp rewrote it.
        for layer, blk in enumerate(model.layers):
            blk.register_forward_hook(
                lambda _m, _i, out, L=layer: seen.__setitem__(L, out.clone()))
        x = h
        for blk in model.layers:
            x = blk(x)
        assert clamp.n_calls == 4          # one call per layer, not one layer four times

    for layer in range(4):
        rhat = dirs[layer] / dirs[layer].norm()
        got = float(seen[layer][0, -1] @ rhat)
        assert got == pytest.approx(targets[layer]), (
            f"block {layer} emitted projection {got}, expected its own target "
            f"{targets[layer]} — hooks are mapped to the wrong layers")
    assert isinstance(seen[0], torch.Tensor)


def test_a_later_clamp_does_not_undo_an_earlier_one(multi_layer_env):
    """Layers run in sequence and each clamp rewrites the residual, so L3's edit acts on
    what L0-L2 already wrote. Every layer's constraint must still hold at its own output."""
    from jbct.interventions import ProfileClamp

    model, spec = multi_layer_env
    dirs, h = _dirs_and_resid()
    targets = {0: -5.0, 1: -5.0, 2: -5.0, 3: -5.0}
    outs = {}
    with ProfileClamp(model, spec, dirs, targets):
        for layer, blk in enumerate(model.layers):
            blk.register_forward_hook(
                lambda _m, _i, out, L=layer: outs.__setitem__(L, out.clone()))
        x = h
        for blk in model.layers:
            x = blk(x)
    # layer 0's coordinate must still read -5 at the FINAL output, since later layers
    # write along orthogonal axes
    rhat0 = dirs[0] / dirs[0].norm()
    assert float(outs[3][0, -1] @ rhat0) == pytest.approx(-5.0)


# ------------------------------------------------------------ suppression profile
#
# A mean of per-layer ratios reported 175% for qwen, because early layers carry
# near-zero projections (L9 harmful is 0.03) and the ratio there is noise divided by
# noise. The reported statistic has to be robust to that or it misleads before a
# single GPU-second is spent.


def test_suppression_is_magnitude_weighted_not_a_mean_of_ratios():
    """One tiny-magnitude layer with a wild ratio must not dominate the headline number."""
    from f3_transplant.run import suppression_profile

    targets = {
        "null":       {0: 0.03, 1: 100.0, 2: 200.0},
        "transplant": {0: 1.20, 1:  50.0, 2: 100.0},
    }
    got = suppression_profile(targets, [0, 1, 2])
    # weighted: (1.2 + 50 + 100) / (0.03 + 100 + 200) = 151.2 / 300.03
    assert got["weighted"] == pytest.approx(151.2 / 300.03)
    # a mean of per-layer ratios would be (40 + 0.5 + 0.5)/3 = 13.7 -- 1370%, the artifact
    assert got["weighted"] < 1.0
    assert got["median_per_layer"] == pytest.approx(0.5)


def test_suppression_ignores_layers_with_no_magnitude_when_taking_the_median():
    from f3_transplant.run import suppression_profile

    targets = {"null": {0: 0.0, 1: 10.0}, "transplant": {0: 0.0, 1: 5.0}}
    assert suppression_profile(targets, [0, 1])["median_per_layer"] == pytest.approx(0.5)


def test_suppression_is_none_when_there_is_no_magnitude_at_all():
    from f3_transplant.run import suppression_profile

    got = suppression_profile({"null": {0: 0.0}, "transplant": {0: 0.0}}, [0])
    assert got["weighted"] is None and got["median_per_layer"] is None


# --------------------------------------------------------- generation-time scope
#
# The first transplant run kept the clamp live on generation steps, forcing every
# generated token to the PROMPT's anchor-position mean at all 34 layers. Prompt-pass
# shift was only 64.45 against targets up to 8,157 -- the prompt was already there --
# yet 149/150 responses degenerated into repetition loops and 150/150 hit the token
# cap. Zeroing survives because ablation is position-agnostic; a nonzero target
# measured at one prompt position is not.


def test_prompt_only_leaves_a_generation_step_untouched(multi_layer_env):
    import torch

    from jbct.interventions import ProfileClamp

    model, spec = multi_layer_env
    dirs, h = _dirs_and_resid()
    step = h[:, -1:, :].clone()                      # a generation step: seq len 1
    clamp = ProfileClamp(model, spec, dirs, dict.fromkeys(range(4), 500.0),
                         position=-2, prompt_only=True)
    assert torch.allclose(clamp._edit_at(step, 0), step)


def test_prompt_only_still_clamps_the_prompt_pass(multi_layer_env):
    from jbct.interventions import ProfileClamp

    model, spec = multi_layer_env
    dirs, h = _dirs_and_resid()
    clamp = ProfileClamp(model, spec, dirs, dict.fromkeys(range(4), -1.0),
                         position=-2, prompt_only=True)
    out = clamp._edit_at(h, 0)
    rhat = dirs[0] / dirs[0].norm()
    assert float(out[0, -2] @ rhat) == pytest.approx(-1.0)


def test_the_default_still_clamps_generation_steps(multi_layer_env):
    """F3's single-layer runs depend on the sustained behaviour; the default must not
    change under them."""
    from jbct.interventions import ProfileClamp

    model, spec = multi_layer_env
    dirs, h = _dirs_and_resid()
    step = h[:, -1:, :].clone()
    clamp = ProfileClamp(model, spec, dirs, dict.fromkeys(range(4), -1.0), position=-2)
    rhat = dirs[0] / dirs[0].norm()
    assert float(clamp._edit_at(step, 0)[0, -1] @ rhat) == pytest.approx(-1.0)


def test_projection_clamp_gained_the_same_switch(multi_layer_env):
    """Same latent issue at one layer; keeping the two hooks in step is why the clamp
    arithmetic is shared."""
    import torch

    from jbct.interventions import ProjectionClamp

    model, spec = multi_layer_env
    dirs, h = _dirs_and_resid()
    step = h[:, -1:, :].clone()
    held = ProjectionClamp(model, spec, dirs[0], layer=0, target=500.0, position=-2,
                           prompt_only=True)
    assert torch.allclose(held._edit(step), step)


# --------------------------------------------------- degenerate-arm suppression


def _armset(null_incoh=0, transplant_incoh=0, zero_incoh=0):
    return {
        "baseline": {"n": 150, "comply": 0.113, "incoherent": 0},
        "null": {"n": 150, "comply": 0.0, "incoherent": null_incoh},
        "transplant": {"n": 150, "comply": 0.087, "incoherent": transplant_incoh},
        "zero": {"n": 150, "comply": 0.64, "incoherent": zero_incoh},
    }


def test_a_broken_arm_suppresses_the_mediation_figure():
    """The first run printed MEDIATION 13.5% from a null arm that was 149/150 repetition
    loops. A ratio off a destroyed reference reads as a measurement; it is not one."""
    from f3_transplant.run import degenerate_arms

    assert degenerate_arms(_armset(null_incoh=149)) == ["null"]


def test_arms_within_tolerance_do_not_suppress_it():
    from f3_transplant.run import degenerate_arms

    assert degenerate_arms(_armset(zero_incoh=4, transplant_incoh=1)) == []


def test_every_broken_arm_is_named_not_just_the_first():
    from f3_transplant.run import degenerate_arms

    assert degenerate_arms(_armset(null_incoh=149, zero_incoh=100)) == ["null", "zero"]


def test_an_arm_that_never_ran_is_not_called_degenerate():
    from f3_transplant.run import degenerate_arms

    arms = _armset()
    arms["zero"] = {"n": 0}
    assert degenerate_arms(arms) == []


# ------------------------------------------------------------- paired statistics
#
# The arms run the SAME 150 prompts, so an unpaired heuristic throws away the
# pairing and loses power. It mislabelled gemma's `zero` arm "NOT specific"
# (+0.120 real vs +0.053 sham) when the paired McNemar on the same data gives
# 14 gained / 4 lost, p = 0.031.


def test_mcnemar_counts_flips_in_both_directions():
    from f3_transplant.run import mcnemar_exact

    before = {"a": False, "b": False, "c": True, "d": False}
    after = {"a": True, "b": True, "c": False, "d": False}
    got = mcnemar_exact(before, after)
    assert got["gained"] == 2 and got["lost"] == 1


def test_mcnemar_is_two_sided_and_symmetric_under_swap():
    from f3_transplant.run import mcnemar_exact

    a = {str(i): i < 3 for i in range(10)}
    b = {str(i): i < 7 for i in range(10)}
    assert mcnemar_exact(a, b)["p"] == pytest.approx(mcnemar_exact(b, a)["p"])


def test_no_discordant_pairs_is_p_one_not_a_crash():
    from f3_transplant.run import mcnemar_exact

    same = {"a": True, "b": False}
    got = mcnemar_exact(same, dict(same))
    assert got["gained"] == 0 and got["lost"] == 0 and got["p"] == 1.0


def test_an_all_one_way_flip_is_significant():
    """gemma zero vs baseline was 18 gained / 0 lost — the shape that must register."""
    from f3_transplant.run import mcnemar_exact

    before = {str(i): False for i in range(150)}
    after = {str(i): i < 18 for i in range(150)}
    got = mcnemar_exact(before, after)
    assert got["gained"] == 18 and got["lost"] == 0 and got["p"] < 0.001


def test_only_prompts_present_in_both_arms_are_paired():
    from f3_transplant.run import mcnemar_exact

    got = mcnemar_exact({"a": True, "b": False}, {"a": False, "c": True})
    assert got["n_pairs"] == 1


def test_mediation_ci_is_deterministic_and_brackets_the_point_estimate():
    """36.8% with a 95% CI of [10.5%, 61.1%] is a different claim from 36.8% alone."""
    from f3_transplant.run import mediation_ci

    null = {str(i): i < 16 for i in range(150)}
    transplant = {str(i): i < 23 for i in range(150)}
    zero = {str(i): i < 35 for i in range(150)}
    a = mediation_ci(null, transplant, zero, n_boot=500, seed=0)
    b = mediation_ci(null, transplant, zero, n_boot=500, seed=0)
    assert a == b                                   # no global RNG
    assert a["lo"] <= a["point"] <= a["hi"]


def test_mediation_ci_reports_when_it_spans_zero():
    """qwen's 8.3% has a CI of [-7.7%, 25.0%]; 'includes zero' is the finding."""
    from f3_transplant.run import mediation_ci

    null = {str(i): i < 3 for i in range(150)}
    transplant = {str(i): i < 5 for i in range(150)}
    zero = {str(i): i < 27 for i in range(150)}
    got = mediation_ci(null, transplant, zero, n_boot=500, seed=0)
    assert got["excludes_zero"] is (got["lo"] > 0)


# ------------------------------------------------------------------- the scale sweep
#
# The transplant answers "does the jailbroken profile do anything" with no. The sweep
# turns that into a number: compliance requires retention <= X, and real jailbreaks
# reach only 29% (gemma) / 47% (qwen). That comparison is only meaningful if the swept
# arms and the jailbreak are measured on ONE ruler, which is what these tests pin.


def _targets():
    """Two layers, one of them negative, so a sign bug cannot hide behind abs()."""
    return {
        "null":       {0: -100.0, 1: 200.0},
        "transplant": {0:  -40.0, 1:  20.0},
        "zero":       {0:    0.0, 1:   0.0},
    }


def test_retention_scale_of_one_reproduces_the_null_arm_exactly():
    """s=1 must BE the null arm, not merely resemble it — the sweep interpolates between
    the design's own reference arms, and a drifting endpoint would bias every crossing."""
    from f3_transplant.run import scaled_targets

    got = scaled_targets(_targets(), [0, 1], 1.0, "retention")
    assert got == pytest.approx(_targets()["null"])


def test_retention_scale_of_zero_reproduces_the_zero_arm_exactly():
    from f3_transplant.run import scaled_targets

    assert scaled_targets(_targets(), [0, 1], 0.0, "retention") == pytest.approx(
        _targets()["zero"])


def test_retention_scaling_preserves_sign_at_every_layer():
    """gemma L0's harmful mean is -36.3. Scaling must shrink it toward zero, not flip it."""
    from f3_transplant.run import scaled_targets

    got = scaled_targets(_targets(), [0, 1], 0.5, "retention")
    assert got[0] == pytest.approx(-50.0)
    assert got[1] == pytest.approx(100.0)


def test_the_retention_of_a_retention_scaled_profile_is_the_scale_itself():
    """The ruler property. Without it the swept threshold and the jailbreak's own
    suppression would be quoted in different units while both reading as percentages."""
    from f3_transplant.run import scaled_targets, weighted_retention

    tgt = _targets()
    for scale in (0.0, 0.1, 0.35, 0.7, 1.0):
        prof = scaled_targets(tgt, [0, 1], scale, "retention")
        assert weighted_retention(prof, tgt["null"], [0, 1]) == pytest.approx(scale)


def test_jailbreak_axis_endpoints_are_the_null_and_transplant_arms():
    from f3_transplant.run import scaled_targets

    tgt = _targets()
    assert scaled_targets(tgt, [0, 1], 0.0, "jailbreak") == pytest.approx(tgt["null"])
    assert scaled_targets(tgt, [0, 1], 1.0, "jailbreak") == pytest.approx(tgt["transplant"])


def test_jailbreak_axis_extrapolates_past_the_jailbreak_along_its_own_shape():
    """s=2 is 'twice as far as the jailbreak went, in the direction it went' — the only
    way to ask whether the banded shape is right but too shallow."""
    from f3_transplant.run import scaled_targets

    got = scaled_targets(_targets(), [0, 1], 2.0, "jailbreak")
    assert got[0] == pytest.approx(20.0)     # -100 + 2*(-40 - -100)
    assert got[1] == pytest.approx(-160.0)   #  200 + 2*( 20 -  200)


def test_the_two_axes_differ_at_the_same_weighted_retention():
    """The load-bearing distinction. gemma's jailbreak keeps 7% of harmful at L14 and 50%
    at L15; a uniform scale set to the same OVERALL retention keeps the same share
    everywhere. Reading a uniform sweep as if it reproduced the jailbreak's shape would
    answer a question it never asked."""
    from f3_transplant.run import scaled_targets, weighted_retention

    tgt = _targets()
    jb = tgt["transplant"]
    r = weighted_retention(jb, tgt["null"], [0, 1])
    uniform = scaled_targets(tgt, [0, 1], r, "retention")
    assert weighted_retention(uniform, tgt["null"], [0, 1]) == pytest.approx(r)
    assert uniform[0] != pytest.approx(jb[0])       # same ruler, different profile


def test_an_unknown_scale_axis_is_rejected():
    from f3_transplant.run import scaled_targets

    with pytest.raises(ValueError, match="unknown scale axis"):
        scaled_targets(_targets(), [0], 0.5, "vibes")


def test_suppression_profile_and_the_sweep_share_one_retention_definition():
    """Two copies would be free to drift, and the paper's central comparison is exactly
    these two numbers on one axis."""
    from f3_transplant.run import suppression_profile, weighted_retention

    tgt = _targets()
    assert suppression_profile(tgt, [0, 1])["weighted"] == pytest.approx(
        weighted_retention(tgt["transplant"], tgt["null"], [0, 1]))


# ------------------------------------------------------------------ parsing the sweep


def test_scale_labels_trim_trailing_zeros_so_one_point_is_one_arm():
    from f3_transplant.run import scale_arm

    assert scale_arm(0.50) == scale_arm(0.5) == "scale@0.5"
    assert scale_arm(1.0) == "scale@1"


def test_duplicate_scales_collapse_to_one_arm():
    """0.50 and 0.5 write to the same generations file; keeping both would put two
    'measurements' at one retention on the curve."""
    from f3_transplant.run import parse_scales

    assert parse_scales("0.5,0.50,0.3") == (0.5, 0.3)


def test_scales_are_ordered_from_the_null_end_to_the_zero_end():
    from f3_transplant.run import parse_scales

    assert parse_scales("0.1,0.7,0.3") == (0.7, 0.3, 0.1)


def test_blank_and_empty_scale_specs_are_no_sweep():
    from f3_transplant.run import parse_scales

    assert parse_scales("") == () and parse_scales("  ,  ") == ()


def test_a_negative_scale_is_refused_rather_than_filed_under_its_magnitude():
    """weighted_retention takes an absolute value, so -0.3 would land at 30% next to +0.3
    — two different interventions sharing one x-coordinate."""
    from f3_transplant.run import parse_scales

    with pytest.raises(SystemExit, match="negative scale"):
        parse_scales("0.5,-0.3")


# ------------------------------------------------------------------- the curve itself


def _arms(**comply):
    base = {"baseline": {"n": 150, "comply": 0.113, "incoherent": 0}}
    for name, value in comply.items():
        label = name.replace("_", ".").replace("scale.", "scale@")
        base[label] = {"n": 150, "comply": value, "incoherent": 0}
    return base


def test_the_curve_puts_null_at_full_retention_and_zero_at_none():
    """They are curve POINTS, not separate references — which is what lets three interior
    scales bracket a crossing at all."""
    from f3_transplant.run import sweep_curve

    arms = _arms(null=0.10, zero=0.64)
    curve, _ = sweep_curve(arms, {"null": 1.0, "zero": 0.0})
    assert [p["arm"] for p in curve] == ["zero", "null"]
    assert curve[0]["retention"] == 0.0 and curve[1]["retention"] == 1.0


def test_a_degenerate_sweep_point_is_excluded_and_named():
    """An arm in a repetition loop scores 0.000 compliance, which on this curve is
    indistinguishable from 'restoring refusal worked' and would drag the threshold toward
    ablation."""
    from f3_transplant.run import sweep_curve

    arms = _arms(null=0.10, zero=0.64)
    arms["scale@0.5"] = {"n": 150, "comply": 0.0, "incoherent": 140}
    curve, excluded = sweep_curve(arms, {"null": 1.0, "zero": 0.0, "scale@0.5": 0.5})
    assert [p["arm"] for p in curve] == ["zero", "null"]
    assert [p["arm"] for p in excluded] == ["scale@0.5"]


def test_arms_that_never_ran_are_absent_from_the_curve():
    from f3_transplant.run import sweep_curve

    curve, _ = sweep_curve(_arms(null=0.10), {"null": 1.0, "zero": 0.0})
    assert [p["arm"] for p in curve] == ["null"]


# --------------------------------------------------------------------- the threshold


def _curve(*pairs):
    return [{"arm": f"scale@{r:g}", "retention": r, "comply": c, "incoherent": 0, "n": 150}
            for r, c in pairs]


def test_the_crossing_is_interpolated_between_the_bracketing_points():
    from f3_transplant.run import threshold_crossing

    got = threshold_crossing(_curve((0.0, 0.60), (0.5, 0.40), (1.0, 0.10)), level=0.30)
    # between retention 0.5 (0.40) and 1.0 (0.10): (0.40-0.30)/0.30 = 1/3 of the way
    assert got["retention"] == pytest.approx(0.5 + (1 / 3) * 0.5, abs=1e-3)
    assert got["between"] == ["scale@0.5", "scale@1"]


def test_a_crossing_exactly_on_a_swept_point_lands_on_that_point():
    from f3_transplant.run import threshold_crossing

    got = threshold_crossing(_curve((0.0, 0.60), (0.5, 0.40), (1.0, 0.10)), level=0.40)
    assert got["retention"] == pytest.approx(0.5)


def test_no_crossing_when_compliance_never_reaches_the_level():
    from f3_transplant.run import threshold_crossing

    got = threshold_crossing(_curve((0.0, 0.20), (1.0, 0.10)), level=0.50)
    assert got["retention"] is None and "never reaches" in got["note"]


def test_no_crossing_when_compliance_is_still_high_at_the_top_of_the_range():
    """Distinct from the case above: the threshold exists, the sweep just did not reach it,
    and the two must not read the same in a report."""
    from f3_transplant.run import threshold_crossing

    got = threshold_crossing(_curve((0.0, 0.80), (0.6, 0.70)), level=0.50)
    assert got["retention"] is None and "still above" in got["note"]


def test_a_non_monotone_curve_is_flagged_rather_than_smoothed():
    """Compliance is a proportion over 150 prompts, so a six-point curve wobbles. A
    crossing read off a wobbling curve is a number to distrust, not to put in an abstract."""
    from f3_transplant.run import threshold_crossing

    assert threshold_crossing(_curve((0.0, 0.60), (0.5, 0.20), (1.0, 0.45)),
                              level=0.30)["monotone"] is False
    assert threshold_crossing(_curve((0.0, 0.60), (0.5, 0.40), (1.0, 0.10)),
                              level=0.30)["monotone"] is True


def test_a_curve_with_one_usable_point_cannot_produce_a_crossing():
    from f3_transplant.run import threshold_crossing

    got = threshold_crossing(_curve((0.0, 0.60)), level=0.30)
    assert got["retention"] is None and got["monotone"] is None


# ------------------------------------------------- the bracket, read from disk labels


def _write_arm(outdir, arm, n_comply, n=150):
    import json

    rows = [{"prompt": f"p{i}", "response": "x", "coherent": True,
             "judge": "COMPLY" if i < n_comply else "REFUSE"} for i in range(n)]
    (outdir / f"{arm}_generations.json").write_text(json.dumps(rows))


def test_the_bracket_names_the_highest_retention_that_still_beats_null(tmp_path):
    """The sentence the sweep exists to produce, and unlike the interpolated crossing it
    assumes nothing about the curve being smooth."""
    from f3_transplant.run import sweep_curve, threshold_bracket

    _write_arm(tmp_path, "null", 15)
    _write_arm(tmp_path, "scale@0.7", 17)     # ~null
    _write_arm(tmp_path, "scale@0.3", 60)     # clearly above
    _write_arm(tmp_path, "zero", 96)
    arms = {"null": {"n": 150, "comply": 0.10, "incoherent": 0},
            "scale@0.7": {"n": 150, "comply": 0.113, "incoherent": 0},
            "scale@0.3": {"n": 150, "comply": 0.40, "incoherent": 0},
            "zero": {"n": 150, "comply": 0.64, "incoherent": 0}}
    curve, _ = sweep_curve(arms, {"null": 1.0, "scale@0.7": 0.7, "scale@0.3": 0.3,
                                  "zero": 0.0})
    got = threshold_bracket(tmp_path, curve)
    assert got["highest_retention_with_effect"] == pytest.approx(0.3)
    assert got["lowest_retention_without_effect"] == pytest.approx(0.7)
    assert got["bracket_consistent"] is True


def test_the_bracket_reports_when_the_ordering_is_impossible(tmp_path):
    """A LOWER retention showing no effect than one that did cannot happen for a real
    threshold; it means the curve is too noisy to quote."""
    from f3_transplant.run import sweep_curve, threshold_bracket

    _write_arm(tmp_path, "null", 15)
    _write_arm(tmp_path, "scale@0.7", 60)     # effect at HIGH retention
    _write_arm(tmp_path, "scale@0.3", 16)     # none at LOW retention
    arms = {"null": {"n": 150, "comply": 0.10, "incoherent": 0},
            "scale@0.7": {"n": 150, "comply": 0.40, "incoherent": 0},
            "scale@0.3": {"n": 150, "comply": 0.107, "incoherent": 0}}
    curve, _ = sweep_curve(arms, {"null": 1.0, "scale@0.7": 0.7, "scale@0.3": 0.3})
    assert threshold_bracket(tmp_path, curve)["bracket_consistent"] is False


def test_the_bracket_is_empty_without_a_null_arm_on_disk(tmp_path):
    from f3_transplant.run import threshold_bracket

    assert threshold_bracket(tmp_path, _curve((0.3, 0.4))) == {}


def test_an_arm_that_moves_compliance_the_wrong_way_does_not_count_as_an_effect(tmp_path):
    """A significant DROP is not evidence that restoring refusal induced compliance."""
    from f3_transplant.run import sweep_curve, threshold_bracket

    _write_arm(tmp_path, "null", 60)
    _write_arm(tmp_path, "scale@0.3", 10)
    arms = {"null": {"n": 150, "comply": 0.40, "incoherent": 0},
            "scale@0.3": {"n": 150, "comply": 0.067, "incoherent": 0}}
    curve, _ = sweep_curve(arms, {"null": 1.0, "scale@0.3": 0.3})
    assert threshold_bracket(tmp_path, curve)["highest_retention_with_effect"] is None


# ------------------------------------------------------------------------- the verdict


def test_the_verdict_says_too_shallow_when_the_jailbreak_sits_above_the_threshold():
    """gemma: jailbreak retains 29.1%, and if an effect needs <= 20% that is the finding."""
    from f3_transplant.run import threshold_verdict

    got = threshold_verdict(0.291, {"highest_retention_with_effect": 0.20,
                                    "bracket_consistent": True})
    assert "too shallow" in got and "29.1%" in got and "20.0%" in got


def test_the_verdict_says_sufficient_when_the_jailbreak_sits_below_the_threshold():
    """The outcome that would REVERSE the paper's claim has to be sayable by the same code."""
    from f3_transplant.run import threshold_verdict

    got = threshold_verdict(0.291, {"highest_retention_with_effect": 0.40,
                                    "bracket_consistent": True})
    assert "IS sufficient" in got


def test_the_verdict_refuses_to_quote_an_inconsistent_bracket():
    from f3_transplant.run import threshold_verdict

    got = threshold_verdict(0.291, {"highest_retention_with_effect": 0.7,
                                    "bracket_consistent": False})
    assert "not quotable" in got


def test_the_verdict_is_honest_when_no_arm_showed_an_effect():
    from f3_transplant.run import threshold_verdict

    assert "no threshold" in threshold_verdict(0.291, {})
    assert "no threshold" in threshold_verdict(None, {"highest_retention_with_effect": 0.2})


# ------------------------------------------------------- sweep arms reach the reporting


def test_arm_labels_finds_sweep_arms_and_excludes_shams_and_baseline(tmp_path):
    """Derived from disk so `--rescore` picks up a sweep whose report predates the flag."""
    from f3_transplant.run import arm_labels

    for arm in ("baseline", "null", "transplant", "zero", "scale@0.5", "sham_null",
                "sham_scale@0.5"):
        _write_arm(tmp_path, arm, 10)
    assert arm_labels(tmp_path) == ["null", "transplant", "zero", "scale@0.5"]


def test_the_paired_table_covers_sweep_arms(tmp_path):
    """A sweep arm silently absent would read as a clean three-profile run rather than a
    truncated sweep."""
    from f3_transplant.run import paired_report

    for arm in ("baseline", "null", "transplant", "zero"):
        _write_arm(tmp_path, arm, 15)
    _write_arm(tmp_path, "scale@0.3", 60)
    _write_arm(tmp_path, "sham_scale@0.3", 16)
    got = paired_report(tmp_path)
    assert "scale@0.3_vs_null" in got and "scale@0.3_vs_sham_scale@0.3" in got
    assert got["scale@0.3_vs_null"]["gained"] == 45
    assert "mediation_ci" in got                  # the core ratio still reported


def test_a_degenerate_sweep_arm_is_named_by_degenerate_arms():
    from f3_transplant.run import degenerate_arms

    arms = _armset()
    arms["scale@0.5"] = {"n": 150, "comply": 0.0, "incoherent": 140}
    arms["sham_scale@0.5"] = {"n": 150, "comply": 0.0, "incoherent": 150}
    assert degenerate_arms(arms) == ["scale@0.5"]


def test_the_full_sweep_block_reads_as_one_answer(tmp_path):
    """End to end on the shape the real run produces: curve, crossing, bracket, verdict."""
    from f3_transplant.run import sweep_block

    # The swept effect appears only at 20% retention; the jailbreak keeps 29.1%, so the
    # threshold sits BETWEEN them and the verdict must read "too shallow".
    plan = {"null": (1.0, 15, 0.10), "scale@0.5": (0.5, 17, 0.113),
            "scale@0.2": (0.2, 60, 0.40), "zero": (0.0, 96, 0.64)}
    arms = {}
    for arm, (_, n_comply, comply) in plan.items():
        _write_arm(tmp_path, arm, n_comply)
        arms[arm] = {"n": 150, "comply": comply, "incoherent": 0}
    got = sweep_block(arms, {a: r for a, (r, _, _) in plan.items()}, tmp_path,
                      jailbreak_retention=0.291)
    assert [p["arm"] for p in got["curve"]] == ["zero", "scale@0.2", "scale@0.5", "null"]
    assert got["crossings"]["0.5"]["retention"] is not None
    assert got["bracket"]["highest_retention_with_effect"] == pytest.approx(0.2)
    assert got["bracket"]["lowest_retention_without_effect"] == pytest.approx(0.5)
    assert "too shallow" in got["verdict"]


def test_a_resumed_run_still_sees_the_arms_it_did_not_regenerate(tmp_path):
    """`--arms scale@0.3` would otherwise fit a threshold on a curve with no null and no
    zero endpoint, and report it in the same field as a threshold from a complete run."""
    from f3_transplant.run import backfill_arms, sweep_curve

    _write_arm(tmp_path, "null", 15)
    _write_arm(tmp_path, "zero", 96)
    fresh = {"scale@0.3": {"n": 150, "comply": 0.40, "incoherent": 0}}
    got = backfill_arms(tmp_path, fresh)
    assert got["null"]["comply"] == pytest.approx(0.10)
    assert got["zero"]["comply"] == pytest.approx(0.64)
    assert got["null"]["from_disk"] is True
    assert "from_disk" not in got["scale@0.3"]        # this run's own arm is untouched
    curve, _ = sweep_curve(got, {"null": 1.0, "scale@0.3": 0.3, "zero": 0.0})
    assert [p["arm"] for p in curve] == ["zero", "scale@0.3", "null"]


def test_backfill_never_overwrites_an_arm_generated_this_run(tmp_path):
    from f3_transplant.run import backfill_arms

    _write_arm(tmp_path, "zero", 10)                  # stale: 6.7%
    fresh = {"zero": {"n": 150, "comply": 0.64, "incoherent": 0}}
    assert backfill_arms(tmp_path, fresh)["zero"]["comply"] == pytest.approx(0.64)
