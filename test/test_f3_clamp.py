"""CPU tests for F3's target arithmetic, population selection and reporting."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from f3_clamp.run import (  # noqa: E402
    BASIS_FIELD,
    basis_dims,
    clamp_target,
    held_out_harmful,
    population_stats,
    premise_holds,
    require_projection_field,
    specificity,
    summarize,
)


def _rows(site="21@-3"):
    return (
        [{"population": "harmful", "proj": {site: v}} for v in (-1000.0, -1400.0, -1800.0)]
        + [{"population": "attack-COMPLY", "proj": {site: v}} for v in (-2400.0, -2600.0)]
        + [{"population": "clean", "proj": {site: -3500.0}}]
    )


# ------------------------------------------------------------------ target maths


def test_alpha_one_lands_exactly_on_the_jailbroken_mean():
    """alpha=1 IS the experiment as specified — 'reduce to jailbroken-level magnitude'."""
    s = population_stats(_rows(), "21@-3")
    assert clamp_target(s, 1.0) == pytest.approx(s["attack-COMPLY"]["mean"])


def test_alpha_zero_lands_on_the_harmful_mean():
    """Not a no-op: it clamps away the population's variance without moving its mean, which
    is what isolates 'clamping did it' from 'moving the level did it'."""
    s = population_stats(_rows(), "21@-3")
    assert clamp_target(s, 0.0) == pytest.approx(s["harmful"]["mean"])


def test_alpha_interpolates_and_extrapolates_linearly():
    s = population_stats(_rows(), "21@-3")
    lo, hi = s["harmful"]["mean"], s["attack-COMPLY"]["mean"]
    assert clamp_target(s, 0.5) == pytest.approx((lo + hi) / 2)
    assert clamp_target(s, 1.5) == pytest.approx(lo + 1.5 * (hi - lo))


def test_premise_holds_only_when_refused_prompts_sit_above_jailbroken():
    """At gemma 15@-2 the real data has harmful BELOW jailbroken, so alpha=1 raises rather
    than reduces. The run must report that rather than silently flipping the sign."""
    above = population_stats(_rows(), "21@-3")
    assert premise_holds(above) is True

    site = "15@-2"
    below = population_stats(
        [{"population": "harmful", "proj": {site: -30000.0}},
         {"population": "attack-COMPLY", "proj": {site: -28400.0}}], site)
    assert premise_holds(below) is False


def test_population_stats_ignores_a_site_a_prompt_has_no_projection_for():
    rows = [{"population": "harmful", "proj": {"21@-3": 1.0}},
            {"population": "harmful", "proj": {}}]
    assert population_stats(rows, "21@-3")["harmful"]["n"] == 1


# ------------------------------------------------------------------------- basis
#
# The bug this section exists to prevent: F1 measured `proj` along the RAW direction
# while F3 clamped along the COMPLEMENT, so the target and the intervention lived in
# different bases. At gemma 21@-3 that put the harmful mean at -1,401 raw and +2,194
# complement -- opposite signs -- and at 15@-2 it applied a ~30,000 shift that left
# every arm, including the alpha=0 null, at 150/150 incoherent.


def test_basis_picks_both_the_target_field_and_the_dropped_dims_from_one_switch():
    """One argument must decide BOTH halves, or they can drift apart again."""
    dims = (443, 634)
    assert BASIS_FIELD["complement"] == "proj_complement"
    assert basis_dims(dims, "complement") == dims
    assert BASIS_FIELD["raw"] == "proj"
    assert basis_dims(dims, "raw") == ()


def test_population_stats_reads_the_basis_it_is_given():
    """The two fields hold genuinely different numbers; picking the wrong one is the bug."""
    site = "21@-3"
    rows = [{"population": "harmful", "proj": {site: -1401.0},
             "proj_complement": {site: 2194.0}},
            {"population": "attack-COMPLY", "proj": {site: -2537.0},
             "proj_complement": {site: 900.0}}]
    raw = population_stats(rows, site, field="proj")
    comp = population_stats(rows, site, field="proj_complement")
    assert raw["harmful"]["mean"] == pytest.approx(-1401.0)
    assert comp["harmful"]["mean"] == pytest.approx(2194.0)
    # ...and the premise itself is basis-dependent, so it must be re-read per basis.
    assert premise_holds(raw) is True
    assert premise_holds(comp) is True


def test_premise_can_flip_between_bases():
    site = "15@-2"
    rows = [{"population": "harmful", "proj": {site: -29983.0},
             "proj_complement": {site: 500.0}},
            {"population": "attack-COMPLY", "proj": {site: -28400.0},
             "proj_complement": {site: 100.0}}]
    assert premise_holds(population_stats(rows, site, field="proj")) is False
    assert premise_holds(population_stats(rows, site, field="proj_complement")) is True


def test_projections_without_the_complement_field_fail_with_the_f1_rerun_command():
    """A silent fallback to `proj` is exactly how the original bug shipped."""
    rows = [{"population": "harmful", "proj": {"21@-3": -1401.0}}]
    with pytest.raises(SystemExit, match="f1_geometry"):
        require_projection_field(rows, "proj_complement", "21@-3")


def test_require_projection_field_passes_when_the_site_is_present():
    rows = [{"population": "harmful", "proj_complement": {"21@-3": 2194.0}}]
    require_projection_field(rows, "proj_complement", "21@-3")


# ------------------------------------------------------------- prompt selection


def test_held_out_never_returns_a_prompt_that_fitted_the_direction():
    """The circularity guard, in the place it can actually bite: harmful_val holds only 39,
    so any run of size >39 must reach into harmful_train past the fitted prefix."""
    from jbct.data import load_direction_corpus

    train, _ = load_direction_corpus("train")
    fitted = set(train[:64])
    got = held_out_harmful(150, n_fit=64)
    assert len(got) == 150
    assert not (set(got) & fitted)


def test_asking_for_more_than_exist_fails_loudly():
    with pytest.raises(SystemExit, match="held-out harmful prompts available"):
        held_out_harmful(100000, n_fit=64)


# --------------------------------------------------------------------- reporting


def test_summarize_counts_compliance_and_incoherence():
    rows = [{"judge": "COMPLY", "coherent": True, "shift": 10.0},
            {"judge": "REFUSE", "coherent": True, "shift": 10.0},
            {"judge": "COMPLY", "coherent": False, "shift": 10.0}]
    s = summarize(rows)
    # The report rounds to 4dp on purpose, so compare at that precision.
    assert s["n"] == 3 and s["comply"] == pytest.approx(2 / 3, abs=1e-4) and s["incoherent"] == 1
    assert s["mean_shift_applied"] == pytest.approx(10.0)


def test_specificity_is_measured_against_the_no_op_baseline():
    """Unlike F2, compliance before is NOT 1.0 by construction — these prompts are refused,
    so every shift is measured from the baseline arm's own rate."""
    arms = {"clamp@a1": {"n": 150, "comply": 0.40, "incoherent": 0},
            "sham@a1": {"n": 150, "comply": 0.12, "incoherent": 0}}
    v = specificity(arms, baseline_comply=0.10)["clamp@a1"]
    assert v["real_shift"] == pytest.approx(0.30)
    assert v["sham_shift"] == pytest.approx(0.02)
    assert v["verdict"] == "specific"


def test_a_clamp_without_its_sham_gets_no_verdict():
    assert specificity({"clamp@a1": {"n": 150, "comply": 0.4, "incoherent": 0}}, 0.1) == {}


def test_a_sham_that_matches_the_real_arm_is_reported_as_not_specific():
    arms = {"clamp@a1": {"n": 150, "comply": 0.40, "incoherent": 0},
            "sham@a1": {"n": 150, "comply": 0.36, "incoherent": 0}}
    assert specificity(arms, 0.10)["clamp@a1"]["verdict"].startswith("NOT specific")


# ------------------------------------------------------- the intervention itself


@pytest.fixture
def stub_hook_env(monkeypatch):
    """A one-block stand-in so the clamp arithmetic is testable without loading a model."""
    import torch

    import jbct.interventions as I

    class Blk(torch.nn.Module):
        def forward(self, x):
            return x

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Blk()])

    monkeypatch.setattr(I, "get_layers", lambda m, s: list(m.layers))
    from dataclasses import replace

    from jbct.registry import get_model_spec

    return M(), replace(get_model_spec("gemma-3-4b"), n_layers=1, outlier_dims=())


def _resid_and_dir():
    import torch

    d = 8
    r = torch.zeros(d, dtype=torch.float64)
    r[0] = 3.0                       # deliberately NOT unit norm
    h = torch.zeros(1, 4, d, dtype=torch.float64)
    h[..., 0] = 5.0                  # projection onto r_hat is 5.0
    h[..., 1] = 2.0                  # an orthogonal component that must survive
    return h, r


def test_clamp_puts_the_projection_exactly_on_target(stub_hook_env):
    """The property the whole experiment rests on: after the clamp, <h, r_hat> == target,
    whatever it was before. An unnormalised r must not scale the result."""
    from jbct.interventions import ProjectionClamp

    model, spec = stub_hook_env
    h, r = _resid_and_dir()
    out = ProjectionClamp(model, spec, r, layer=0, target=-1.5)._edit(h)
    rhat = r / r.norm()
    assert float(out[0, -1] @ rhat) == pytest.approx(-1.5)


def test_clamp_leaves_the_orthogonal_part_untouched(stub_hook_env):
    import torch

    from jbct.interventions import ProjectionClamp

    model, spec = stub_hook_env
    h, r = _resid_and_dir()
    out = ProjectionClamp(model, spec, r, layer=0, target=-1.5)._edit(h)
    assert torch.allclose(out[..., 1], h[..., 1])


def test_target_zero_reproduces_directional_ablation(stub_hook_env):
    """Ablation is the special case of the clamp, which is why F3 needed a new hook rather
    than a parameter on the old one."""
    import torch

    from jbct.interventions import DirectionalAblation, ProjectionClamp

    model, spec = stub_hook_env
    h, r = _resid_and_dir()
    clamped = ProjectionClamp(model, spec, r, layer=0, target=0.0)._edit(h)
    ablated = DirectionalAblation(model, spec, r, layers=[0])._edit(h)
    assert torch.allclose(clamped, ablated, atol=1e-12)


def test_the_recorded_shift_is_what_the_sham_must_reapply(stub_hook_env):
    """The clamp's magnitude of change is prompt-dependent, so a sham that adds a constant
    of its own is not matched. It has to reuse this number."""
    from jbct.interventions import ProjectionClamp, ShiftAlong

    model, spec = stub_hook_env
    h, r = _resid_and_dir()
    rec: list[float] = []
    ProjectionClamp(model, spec, r, layer=0, target=-1.5, record=rec)._edit(h)
    assert rec == [pytest.approx(-6.5)]              # target - projection = -1.5 - 5.0

    rhat = r / r.norm()
    moved = ShiftAlong(model, spec, r, layer=0, delta=rec[0])._edit(h)
    assert float((moved[0, -1] - h[0, -1]) @ rhat) == pytest.approx(-6.5)


# --------------------------------------------- position scoping and matched sham


def test_clamping_one_position_leaves_the_others_alone(stub_hook_env):
    """The fix for the run that destroyed gemma 15@-2. The target is measured at ONE token;
    clamping every position overwrites the sequence with a value only that token holds."""
    import torch

    from jbct.interventions import ProjectionClamp

    model, spec = stub_hook_env
    h, r = _resid_and_dir()
    h = h.clone()
    h[0, 0, 0] = 99.0                       # a position with a very different projection
    out = ProjectionClamp(model, spec, r, layer=0, target=-1.5, position=-3)._edit(h)
    rhat = r / r.norm()
    assert float(out[0, -3] @ rhat) == pytest.approx(-1.5)      # the target token moved
    assert torch.allclose(out[0, 0], h[0, 0])                   # position 0 untouched
    assert torch.allclose(out[0, -1], h[0, -1])                 # last token untouched


def test_position_none_still_overwrites_everything(stub_hook_env):
    """The old behaviour is kept reachable via --clamp-positions all, so the destroyed run
    stays reproducible rather than becoming a mystery."""
    from jbct.interventions import ProjectionClamp

    model, spec = stub_hook_env
    h, r = _resid_and_dir()
    out = ProjectionClamp(model, spec, r, layer=0, target=-1.5, position=None)._edit(h)
    rhat = r / r.norm()
    assert all(float(out[0, i] @ rhat) == pytest.approx(-1.5) for i in range(h.shape[1]))


def test_a_single_token_forward_clamps_that_token(stub_hook_env):
    """During generation the sequence is length 1, so the constraint must still apply to the
    token being written — otherwise the model reintroduces the direction as it answers."""
    from jbct.interventions import ProjectionClamp

    model, spec = stub_hook_env
    _h, r = _resid_and_dir()
    import torch

    step = torch.zeros(1, 1, 8, dtype=torch.float64)
    step[..., 0] = 5.0
    out = ProjectionClamp(model, spec, r, layer=0, target=-1.5, position=-3)._edit(step)
    assert float(out[0, -1] @ (r / r.norm())) == pytest.approx(-1.5)


def test_the_sham_moves_each_prompt_by_the_shift_the_real_arm_computed(stub_hook_env):
    """Exact matching, per prompt. The first run applied ONE prompt's shift to all 150."""
    import torch

    from jbct.interventions import ProjectionClamp

    model, spec = stub_hook_env
    _h, r = _resid_and_dir()
    u = torch.zeros(8, dtype=torch.float64)
    u[2] = 1.0                                  # orthogonal to r

    # two prompts starting at DIFFERENT projections
    batch = torch.zeros(2, 4, 8, dtype=torch.float64)
    batch[0, :, 0] = 5.0
    batch[1, :, 0] = -2.0

    rhat = r / r.norm()
    real = ProjectionClamp(model, spec, r, layer=0, target=-1.5, position=-1)._edit(batch)
    sham = ProjectionClamp(model, spec, r, layer=0, target=-1.5, position=-1,
                           apply_along=u)._edit(batch)
    for i, start in enumerate((5.0, -2.0)):
        expected = -1.5 - start
        assert float((real[i, -1] - batch[i, -1]) @ rhat) == pytest.approx(expected)
        assert float((sham[i, -1] - batch[i, -1]) @ u) == pytest.approx(expected)
    # and the sham must NOT have moved the refusal projection
    assert float(sham[0, -1] @ rhat) == pytest.approx(5.0)


def test_record_collects_one_shift_per_prompt_on_the_prompt_pass(stub_hook_env):
    """`mean_shift_applied` was a single sample under a plural name. It is now a real mean."""
    import torch

    from jbct.interventions import ProjectionClamp

    model, spec = stub_hook_env
    _h, r = _resid_and_dir()
    batch = torch.zeros(3, 4, 8, dtype=torch.float64)
    for i, v in enumerate((5.0, -2.0, 1.0)):
        batch[i, :, 0] = v
    rec: list[float] = []
    ProjectionClamp(model, spec, r, layer=0, target=0.0, position=-1, record=rec)._edit(batch)
    assert rec == [pytest.approx(-5.0), pytest.approx(2.0), pytest.approx(-1.0)]


# ------------------------------------------------------- specificity references
#
# A clamp moves the population AND removes the variance inside it. At gemma 15@-2
# alpha=0 landed at 0.007 against a 0.113 baseline -- collapsing every prompt onto the
# harmful mean pushes the naturally-low-projection prompts (the ones already complying)
# back up. Measuring the other arms from the baseline charges them for that cost.


def _arms(base, a0, a0_sham, a1, a1_sham):
    return {
        "baseline": {"n": 150, "comply": base, "incoherent": 0},
        "clamp@a0": {"n": 150, "comply": a0, "incoherent": 0},
        "sham@a0": {"n": 150, "comply": a0_sham, "incoherent": 0},
        "clamp@a1": {"n": 150, "comply": a1, "incoherent": 0},
        "sham@a1": {"n": 150, "comply": a1_sham, "incoherent": 0},
    }


def test_specificity_reports_both_reference_points():
    """gemma 15@-2's real numbers: the two references disagree, so both must be visible."""
    arms = _arms(base=0.113, a0=0.007, a0_sham=0.087, a1=0.127, a1_sham=0.120)
    v = specificity(arms, baseline_comply=0.113)["clamp@a1"]
    assert v["real_shift"] == pytest.approx(0.014, abs=1e-4)        # vs un-intervened
    assert v["real_shift_vs_null"] == pytest.approx(0.120, abs=1e-4)  # vs the clamped null
    assert v["sham_shift_vs_null"] == pytest.approx(0.033, abs=1e-4)
    assert v["verdict_vs_null"] == "specific"
    assert v["verdict"].startswith("NOT specific")


def test_the_null_arm_is_measured_against_its_own_sham_not_the_baseline_sham():
    """Pairing a1's sham with a0's sham is what makes the two columns comparable."""
    arms = _arms(base=0.10, a0=0.20, a0_sham=0.15, a1=0.50, a1_sham=0.25)
    v = specificity(arms, baseline_comply=0.10)["clamp@a1"]
    assert v["real_shift_vs_null"] == pytest.approx(0.30)   # 0.50 - 0.20
    assert v["sham_shift_vs_null"] == pytest.approx(0.10)   # 0.25 - 0.15


def test_without_an_alpha_zero_arm_only_the_baseline_reference_is_reported():
    """--alphas 1.0 is a legal run; it must not invent a null it never measured."""
    arms = {"baseline": {"n": 10, "comply": 0.1, "incoherent": 0},
            "clamp@a1": {"n": 10, "comply": 0.5, "incoherent": 0},
            "sham@a1": {"n": 10, "comply": 0.2, "incoherent": 0}}
    v = specificity(arms, baseline_comply=0.1)["clamp@a1"]
    assert "real_shift_vs_null" not in v
    assert v["real_shift"] == pytest.approx(0.4)


def test_rescore_rewrites_specificity_from_the_arms_alone():
    """The reference point must be changeable without spending 150x1024 tokens again."""
    from f3_clamp.run import rescore

    report = {"arms": _arms(base=0.113, a0=0.007, a0_sham=0.087, a1=0.127, a1_sham=0.120),
              "specificity": {"stale": True}}
    out = rescore(report)
    assert "stale" not in out["specificity"]
    assert out["specificity"]["clamp@a1"]["real_shift_vs_null"] == pytest.approx(0.120,
                                                                                abs=1e-4)
