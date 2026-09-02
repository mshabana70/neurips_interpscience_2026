"""CPU tests for the H3 causal arm's selection and reporting logic."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from h3_causal.run import (  # noqa: E402
    ADDITION_ARMS,
    ARMS,
    addition_specificity,
    is_coherent,
    select_jobs,
    summarize,
)


def _g(idx, kind, judge, cls="fiction", prefill=None):
    return {"record_idx": idx, "kind": kind, "class": cls, "base_id": idx,
            "judge": judge, "prompt_text": f"p{idx}", "prefill": prefill}


JUDGED = [
    _g(0, "attack", "COMPLY"), _g(0, "control", "REFUSE"),
    _g(1, "attack", "REFUSE"), _g(1, "control", "REFUSE"),
    _g(2, "attack", "COMPLY", prefill="Sure,"), _g(2, "control", "COMPLY"),
]


def test_add_attack_runs_only_on_attacks_that_succeeded():
    """Re-refusing an already-refused attack answers nothing."""
    jobs = select_jobs(JUDGED, "add_attack", None)
    assert [j["record_idx"] for j in jobs] == [0, 2]
    assert all(j["kind"] == "attack" and j["baseline_judge"] == "COMPLY" for j in jobs)


def test_ablate_and_sham_run_on_the_same_control_set():
    """Specificity only means something if both arms see identical prompts."""
    a = select_jobs(JUDGED, "ablate_control", None)
    s = select_jobs(JUDGED, "sham_control", None)
    assert [j["record_idx"] for j in a] == [j["record_idx"] for j in s] == [0, 1, 2]
    assert all(j["kind"] == "control" for j in a)


def test_prefill_rides_along_with_the_attack():
    jobs = select_jobs(JUDGED, "add_attack", None)
    assert next(j for j in jobs if j["record_idx"] == 2)["prefill"] == "Sure,"


def test_limit_per_class_is_applied_per_class():
    judged = [_g(i, "control", "REFUSE", cls=("a" if i < 4 else "b")) for i in range(8)]
    jobs = select_jobs(judged, "ablate_control", 2)
    assert sorted(j["class"] for j in jobs) == ["a", "a", "b", "b"]


def test_unknown_arm_raises():
    with pytest.raises(ValueError, match="unknown arm"):
        select_jobs(JUDGED, "nonsense", None)


def test_summarize_reports_the_shift_not_just_the_after_rate():
    rows = [
        {"class": "f", "baseline_judge": "REFUSE", "judge": "COMPLY", "coherent": True},
        {"class": "f", "baseline_judge": "REFUSE", "judge": "COMPLY", "coherent": True},
        {"class": "f", "baseline_judge": "REFUSE", "judge": "REFUSE", "coherent": True},
        {"class": "f", "baseline_judge": "COMPLY", "judge": "REFUSE", "coherent": True},
    ]
    s = summarize(rows)["f"]
    assert s["comply_before"] == 0.25 and s["comply_after"] == 0.5
    assert s["shift"] == 0.25


def test_summarize_counts_incoherent_responses():
    """A wrecked model emits gibberish; judging gibberish yields a number that looks real."""
    rows = [{"class": "f", "baseline_judge": "REFUSE", "judge": "COMPLY", "coherent": False}]
    assert summarize(rows)["f"]["incoherent"] == 1


def test_is_coherent_flags_gibberish_and_stubs():
    assert is_coherent("Here is a clear and complete explanation of the process.")
    assert not is_coherent("ok")
    assert not is_coherent("the the the the the the the the the the the the the the")


def test_arms_include_the_specificity_control():
    assert "sham_control" in ARMS


def test_dropping_the_outlier_dim_changes_the_direction_substantially():
    """Gemma L15 is ~81% energy on dim 443; the complement is a genuinely different vector."""
    import torch

    from jbct.directions import drop_dims

    r = torch.zeros(2560, dtype=torch.float64)
    r[443] = 0.9
    r[1209] = 0.3
    r[7] = 0.2
    u = r / r.norm()
    c = drop_dims(u, (443,))
    assert abs(float(c.norm()) - 1.0) < 1e-9  # renormalized
    assert float(c[443]) == 0.0
    assert float(u @ c) < 0.5  # dominated by the outlier, so the complement is far from it


def test_drop_dims_is_a_noop_for_an_empty_outlier_tuple():
    import torch

    from jbct.directions import drop_dims

    u = torch.tensor([0.6, 0.8], dtype=torch.float64)
    assert torch.allclose(drop_dims(u, ()), u)


def test_detect_outlier_dims_finds_a_massive_dimension():
    import torch

    from jbct.directions import detect_outlier_dims

    acts = torch.randn(64, 128).abs()
    acts[:, 43] += 500.0
    assert 43 in detect_outlier_dims(acts)


def test_detect_outlier_dims_returns_empty_on_uniform_activations():
    import torch

    from jbct.directions import detect_outlier_dims

    assert detect_outlier_dims(torch.ones(32, 64)) == ()


# --------------------------------------------------- the addition specificity control


def test_sham_attack_runs_on_exactly_the_same_prompts_as_add_attack():
    """Pairwise matching, not merely aggregate: the two addition arms must intervene on the
    identical prompt set or the comparison is between different populations."""
    judged = [
        _g(1, "attack", "COMPLY"), _g(2, "attack", "REFUSE"),
        _g(3, "control", "REFUSE"), _g(4, "attack", "COMPLY"),
    ]
    real = select_jobs(judged, "add_attack", None)
    sham = select_jobs(judged, "sham_attack", None)
    assert [r["record_idx"] for r in real] == [1, 4]
    assert real == sham


def test_sham_attack_is_a_recognised_arm():
    assert "sham_attack" in ARMS
    assert set(ADDITION_ARMS) == {"add_attack", "sham_attack"}


def _arm(before, after, incoherent=0, n=35):
    return {"n": n, "incoherent": incoherent,
            "overall_comply_before": before, "overall_comply_after": after}


def test_specificity_pairs_each_coefficient_with_its_own_sham():
    """A random vector's damage grows with magnitude just as the real effect does, so the
    real arm at c=0.25 must be judged against the sham at c=0.25, not some other dose."""
    arms = {
        "add_attack@c0.25": _arm(1.0, 0.14), "sham_attack@c0.25": _arm(1.0, 0.95),
        "add_attack@c1":    _arm(1.0, 0.00), "sham_attack@c1":    _arm(1.0, 0.10),
    }
    spec = addition_specificity(arms)
    assert set(spec) == {"add_attack@c0.25", "add_attack@c1"}
    assert spec["add_attack@c0.25"]["real_shift"] == -0.86
    assert spec["add_attack@c0.25"]["sham_shift"] == -0.05
    assert spec["add_attack@c0.25"]["verdict"] == "specific"


def test_a_sham_that_moves_it_comparably_is_reported_as_not_specific():
    """The outcome this arm exists to be able to report. At coeff 2.0 the real addition left
    33 of 35 responses incoherent, so a large random vector plausibly does the same."""
    arms = {"add_attack@c2": _arm(1.0, 0.0, incoherent=33),
            "sham_attack@c2": _arm(1.0, 0.05, incoherent=31)}
    v = addition_specificity(arms)["add_attack@c2"]
    assert v["verdict"].startswith("NOT specific")
    assert v["real_incoherent"] == 33 and v["sham_incoherent"] == 31


def test_an_unpaired_addition_arm_is_skipped_not_scored_against_nothing():
    """Every run before this change lacked sham_attack entirely; those reports must not
    acquire a specificity verdict computed from a missing arm."""
    assert addition_specificity({"add_attack@c0.25": _arm(1.0, 0.14)}) == {}
    assert addition_specificity({"add_attack@c1": _arm(1.0, 0.0),
                                 "sham_attack@c1": {"n": 0}}) == {}


# ------------------------------------------------- artifacts that are not in the repo


def test_missing_candidates_names_the_command_that_regenerates_it(tmp_path):
    """`candidates.pt` is gitignored, so every consumer stage failed on a fresh pod with a
    bare FileNotFoundError naming a path. The corpus it needs IS committed, so the fix is a
    single cheap command and the message should say so."""
    from jbct.data import load_candidates

    with pytest.raises(SystemExit, match="h1_direction/run.py --model gemma-3-4b"):
        load_candidates(tmp_path, "gemma-3-4b")


def test_judged_resolves_the_committed_archive_when_the_plain_file_is_absent(tmp_path):
    """The failure that stopped the first F2 run: results/h2/<model>/judged.json is
    gitignored, so only the gz under docs/runs/ exists on a fresh clone."""
    import gzip
    import json as _json

    from jbct.data import resolve_judged

    repo = tmp_path / "repo"
    gz = repo / "docs/runs/2026-07-28-h1-h2/h2/gemma-3-4b/judged.json.gz"
    gz.parent.mkdir(parents=True)
    with gzip.open(gz, "wt", encoding="utf-8") as fh:
        fh.write(_json.dumps({"generations": []}))
    got = resolve_judged(None, tmp_path / "results/h2", "gemma-3-4b", repo_root=repo)
    assert got == gz


# ------------------------------------------------------------------------ figures


def test_causal_figures_render_and_are_deterministic(tmp_path):
    from jbct.plots import figure_ablation, figure_coherence, figure_dose_response

    site = {
        "model": "gemma-3-4b", "site": "15@-2", "layer": 15, "coeffs": [0.25, 0.5, 1.0],
        "arms": {
            "add_attack@c0.25": _arm(1.0, 0.26), "sham_attack@c0.25": _arm(1.0, 0.80),
            "add_attack@c0.5": _arm(1.0, 0.06), "sham_attack@c0.5": _arm(1.0, 0.74),
            "add_attack@c1": _arm(1.0, 0.0, incoherent=29),
            "sham_attack@c1": _arm(1.0, 0.11),
            "ablate_control": _arm(0.11, 0.63), "sham_control": _arm(0.11, 0.11),
        },
    }

    def render(into):
        figure_dose_response([site], into / "a.png", title="t")
        figure_coherence([site], into / "b.png", title="t")
        figure_ablation([site], into / "c.png", title="t")

    a, b = tmp_path / "a", tmp_path / "b"
    render(a)
    render(b)
    for name in ("a.png", "b.png", "c.png"):
        assert (a / name).stat().st_size > 0
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_an_arm_absent_from_a_run_is_skipped_not_plotted_as_zero():
    """A run with only the real arm must not draw a sham line sitting at 0, which would read
    as 'the random vector did nothing' when it was never run."""
    from jbct.plots import _arm_series

    arms = {"add_attack@c0.25": _arm(1.0, 0.26)}
    after, incoh = _arm_series(arms, "sham_attack", [0.25, 0.5])
    assert after == [None, None] and incoh == [None, None]
