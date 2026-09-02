"""CPU-only tests for the paper-closure run's populations and statistics."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from paper_closure.run import (  # noqa: E402
    _stage_names,
    class_profiles,
    exact_paired_p,
    holm_adjust,
    paired_stats,
    select_restoration_jobs,
    select_scope_jobs,
)


def _projection(population, cls, judge, value, *, prefill=False):
    return {
        "population": population,
        "class": cls,
        "judge": judge,
        "has_prefill": prefill,
        "proj_complement": {"0@-1": value, "1@-1": value + 1},
    }


def test_class_profiles_use_successful_attacks_and_refusing_controls():
    rows = [
        _projection("attack-COMPLY", "fiction", "COMPLY", 2),
        _projection("attack-COMPLY", "fiction", "COMPLY", 4),
        _projection("attack-REFUSE", "fiction", "REFUSE", 100),
        _projection("control", "fiction", "REFUSE", 8),
        _projection("control", "fiction", "REFUSE", 10),
        _projection("control", "fiction", "COMPLY", -100),
    ]
    got = class_profiles(rows, -1, [0, 1], min_class_n=2)
    assert got["fiction"]["attack"] == {0: 3.0, 1: 4.0}
    assert got["fiction"]["control"] == {0: 9.0, 1: 10.0}
    assert got["fiction"]["n_attack"] == got["fiction"]["n_control_refuse"] == 2


def test_prefill_is_excluded_from_position_matched_profiles():
    rows = [
        _projection("attack-COMPLY", "prefill", "COMPLY", 1, prefill=True),
        _projection("control", "prefill", "REFUSE", 2),
    ]
    assert class_profiles(rows, -1, [0], min_class_n=1) == {}


def _judged(idx, kind, judge, cls="fiction", prefill=None):
    return {"record_idx": idx, "base_id": idx, "kind": kind, "judge": judge,
            "class": cls, "prompt_text": f"p{idx}", "prefill": prefill}


def test_scope_arms_use_all_controls_and_stable_pair_ids():
    rows = [_judged(1, "attack", "COMPLY"), _judged(1, "control", "REFUSE"),
            _judged(2, "control", "COMPLY", cls="roleplay")]
    got = select_scope_jobs(rows)
    assert [x["pair_id"] for x in got] == ["fiction:1", "roleplay:2"]
    assert all(x["kind"] == "control" for x in got)


def test_restoration_uses_only_successful_nonprefill_attacks_in_eligible_classes():
    rows = [
        _judged(1, "attack", "COMPLY"),
        _judged(2, "attack", "REFUSE"),
        _judged(3, "control", "COMPLY"),
        _judged(4, "attack", "COMPLY", cls="prefill", prefill="Sure"),
        _judged(5, "attack", "COMPLY", cls="roleplay"),
    ]
    got = select_restoration_jobs(rows, {"fiction"})
    assert list(got) == ["fiction"]
    assert [x["pair_id"] for x in got["fiction"]] == ["fiction:1"]


def test_paired_stats_use_identical_prompts_and_exact_discordance():
    real = [
        {"pair_id": "a", "judge": "COMPLY"},
        {"pair_id": "b", "judge": "COMPLY"},
        {"pair_id": "c", "judge": "REFUSE"},
        {"pair_id": "d", "judge": "REFUSE"},
    ]
    ref = [
        {"pair_id": "a", "judge": "REFUSE"},
        {"pair_id": "b", "judge": "REFUSE"},
        {"pair_id": "c", "judge": "COMPLY"},
        {"pair_id": "d", "judge": "REFUSE"},
    ]
    got = paired_stats(real, ref, seed=1)
    assert got["delta"] == 0.25
    assert got["discordant_real_only"] == 2
    assert got["discordant_reference_only"] == 1
    assert got["p_exact"] == exact_paired_p(2, 1) == 1.0


def test_paired_stats_reject_mismatched_populations():
    with pytest.raises(ValueError, match="identical pair_id"):
        paired_stats([{"pair_id": "a", "judge": "COMPLY"}],
                     [{"pair_id": "b", "judge": "COMPLY"}])


def test_holm_adjustment_is_monotone_in_rank():
    got = holm_adjust([0.01, 0.04, 0.02])
    assert got == pytest.approx([0.03, 0.04, 0.04])


def test_stage_parser_rejects_typos_before_loading_a_model():
    assert _stage_names("scope,analyze") == {"scope", "analyze"}
    with pytest.raises(ValueError, match="unknown stages"):
        _stage_names("scope,transpalnt")
