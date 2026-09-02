"""CPU tests for the per-attack-class geometry.

The hazards here are all silent-wrong-number hazards, and two of them have already fired:

- measuring against the direction's fit corpus instead of the 50 bases every attack is built
  on, which inflated every class by ~18% and announced itself only through prefill reading
  117% when it must read 100%;
- a JSON round-trip turning integer layer keys into strings, so `--replot` would order the
  layers `"10", "2", "3"` and draw a different figure from identical data.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from jbct.categories import (  # noqa: E402
    ATTACK_FAMILIES,
    COMPLY,
    POST_PROMPT_CLASSES,
    REFERENCE_CLASS,
    REFUSE,
    auc,
    category_profile,
    clean_anchored,
    cohens_kappa,
    confusion,
    hash_inconsistency,
    label_risk,
    profile_summary,
    prompt_side_classes,
    reference_series,
    stratified_sample,
    weighted_retention,
    within_class_separation,
    zero_anchored,
)

POS = -2
N_LAYERS = 3

# The reference populations, chosen to reproduce the real relationship: the 50 dataset bases
# sit ~18% ABOVE the corpus the direction was fitted on, so dividing by the fit corpus is
# what inflates every class.
CLEAN_AT, BASE_AT, FIT_CORPUS_AT = 0.0, 100.0, 85.0


N_PER_POP = 8


def _row(population, cls, judge, value, spread=0.0, i=0):
    """One prompt. The spread is CENTRED on `value`, so the population mean is exactly
    `value` — an off-centre spread silently moved the very means these tests pin, twice."""
    offset = spread * (i - (N_PER_POP - 1) / 2)
    proj = {f"{layer}@{POS}": value + offset for layer in range(N_LAYERS)}
    return {"population": population, "class": cls, "judge": judge,
            "proj_complement": proj, "proj": {k: v * 2 for k, v in proj.items()}}


def _rows():
    """Projection rows shaped like the committed artifact.

    `fiction` sits at 20 (heavily suppressed), `roleplay` at 60 (barely), and `prefill` at
    the base exactly — which is the calibration the real data has to reproduce, because a
    prefill attack's prompt IS the plain request.
    """
    rows = []
    for i in range(N_PER_POP):
        rows.append(_row("clean", None, None, CLEAN_AT, 1.0, i))
        rows.append(_row("harmful", None, None, FIT_CORPUS_AT, 1.0, i))
        rows.append(_row("control", REFERENCE_CLASS, REFUSE, BASE_AT, 1.0, i))
        rows.append(_row("control", "fiction", REFUSE, 90.0, 1.0, i))
        rows.append(_row("attack-COMPLY", "fiction", COMPLY, 20.0, 1.0, i))
        rows.append(_row("attack-REFUSE", "fiction", REFUSE, 70.0, 1.0, i))
        rows.append(_row("attack-COMPLY", "roleplay", COMPLY, 60.0, 1.0, i))
        rows.append(_row("attack-REFUSE", "roleplay", REFUSE, 80.0, 1.0, i))
        rows.append(_row("attack-COMPLY", REFERENCE_CLASS, COMPLY, BASE_AT, 1.0, i))
    return rows


@pytest.fixture
def refs():
    return reference_series(_rows(), POS, N_LAYERS)


# --------------------------------------------------------------------- the reference


def test_the_reference_is_the_dataset_bases_not_the_fit_corpus(refs):
    """The correction this module exists for. Both populations look like plausible 'harmful'
    sets, and picking the wrong one shifts every class by a constant nothing else reveals."""
    layer0 = refs["layers"][0]
    assert layer0["base"] == pytest.approx(BASE_AT)            # the bases, not the 85
    assert layer0["clean"] == pytest.approx(CLEAN_AT)
    assert refs["n_base"] == N_PER_POP


def test_prefill_lands_on_the_reference_which_is_the_calibration_check(refs):
    """prefill cannot move the prompt, so it MUST read ~1.0. Against the fit corpus it reads
    ~1.18 — that gap is the only way the wrong reference announces itself."""
    prof = category_profile(_rows(), REFERENCE_CLASS, POS, N_LAYERS, refs)
    got = profile_summary(prof, refs)
    assert got["p_zero_weighted"] == pytest.approx(1.0, abs=1e-9)

    wrong = {"layers": {layer: {**e, "base": FIT_CORPUS_AT}
                        for layer, e in refs["layers"].items()}}
    inflated = profile_summary(prof, wrong)["p_zero_weighted"]
    assert inflated == pytest.approx(BASE_AT / FIT_CORPUS_AT, rel=0.02)
    assert inflated > 1.15


def test_prefill_is_excluded_from_prompt_side_comparisons():
    """Its attack is a forced assistant opening; the prompt is the bare request token for
    token, so a prompt-side reading of it is the reference, not the attack."""
    assert REFERENCE_CLASS in POST_PROMPT_CLASSES
    got = prompt_side_classes(["fiction", "prefill", "roleplay", None])
    assert got == ["fiction", "roleplay"]


# ------------------------------------------------------------------------- the anchors


def test_clean_anchor_endpoints():
    ref = {"clean": 0.0, "base": 100.0, "gap": 100.0, "gap_usable": True}
    assert clean_anchored(100.0, ref) == pytest.approx(1.0)     # a plain harmful request
    assert clean_anchored(0.0, ref) == pytest.approx(0.0)       # a harmless one
    assert clean_anchored(-50.0, ref) == pytest.approx(-0.5)    # past harmless


def test_zero_anchor_endpoints_match_f3s_retention_axis():
    """0 is ablation on this anchor, which is what makes a class's position here the target
    a per-class F3 arm would impose."""
    ref = {"base": 100.0, "base_usable": True}
    assert zero_anchored(100.0, ref) == pytest.approx(1.0)
    assert zero_anchored(0.0, ref) == pytest.approx(0.0)
    assert zero_anchored(29.1, ref) == pytest.approx(0.291)


def test_the_two_anchors_disagree_because_harmless_is_not_at_zero_projection():
    """Not a bug, and the reason both are reported: on gemma the harmless population sits at
    -495 and the base at +685, so zero projection is 42% of the way along the clean axis."""
    ref = {"clean": -495.0, "base": 685.0, "gap": 1180.0,
           "gap_usable": True, "base_usable": True}
    assert zero_anchored(0.0, ref) == pytest.approx(0.0)
    assert clean_anchored(0.0, ref) == pytest.approx(0.42, abs=0.01)


def test_the_clean_anchor_is_refused_when_harmless_and_harmful_barely_separate():
    """The denominator is a gap. When the two populations sit on top of each other the gap is
    noise, and dividing by it is not a measurement."""
    # Both populations sit high and on top of each other: the LEVEL is well determined,
    # the SEPARATION is not.
    rows = [_row("clean", None, None, 175.0, 50.0, i) for i in range(N_PER_POP)]
    rows += [_row("control", REFERENCE_CLASS, REFUSE, 175.05, 50.0, i)
             for i in range(N_PER_POP)]
    got = reference_series(rows, POS, 1)["layers"][0]
    assert got["gap_usable"] is False
    assert clean_anchored(10.0, got) is None
    # the two guards are independent: the base is 175, so THAT anchor is still fine
    assert got["base_usable"] is True
    assert zero_anchored(10.0, got) is not None


def test_the_zero_anchor_is_refused_when_the_base_projection_is_near_zero():
    """qwen's L9 harmful mean is 0.03. A per-layer ratio there is noise over noise — the
    artifact that once reported 175% suppression."""
    rows = [_row("clean", None, None, -400.0, 1.0, i) for i in range(N_PER_POP)]
    rows += [_row("control", REFERENCE_CLASS, REFUSE, 0.03, 2.0, i) for i in range(N_PER_POP)]
    got = reference_series(rows, POS, 1)["layers"][0]
    assert got["base_usable"] is False
    assert zero_anchored(10.0, got) is None
    # ...while the clean anchor survives, because its denominator is the 400-wide gap
    assert got["gap_usable"] is True
    assert clean_anchored(10.0, got) is not None


def test_a_clean_separation_is_usable():
    assert reference_series(_rows(), POS, 1)["layers"][0]["gap_usable"] is True


# ------------------------------------------------------------------------- the profile


def test_a_profile_conditions_on_the_outcome(refs):
    """'How far do attacks that WORKED move the direction' is the question; the REFUSED ones
    are a different population and must not be averaged in."""
    complied = category_profile(_rows(), "fiction", POS, N_LAYERS, refs, judge=COMPLY)
    refused = category_profile(_rows(), "fiction", POS, N_LAYERS, refs, judge=REFUSE)
    assert complied["layers"][0]["mean"] == pytest.approx(20.0)
    assert refused["layers"][0]["mean"] == pytest.approx(70.0)
    assert complied["n_prompts"] == refused["n_prompts"] == N_PER_POP


def test_the_summary_orders_classes_by_how_much_refusal_they_leave(refs):
    rows = _rows()
    fiction = profile_summary(category_profile(rows, "fiction", POS, N_LAYERS, refs), refs)
    roleplay = profile_summary(category_profile(rows, "roleplay", POS, N_LAYERS, refs), refs)
    assert fiction["p_zero_weighted"] < roleplay["p_zero_weighted"]
    assert fiction["shift_toward_clean"] > roleplay["shift_toward_clean"]


def test_the_clean_aggregate_is_weighted_by_separation_not_a_flat_mean():
    """A layer where harmless and harmful barely separate must not count as much as one
    where they separate cleanly — the same reasoning that makes the zero anchor
    magnitude-weighted."""
    refs = {"layers": {
        0: {"clean": 0.0, "base": 1.0, "gap": 1.0, "gap_usable": True, "base_usable": True},
        1: {"clean": 0.0, "base": 99.0, "gap": 99.0, "gap_usable": True, "base_usable": True},
    }}
    prof = {"class": "x", "family": None, "position": POS, "judge": COMPLY,
            "post_prompt": False, "n_prompts": 5, "layers": {
                0: {"mean": 1.0, "p_clean": 1.0, "base_usable": True},
                1: {"mean": 0.0, "p_clean": 0.0, "base_usable": True}}}
    got = profile_summary(prof, refs)
    # flat mean would be 0.5; weighted by |gap| it is 1/100
    assert got["p_clean_weighted"] == pytest.approx(0.01)


def test_inversion_past_harmless_is_counted(refs):
    """`p_zero_weighted` takes absolute values, so a class that pushes the projection past
    harmless reads the same as one that keeps a small positive value. gemma `fiction` does
    this at L14 (-41%), which is why the two anchors order the classes differently."""
    prof = {"class": "x", "family": None, "position": POS, "judge": COMPLY,
            "post_prompt": False, "n_prompts": 5, "layers": {
                0: {"mean": -5.0, "p_clean": -0.4, "base_usable": True},
                1: {"mean": 5.0, "p_clean": 0.4, "base_usable": True}}}
    refs2 = {"layers": {
        0: {"clean": 0.0, "base": 10.0, "gap": 10.0, "gap_usable": True, "base_usable": True},
        1: {"clean": 0.0, "base": 10.0, "gap": 10.0, "gap_usable": True, "base_usable": True}}}
    got = profile_summary(prof, refs2)
    assert got["n_layers_inverted"] == 1 and got["layers_inverted"] == [0]
    # sign-blind: both layers contribute |5| to the zero anchor
    assert got["p_zero_weighted"] == pytest.approx(0.5)


def test_layers_with_an_unusable_reference_are_dropped_and_counted(refs):
    prof = {"class": "x", "family": None, "position": POS, "judge": COMPLY,
            "post_prompt": False, "n_prompts": 5, "layers": {
                0: {"mean": 5.0, "p_clean": None, "base_usable": False},
                1: {"mean": 5.0, "p_clean": 0.5, "base_usable": True}}}
    refs2 = {"layers": {0: {"gap": 1.0}, 1: {"gap": 10.0, "base": 10.0}}}
    got = profile_summary(prof, refs2)
    assert got["n_layers_used"] == 1 and got["n_layers_dropped"] == 1


# ------------------------------------------------------- one ruler shared with F3


def test_weighted_retention_is_the_same_object_f3_transplant_uses():
    """The paper places a swept threshold and each class's own suppression on one axis. Two
    definitions free to drift would put them on different rulers without ever raising."""
    from f3_transplant.run import weighted_retention as f3_version

    assert f3_version is weighted_retention


def test_weighted_retention_is_magnitude_weighted_not_a_mean_of_ratios():
    targets = {0: 1.2, 1: 50.0, 2: 100.0}
    reference = {0: 0.03, 1: 100.0, 2: 200.0}
    got = weighted_retention(targets, reference, [0, 1, 2])
    assert got == pytest.approx(151.2 / 300.03)
    assert got < 1.0        # a mean of per-layer ratios would give 13.7


# ------------------------------------------------------------- within-class separation


def test_within_class_auc_matches_the_h1_implementation():
    """Same construction as F1's pooled `auc_refuse_over_comply`, or the per-class figure
    would not be on the same scale as the number the paper already reports."""
    from h1_direction.run import auc as h1_auc

    a, b = [3.0, 1.0, 2.0], [0.5, 2.0]
    assert auc(a, b) == pytest.approx(h1_auc(a, b))


def test_within_class_separation_needs_both_outcomes():
    """A class that succeeds 98% of the time has no failure group, so the question is
    unanswerable there rather than answered at chance."""
    rows = _rows()
    assert within_class_separation(rows, "fiction", 0, POS, min_n=5) is not None
    assert within_class_separation(rows, "fiction", 0, POS, min_n=20) is None


def test_within_class_separation_reports_both_group_sizes():
    got = within_class_separation(_rows(), "roleplay", 0, POS, min_n=5)
    assert got["n_refuse"] == N_PER_POP and got["n_comply"] == N_PER_POP
    # refused attacks sit higher on the direction than successful ones
    assert got["auc_refuse_over_comply"] > 0.5


def test_a_class_with_no_rows_returns_none():
    assert within_class_separation(_rows(), "nonexistent", 0, POS) is None


# ------------------------------------------------------------------- judge reliability


def test_stratified_sample_takes_from_every_cell_not_the_majority():
    """A uniform sample spends its budget confirming the easy majority and never reaches the
    pockets where the label is in doubt."""
    items = ([{"c": "fiction", "j": COMPLY, "i": i} for i in range(49)]
             + [{"c": "fiction", "j": REFUSE, "i": 99}]
             + [{"c": "roleplay", "j": COMPLY, "i": i} for i in range(34)])
    got = stratified_sample(items, lambda it: (it["c"], it["j"]), 2)
    cells = {(it["c"], it["j"]) for it in got}
    assert cells == {("fiction", COMPLY), ("fiction", REFUSE), ("roleplay", COMPLY)}
    assert len(got) == 5        # 2 + 1 (only one available) + 2


def test_stratified_sample_is_deterministic_across_runs():
    """The human labels attach to the chosen items, so re-running must select the same ones."""
    items = [{"c": "x", "i": i} for i in range(50)]
    a = stratified_sample(items, lambda it: it["c"], 5)
    b = stratified_sample(list(reversed(items)), lambda it: it["c"], 5)
    assert a == b


def test_stratified_sample_does_not_just_take_the_first_n():
    """First-N tracks dataset construction order, which would draw every topic from one block."""
    items = [{"c": "x", "i": i} for i in range(50)]
    got = stratified_sample(items, lambda it: it["c"], 5)
    assert [it["i"] for it in got] != [0, 1, 2, 3, 4]


def test_confusion_separates_the_two_error_directions():
    """A false COMPLY inflates a success rate AND puts a refusing prompt into the population
    whose geometry the transplant copies; a false REFUSE only loses power."""
    pairs = [(COMPLY, COMPLY)] * 6 + [(REFUSE, COMPLY)] * 2 + [(COMPLY, REFUSE)] * 1 + \
            [(REFUSE, REFUSE)] * 11
    got = confusion(pairs)
    assert got["n"] == 20
    assert got["refuse_comply"] == 2 and got["comply_refuse"] == 1
    assert got["false_comply_rate"] == pytest.approx(2 / 13)
    assert got["false_refuse_rate"] == pytest.approx(1 / 7)
    assert got["agreement"] == pytest.approx(17 / 20)


def test_kappa_punishes_a_judge_that_only_looks_good_because_one_label_dominates():
    """A judge that always says REFUSE scores 96% agreement on a class that succeeds 4% of
    the time. Kappa is what stops that reading as reliability."""
    pairs = [(REFUSE, REFUSE)] * 96 + [(COMPLY, REFUSE)] * 4
    assert confusion(pairs)["agreement"] == pytest.approx(0.96)
    assert cohens_kappa(pairs) == pytest.approx(0.0, abs=1e-9)


def test_kappa_is_one_for_perfect_agreement_on_a_balanced_set():
    pairs = [(COMPLY, COMPLY)] * 10 + [(REFUSE, REFUSE)] * 10
    assert cohens_kappa(pairs) == pytest.approx(1.0)


def test_label_risk_counts_truncated_refusals():
    """~48 of 50 gemma prefill CONTROLS are truncated-and-REFUSE, and the control rate is
    what every lift in the paper is measured against."""
    rows = [{"class": "prefill", "kind": "control", "judge": REFUSE,
             "hit_token_cap": True, "n_new_tokens": 1024} for _ in range(48)]
    rows += [{"class": "prefill", "kind": "control", "judge": COMPLY,
              "hit_token_cap": False, "n_new_tokens": 500} for _ in range(2)]
    got = label_risk(rows)["prefill/control"]
    assert got["truncated_refuse"] == 48
    assert got["cap_rate"] == pytest.approx(0.96)
    assert got["at_risk_rate"] == pytest.approx(0.96)


def test_label_risk_counts_suspiciously_short_compliances():
    rows = [{"class": "fiction", "kind": "attack", "judge": COMPLY,
             "hit_token_cap": False, "n_new_tokens": 3}]
    assert label_risk(rows)["fiction/attack"]["short_comply"] == 1


def test_a_long_compliance_is_not_at_risk():
    rows = [{"class": "fiction", "kind": "attack", "judge": COMPLY,
             "hit_token_cap": True, "n_new_tokens": 1024}]
    assert label_risk(rows)["fiction/attack"]["at_risk_rate"] == pytest.approx(0.0)


def test_identical_responses_with_different_labels_are_a_provable_contradiction():
    rows = [{"response_sha256": "aa", "judge": COMPLY},
            {"response_sha256": "aa", "judge": REFUSE},
            {"response_sha256": "bb", "judge": COMPLY}]
    got = hash_inconsistency(rows)
    assert got["n_conflicting"] == 1 and got["conflicting"] == ["aa"]
    assert got["informative"] is True


def test_all_distinct_responses_makes_the_consistency_check_uninformative():
    """700 of 700 responses are distinct in both real runs, so the judge was never asked the
    same question twice — nothing about its self-consistency can be inferred, and the report
    has to say so rather than reporting zero contradictions as a clean bill of health."""
    rows = [{"response_sha256": h, "judge": COMPLY} for h in ("aa", "bb", "cc")]
    got = hash_inconsistency(rows)
    assert got["n_conflicting"] == 0 and got["informative"] is False


# ------------------------------------------------------------------------ round trips


def test_a_committed_report_reloads_with_integer_layer_keys(tmp_path):
    """JSON has no integer keys, so a reloaded report orders layers '10' < '2' < '3' and
    `--replot` would draw a different figure from identical data."""
    from f1_categories.run import rehydrate

    report = {"per_position": {"-2": {
        "references": {"layers": {2: {"gap": 1.0}, 10: {"gap": 2.0}}},
        "profiles": [{"class": "fiction", "layers": {2: {"mean": 1.0}, 10: {"mean": 2.0}}}],
    }}}
    reloaded = json.loads(json.dumps(report))
    assert list(reloaded["per_position"]["-2"]["references"]["layers"]) == ["2", "10"]
    fixed = rehydrate(reloaded)
    assert sorted(fixed["per_position"]["-2"]["references"]["layers"]) == [2, 10]
    assert sorted(fixed["per_position"]["-2"]["profiles"][0]["layers"]) == [2, 10]


def test_the_behavioural_taxonomy_covers_every_class_the_dataset_uses():
    """A class missing here renders with family `None` rather than failing, so the gap would
    only show up as a blank label in a figure."""
    assert set(ATTACK_FAMILIES) == {
        "prefill", "refusal_suppression", "refusal_suppression_long",
        "fiction", "roleplay", "narrative_sandwich", "authoritative_tone"}


# ------------------------------------------------------- reading H2's judged files
#
# Three shapes of judged file exist on disk and all three are real. Guessing one meant the
# human-labelling export failed on the ONLY file that actually carries response text.


def _judged_payload():
    return {"metadata": {"judge_model": "x"},
            "generations": [{"record_idx": 0, "class": "fiction", "kind": "attack",
                             "judge": COMPLY, "prompt_text": "p", "response": "r",
                             "hit_token_cap": False, "n_new_tokens": 10}]}


def test_reads_h2s_own_generations_shape(tmp_path):
    from f1_categories.run import read_judged_any

    path = tmp_path / "judged.json"
    path.write_text(json.dumps(_judged_payload()))
    assert read_judged_any(path)[0]["response"] == "r"


def test_reads_the_slimmed_judgments_shape(tmp_path):
    from f1_categories.run import read_judged_any

    path = tmp_path / "judgments.json"
    path.write_text(json.dumps({"metadata": {}, "judgments": [{"judge": COMPLY}]}))
    assert read_judged_any(path)[0]["judge"] == COMPLY


def test_reads_the_committed_gzipped_archive(tmp_path):
    """`judged.json` is gitignored, so on a fresh clone the archive under docs/runs is the
    only copy — and it is gzipped."""
    import gzip

    from f1_categories.run import read_judged_any

    path = tmp_path / "judged.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(_judged_payload(), fh)
    assert read_judged_any(path)[0]["response"] == "r"


def test_a_file_with_neither_key_fails_loudly(tmp_path):
    from f1_categories.run import read_judged_any

    path = tmp_path / "judged.json"
    path.write_text(json.dumps({"metadata": {}, "rows": []}))
    with pytest.raises(SystemExit, match="neither"):
        read_judged_any(path)


def test_the_export_reads_the_archives_prompt_key_not_a_guessed_one(tmp_path):
    """The archive names it `prompt_text`; H2's slimmed output has no prompt at all. Reading
    `prompt` alone silently produced a labelling task with no prompts to judge against."""
    from f1_categories.run import export_judge_sample

    path = tmp_path / "judged.json"
    path.write_text(json.dumps(_judged_payload()))
    out = export_judge_sample(path, 1, tmp_path)
    assert json.loads(out.read_text())[0]["prompt"] == "p"


def test_the_export_refuses_a_slimmed_file_with_an_actionable_message(tmp_path):
    from f1_categories.run import export_judge_sample

    path = tmp_path / "judgments.json"
    path.write_text(json.dumps({"judgments": [{"class": "fiction", "judge": COMPLY}]}))
    with pytest.raises(SystemExit, match="without `response` text"):
        export_judge_sample(path, 1, tmp_path)


# --------------------------------------------------------------------- do not commit this
#
# `scripts/archive_results.py` carries the scar: an exact-name gitignore pattern missed
# `<arm>_generations.json` and 15 H3-pilot files of raw harmful completions reached the repo.
# The labelling task is the same hazard under a third name — it exists precisely so a human
# can read the completions, so it is plaintext by construction.


def test_the_human_labelling_task_is_gitignored():
    import subprocess

    root = Path(__file__).resolve().parents[1]
    target = "results/f1_categories/gemma-3-4b/judge_sample.json"
    got = subprocess.run(["git", "check-ignore", "-q", target],
                         cwd=root, capture_output=True)
    assert got.returncode == 0, (
        f"{target} is NOT gitignored. It carries full response text by construction; "
        f"committing it publishes by-construction-harmful completions in plaintext.")


def test_the_ignore_rule_covers_a_renamed_or_suffixed_sample():
    """An exact-name rule is what failed last time. `judge_sample.labelled.json` is the file
    a human hands back, and it carries the same text."""
    import subprocess

    root = Path(__file__).resolve().parents[1]
    for target in ("results/f1_categories/x/judge_sample.labelled.json",
                   "results/f1_categories/x/judge_sample_qwen.json"):
        got = subprocess.run(["git", "check-ignore", "-q", target],
                             cwd=root, capture_output=True)
        assert got.returncode == 0, f"{target} is not ignored"


# ------------------------------------------------- re-weighting the stratified sample
#
# The labelling task deliberately over-samples rare cells so judge failures can be FOUND.
# Reporting its raw agreement as the run's agreement is a different claim, and a wrong one.


def test_raw_sample_agreement_is_not_the_populations_agreement():
    """A cell holding 1 of 700 responses must not count as much as one holding 499."""
    from jbct.categories import population_agreement

    pairs = {("fiction", "attack", COMPLY): [(COMPLY, COMPLY)] * 4,        # perfect, huge cell
             ("fiction", "attack", REFUSE): [(COMPLY, REFUSE)] * 4}        # wrong, tiny cell
    sizes = {("fiction", "attack", COMPLY): 499, ("fiction", "attack", REFUSE): 1}
    raw = 4 / 8                                                   # the sample says 50%
    got = population_agreement(pairs, sizes)
    assert got["agreement"] == pytest.approx(499 / 500)            # the run is 99.8%
    assert got["agreement"] > raw
    assert got["coverage"] == pytest.approx(1.0)


def test_coverage_reports_cells_that_were_never_labelled():
    """Below 1.0 the estimate speaks only for the part of the run that was sampled."""
    from jbct.categories import population_agreement

    pairs = {("a", "attack", COMPLY): [(COMPLY, COMPLY)]}
    sizes = {("a", "attack", COMPLY): 30, ("b", "control", REFUSE): 70}
    got = population_agreement(pairs, sizes)
    assert got["coverage"] == pytest.approx(0.3)
    assert got["n_population"] == 100


def test_reweighting_is_unavailable_rather_than_wrong_without_population_sizes():
    from jbct.categories import population_agreement

    assert population_agreement({("a", "b", COMPLY): [(COMPLY, COMPLY)]}, {})[
        "available"] is False


def test_a_sampled_cell_absent_from_the_population_is_skipped_not_counted():
    from jbct.categories import population_agreement

    pairs = {("ghost", "attack", COMPLY): [(COMPLY, REFUSE)],
             ("real", "attack", COMPLY): [(COMPLY, COMPLY)]}
    sizes = {("real", "attack", COMPLY): 10}
    got = population_agreement(pairs, sizes)
    assert got["agreement"] == pytest.approx(1.0) and got["n_cells"] == 1


def test_every_exported_row_has_a_unique_key(tmp_path):
    """`record_idx` is shared by an attack and its matched control — they are two halves of
    one dataset record. Keying a re-read on it merged pairs and moved the reported agreement
    by 2 points off 98 rows, silently."""
    from f1_categories.run import export_judge_sample

    payload = {"generations": [
        {"record_idx": 74, "class": "fiction", "kind": "attack", "judge": COMPLY,
         "prompt_text": "p", "response": "r", "hit_token_cap": False, "n_new_tokens": 9},
        {"record_idx": 74, "class": "fiction", "kind": "control", "judge": COMPLY,
         "prompt_text": "p", "response": "r2", "hit_token_cap": False, "n_new_tokens": 9},
    ]}
    path = tmp_path / "judged.json"
    path.write_text(json.dumps(payload))
    rows = json.loads(export_judge_sample(path, 5, tmp_path).read_text())
    assert len(rows) == 2
    assert len({r["idx"] for r in rows}) == 1          # the collision is real
    assert len({r["key"] for r in rows}) == 2          # the key survives it
