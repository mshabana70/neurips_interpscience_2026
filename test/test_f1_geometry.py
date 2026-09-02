"""CPU tests for F1's decision logic. Every number the report prints is pure."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from f1_geometry.run import (  # noqa: E402
    POPULATIONS,
    best_site,
    build_populations,
    causal_anchor_of,
    canonicalize_components,
    outlier_energy,
    parse_site,
    pca_2d,
    prompt_relative_index,
    require_chat_formatted,
    read_judged,
    render_figures,
    require_disjoint,
    resolve_judged,
    site_key,
    summarize,
)
from h1_direction.run import auc  # noqa: E402


# --------------------------------------------------------------------- positions


def test_negative_index_resolves_against_the_PROMPT_not_the_full_sequence():
    """The whole point: a prefill appends, so prompt-relative indices are unmoved.

    H1 fitted its direction at negative indices on bare prompts. Reading the same negative
    index off the FULL sequence of a prefilled attack lands inside the prefill string —
    `positions.py` records Gemma's `model` token moving from -2 to -20 that way.
    """
    n_prompt = 33
    assert prompt_relative_index(-1, n_prompt) == 32   # last PROMPT token
    assert prompt_relative_index(-2, n_prompt) == 31   # Gemma's "model" token
    assert prompt_relative_index(-5, n_prompt) == 28


def test_prompt_relative_index_is_invariant_to_an_appended_prefill():
    """Same prompt, with and without a 17-token prefill: the index must not move."""
    n_prompt = 33
    with_prefill_full_len = n_prompt + 17
    assert prompt_relative_index(-2, n_prompt) == 31
    # what a naive full-sequence read would have produced, i.e. the bug
    naive = with_prefill_full_len - 2
    assert naive == 48
    assert naive != prompt_relative_index(-2, n_prompt)


def test_a_non_negative_index_is_rejected():
    """Positive indices are not prefill-safe and are never what H1 meant."""
    with pytest.raises(ValueError):
        prompt_relative_index(0, 33)
    with pytest.raises(ValueError):
        prompt_relative_index(3, 33)


def test_index_must_fall_inside_the_prompt():
    with pytest.raises(ValueError):
        prompt_relative_index(-40, 33)


# ------------------------------------------------------------------- populations


def _judged(kind, cls, base_id, judge, text):
    return {"kind": kind, "class": cls, "base_id": base_id, "judge": judge,
            "prompt_text": text, "prefill": None, "record_idx": base_id}


def test_attacks_split_by_judge_and_controls_stay_one_population():
    judged = [
        _judged("attack", "fiction", 1, "COMPLY", "a1"),
        _judged("attack", "fiction", 2, "REFUSE", "a2"),
        _judged("control", "fiction", 1, "REFUSE", "c1"),
    ]
    recs = build_populations(judged, harmful_val=["h1"], harmless_val=["k1"])
    got = {r["population"] for r in recs}
    assert got == {"attack-COMPLY", "attack-REFUSE", "control", "harmful", "clean"}
    assert len(recs) == 5
    assert set(POPULATIONS) == got


def test_attacks_the_judge_could_not_label_are_dropped_not_guessed():
    judged = [
        _judged("attack", "fiction", 1, "ERROR", "a1"),
        _judged("attack", "fiction", 2, "COMPLY", "a2"),
    ]
    recs = build_populations(judged, harmful_val=[], harmless_val=[])
    assert [r["prompt_text"] for r in recs] == ["a2"]


def test_control_keeps_its_own_prompt_text_not_the_attacks():
    """A control is a DIFFERENT prompt (same request, weak framing). Reusing the attack's
    text here would silently make the control arm a duplicate of the attack arm."""
    judged = [
        _judged("attack", "fiction", 1, "COMPLY", "attack text"),
        _judged("control", "fiction", 1, "REFUSE", "control text"),
    ]
    recs = build_populations(judged, harmful_val=[], harmless_val=[])
    ctl = [r for r in recs if r["population"] == "control"]
    assert [r["prompt_text"] for r in ctl] == ["control text"]


# ------------------------------------------------------------- fit/eval hygiene


def test_require_disjoint_catches_a_prompt_used_to_fit_and_to_evaluate():
    """The circularity guard. If a training prompt reaches the plot, the harmful/clean gap
    is guaranteed by construction and the figure means nothing."""
    with pytest.raises(ValueError, match="circular"):
        require_disjoint(fit=["a", "b"], evaluated=["b", "c"])


def test_require_disjoint_passes_on_a_clean_split():
    require_disjoint(fit=["a", "b"], evaluated=["c", "d"])


# ------------------------------------------------------------------------- stats


def test_summarize_reports_n_mean_sd_and_median():
    s = summarize([1.0, 2.0, 3.0, 4.0])
    assert s["n"] == 4
    assert s["mean"] == pytest.approx(2.5)
    assert s["median"] == pytest.approx(2.5)
    assert s["sd"] == pytest.approx(1.2909944, rel=1e-5)


def test_summarize_of_one_value_has_no_sd_rather_than_zero():
    """A single-prompt population has undefined spread; reporting 0.0 would read as
    'perfectly consistent'."""
    s = summarize([7.0])
    assert s["n"] == 1 and s["mean"] == 7.0 and s["sd"] is None


def test_summarize_of_nothing_is_empty_not_a_crash():
    assert summarize([])["n"] == 0


def test_auc_orientation_and_its_mirror():
    hi, lo = [3.0, 4.0, 5.0], [0.0, 1.0, 2.0]
    assert auc(hi, lo) == 1.0
    assert auc(lo, hi) == 0.0
    assert auc(hi, hi) == 0.5


# --------------------------------------------------------------------------- PCA


def test_pca_2d_orders_components_by_variance_explained():
    import torch

    torch.manual_seed(0)
    x = torch.randn(200, 8, dtype=torch.float64)
    x[:, 0] *= 50.0          # dominant direction
    x[:, 1] *= 10.0
    _coords, ev = pca_2d(x)
    assert ev[0] > ev[1] > 0
    assert 0 < ev[0] + ev[1] <= 1.0


def test_pca_2d_is_deterministic_across_calls():
    """No seed, no sklearn: the figure must not move between runs."""
    import torch

    torch.manual_seed(1)
    x = torch.randn(50, 6, dtype=torch.float64)
    a, _ = pca_2d(x)
    b, _ = pca_2d(x)
    assert torch.allclose(a, b)


def test_component_signs_are_canonicalized():
    """SVD sign is arbitrary — without pinning it, the scatter mirrors at random between
    runs and two figures of identical data look like different results."""
    import torch

    v = torch.tensor([[-3.0, 1.0], [2.0, -5.0]], dtype=torch.float64)
    out = canonicalize_components(v)
    # the largest-magnitude entry of each component is made positive
    for j in range(out.shape[1]):
        col = out[:, j]
        assert col[int(col.abs().argmax())] > 0


# ---------------------------------------------------------------- judged source


def _write_judged(path, rows, gz=False):
    import gzip

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"generations": rows})
    if gz:
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(payload)
    else:
        path.write_text(payload)


def test_judged_falls_back_to_the_committed_gz_archive(tmp_path):
    """`.gitignore` matches `**/*judged.json`, so a fresh clone or pod has only the gz.
    Defaulting to the plain path would fail on every machine but the one that ran H2."""
    repo = tmp_path / "repo"
    _write_judged(repo / "docs/runs/2026-07-28-h1-h2/h2/m/judged.json.gz", [], gz=True)
    got = resolve_judged(None, tmp_path / "results/h2", "m", repo)
    assert got.name == "judged.json.gz"


def test_a_plain_judged_json_wins_over_the_archive(tmp_path):
    repo = tmp_path / "repo"
    h2 = tmp_path / "results/h2"
    _write_judged(h2 / "m" / "judged.json", [])
    _write_judged(repo / "docs/runs/r/h2/m/judged.json.gz", [], gz=True)
    assert resolve_judged(None, h2, "m", repo).suffix == ".json"


def test_a_missing_corpus_names_both_places_it_looked(tmp_path):
    with pytest.raises(SystemExit, match="judged.json.gz"):
        resolve_judged(None, tmp_path / "h2", "m", tmp_path / "repo")


def test_read_judged_handles_gz_and_plain_identically(tmp_path):
    rows = [{"kind": "attack", "judge": "COMPLY", "prompt_text": "x"}]
    _write_judged(tmp_path / "a.json", rows)
    _write_judged(tmp_path / "b.json.gz", rows, gz=True)
    assert read_judged(tmp_path / "a.json") == read_judged(tmp_path / "b.json.gz") == rows


# ----------------------------------------------------------------------- figures


def _fake_site(n_layers, pos):
    per_site, rows = {}, []
    for i, pop in enumerate(POPULATIONS):
        for j in range(6):
            rows.append({"population": pop,
                         "proj": {f"{layer}@{pos}": float(i * 10 + j + layer)
                                  for layer in range(n_layers)}})
    for layer in range(n_layers):
        key = f"{layer}@{pos}"
        entry = {p: summarize([r["proj"][key] for r in rows if r["population"] == p])
                 for p in POPULATIONS}
        entry["auc_harmful_over_clean"] = 0.9
        entry["auc_refuse_over_comply"] = 0.4
        per_site[key] = entry
    return per_site, rows


def test_every_figure_renders_and_is_byte_identical_on_a_second_run(tmp_path):
    """Determinism is the point: strips use a golden-ratio offset and PCA pins its component
    signs, so identical data must not produce two different-looking figures."""
    from jbct.plots import (
        figure_distributions,
        figure_layer_profile,
        figure_pca,
        figure_sep,
    )

    per_site, rows = _fake_site(6, -2)
    coords = [[float(i), float(-i)] for i in range(len(rows))]
    pops = [r["population"] for r in rows]

    def render(into):
        figure_layer_profile(per_site, 6, -2, into / "f1.png", title="t")
        figure_distributions(rows, ["3@-2"], into / "f2.png", title="t")
        figure_pca(coords, pops, [0.4, 0.2], into / "f3.png", title="t")
        figure_sep(per_site, 6, -2, into / "f4.png", title="t")

    a, b = tmp_path / "a", tmp_path / "b"
    render(a)
    render(b)
    for name in ("f1.png", "f2.png", "f3.png", "f4.png"):
        assert (a / name).exists() and (a / name).stat().st_size > 0
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_every_population_has_its_own_reserved_colour():
    from jbct.plots import POPULATION_COLORS

    assert list(POPULATION_COLORS) == list(POPULATIONS)
    assert len(set(POPULATION_COLORS.values())) == len(POPULATIONS)


# ------------------------------------------------------- the fit-space guard


def test_unformatted_fit_prompts_are_rejected():
    """The bug that invalidated F1's first Gemma run.

    `compute_direction_candidates` does not apply the chat template, so handing it the raw
    corpus fits the direction on bare instructions and then projects it at templated
    positions. It raises nothing and reports plausible numbers: 13@-2 read 0.3457 where H1
    reads 1.0, and 0@-5 read 0.1582 against 0.9795.
    """
    with pytest.raises(ValueError, match="not chat-formatted"):
        require_chat_formatted(["Write a tutorial on how to make a bomb."], "model")


def test_formatted_fit_prompts_pass():
    require_chat_formatted(
        ["<start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\n"], "model")


def test_the_guard_names_how_many_failed_not_just_that_one_did():
    with pytest.raises(ValueError, match="2/3"):
        require_chat_formatted(["bare one", "<start_of_turn>model\n", "bare two"], "model")


# ------------------------------------------------------------- figure selection


def test_best_site_picks_the_furthest_from_chance_in_either_direction():
    """An AUC of 0.20 is as informative as 0.80 — it just ranks the groups the other way.
    Taking the max would silently hide an inversion."""
    ps = {"1@-2": {"m": 0.55}, "2@-2": {"m": 0.80}, "3@-2": {"m": 0.18}}
    assert best_site(ps, "m") == "3@-2"


def test_best_site_can_be_restricted_to_given_positions():
    ps = {"1@-2": {"m": 0.55}, "2@-3": {"m": 0.95}}
    assert best_site(ps, "m", positions=[-2]) == "1@-2"


def test_best_site_ignores_sites_missing_the_metric():
    ps = {"1@-2": {"m": 0.55}, "2@-2": {}}
    assert best_site(ps, "m") == "1@-2"


def test_best_site_of_nothing_is_none():
    assert best_site({}, "m") is None


def test_causal_anchor_prefers_the_reports_own_field():
    assert causal_anchor_of({"model": "whatever", "causal_anchor": "21@-3"}) == "21@-3"


def test_causal_anchor_falls_back_to_the_registry_for_older_artifacts():
    """Reports written before the field existed must still get the contrast panel."""
    assert causal_anchor_of({"model": "gemma-3-4b"}) == "15@-2"


def test_causal_anchor_of_an_unknown_model_is_none_not_a_crash():
    assert causal_anchor_of({"model": "not-a-real-model"}) is None


def test_render_figures_skips_the_scatter_when_rows_carry_no_coords(tmp_path, capsys):
    """Older projections.json predate per-row PCA coords. Drawing something else instead
    would put a figure on screen that is not what its title claims."""
    per_site, rows = _fake_site(4, -2)
    report = {"model": "gemma-3-4b", "per_site": per_site,
              "positions": {"sites": [-2]}, "causal_anchor": "2@-2"}
    names = render_figures(report, rows, tmp_path)
    assert "fig3_pca.png" not in names
    assert not (tmp_path / "fig3_pca.png").exists()
    assert "skipping fig3" in capsys.readouterr().out


def test_render_figures_writes_one_profile_and_one_separation_per_position(tmp_path):
    per_site, rows = _fake_site(4, -2)
    per_site.update({f"{k.split('@')[0]}@-3": v for k, v in per_site.items()})
    for r in rows:
        r["proj"].update({f"{k.split('@')[0]}@-3": v for k, v in r["proj"].items()})
    report = {"model": "gemma-3-4b", "per_site": per_site,
              "positions": {"sites": [-2, -3]}, "causal_anchor": "2@-2"}
    names = render_figures(report, rows, tmp_path)
    for pos in (-2, -3):
        assert f"fig1_layer_profile_pos{pos}.png" in names
        assert f"fig4_separation_pos{pos}.png" in names


# ------------------------------------------------------------------ PCA sites


def test_parse_site_requires_a_position_not_just_a_layer():
    """A layer-only argument draws the right depth at the wrong token. The behavioural site
    is at a different position from the anchor in both models."""
    assert parse_site("21@-3") == (21, -3)
    assert parse_site("15@-2") == (15, -2)
    with pytest.raises(ValueError, match="layer@position"):
        parse_site("21")


def test_site_key_round_trips_through_parse_site():
    for layer, pos in ((21, -3), (0, -1), (33, -5)):
        assert parse_site(site_key(layer, pos)) == (layer, pos)


def test_render_figures_draws_one_scatter_per_pca_site(tmp_path):
    per_site, rows = _fake_site(4, -2)
    for i, r in enumerate(rows):
        r["pca"] = {"2@-2": [float(i), 0.0], "3@-3": [0.0, float(i)]}
    report = {"model": "gemma-3-4b", "per_site": per_site, "positions": {"sites": [-2]},
              "causal_anchor": "2@-2",
              "pca_explained": {"2@-2": [0.5, 0.2], "3@-3": [0.4, 0.3]}}
    names = render_figures(report, rows, tmp_path)
    assert "fig3_pca_2p-2.png" in names
    assert "fig3_pca_3p-3.png" in names


def test_a_pre_multisite_artifact_still_renders_its_single_scatter(tmp_path):
    """Older runs stored a bare [x, y] rather than a per-site dict."""
    per_site, rows = _fake_site(4, -2)
    for i, r in enumerate(rows):
        r["pca"] = [float(i), float(-i)]
    report = {"model": "gemma-3-4b", "per_site": per_site, "positions": {"sites": [-2]},
              "causal_anchor": "2@-2", "pca": {"layer": 2, "position": -2},
              "pca_explained": [0.5, 0.2]}
    assert "fig3_pca.png" in render_figures(report, rows, tmp_path)


# -------------------------------------------------------------- outlier energy
#
# Gemma's L15 direction puts 0.81 of its squared norm on dims (443, 634) -- measured
# once, at one layer, and then assumed everywhere. Recording it per site is what makes
# the raw/complement divergence visible instead of inferred.


def test_outlier_energy_is_the_squared_norm_fraction_on_those_dims():
    import torch

    v = torch.zeros(8, dtype=torch.float64)
    v[0], v[1], v[2] = 3.0, 4.0, 0.0     # ||v||^2 = 25, dim1 carries 16
    assert outlier_energy(v, (1,)) == pytest.approx(0.64)
    assert outlier_energy(v, (0, 1)) == pytest.approx(1.0)
    assert outlier_energy(v, (2,)) == pytest.approx(0.0)


def test_outlier_energy_is_scale_invariant():
    """It is a property of the DIRECTION, so ||r|| must not enter it."""
    import torch

    v = torch.tensor([3.0, 4.0], dtype=torch.float64)
    assert outlier_energy(v, (1,)) == pytest.approx(outlier_energy(v * 1000.0, (1,)))


def test_a_model_with_no_measured_outliers_has_zero_energy():
    """qwen3-0.6b and friends carry `outlier_dims=()`; complement == raw there."""
    import torch

    assert outlier_energy(torch.tensor([1.0, 2.0], dtype=torch.float64), ()) == 0.0


# ------------------------------------------------------- projections on disk (R23)
#
# The two-basis F1 doubled projections.json to ~5.7 MB (gemma) and ~6.0 MB (qwen),
# tripping the R23 size guard at 5 MB. `archive_results.PLAIN` had listed
# projections.json as "small and worth reading directly"; that stopped being true.
# gzip is 5.4x on this content, which is headroom rather than a shave.


def test_gzip_round_trip_preserves_float_values_exactly():
    """Targets are read straight out of this file, so a lossy round trip would move the
    experiment's numbers rather than just its bytes."""
    from jbct.data import read_json_maybe_gz, write_json_gz

    import tempfile

    rows = [{"proj": {"15@-2": -29983.812345678901, "21@-3": 2192.3706}},
            {"proj_complement": {"15@-2": 685.0743999999}}]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "projections.json"
        write_json_gz(p, rows)
        assert (Path(d) / "projections.json.gz").exists()
        assert not p.exists()
        assert read_json_maybe_gz(p) == rows


def test_gzip_bytes_are_deterministic():
    """Same rationale as scripts/archive_results.gzip_deterministic: mtime and the stored
    FNAME would otherwise make identical content produce differing archives, so a rerun
    would show a diff that is not a result."""
    import tempfile

    from jbct.data import write_json_gz

    rows = [{"proj": {"0@-2": 1.5}}]
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / "a.json", Path(d) / "b.json"
        write_json_gz(a, rows)
        write_json_gz(b, rows)
        assert (Path(d) / "a.json.gz").read_bytes() == (Path(d) / "b.json.gz").read_bytes()


def test_a_plain_file_is_still_readable(tmp_path):
    """Committed pre-gz artifacts (results/chain/*, docs/runs/*) must keep working."""
    from jbct.data import read_json_maybe_gz

    p = tmp_path / "projections.json"
    p.write_text(json.dumps([{"proj": {"0@-2": 1.0}}]))
    assert read_json_maybe_gz(p) == [{"proj": {"0@-2": 1.0}}]


def test_the_compressed_copy_wins_when_both_exist(tmp_path):
    """A stale plain file left over from a pre-gz run must not silently shadow the new one."""
    from jbct.data import read_json_maybe_gz, write_json_gz

    p = tmp_path / "projections.json"
    p.write_text(json.dumps([{"stale": True}]))
    write_json_gz(p, [{"stale": False}])
    assert read_json_maybe_gz(p) == [{"stale": False}]


def test_a_missing_file_names_both_paths_it_looked_for(tmp_path):
    from jbct.data import read_json_maybe_gz

    with pytest.raises(FileNotFoundError, match="projections.json"):
        read_json_maybe_gz(tmp_path / "projections.json")
