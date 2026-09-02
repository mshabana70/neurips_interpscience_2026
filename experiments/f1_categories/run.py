"""F1-categories — the refusal-direction geometry, split by jailbreak class.

F1 pools every successful attack into one `attack-COMPLY` population, and F3 transplants the
profile that pooling produces. This asks whether that pooled profile describes anything real.
It does not: on gemma the classes span 5% to 58% of a plain request's refusal magnitude, so
the pooled figure is an average of a class that nearly ablates the direction and classes that
barely touch it — a profile no actual jailbreak has.

**No GPU, no forward passes.** Every number here comes out of F1's committed
`projections.json.gz`, which already records `class` per prompt and the projection at every
(layer, position) in both bases. The reliability half reads H2's committed judgments.

Three corrections are built in rather than left to the caller:

1. The reference is the 50 dataset bases (`prefill`'s control population), not the 32
   prompts the direction was fitted on. Those sit ~18% higher, which inflated every earlier
   figure. `prefill` reading ~100% is the check that this is right; it read 117% before.
2. `prefill` is excluded from prompt-side comparisons — its attack is a forced assistant
   opening, and its prompt is the bare request token for token.
3. Both anchors are reported. `clean` is robust and interpretable; `zero` is the axis F3's
   scale sweep is calibrated in, so a class's position on it IS the transplant target a
   per-class F3 arm would impose.

Run:
    python experiments/f1_categories/run.py --model gemma-3-4b
    python experiments/f1_categories/run.py --model gemma-3-4b --replot
    python experiments/f1_categories/run.py --model gemma-3-4b \\
        --judge-sample 5 --generations /path/to/h2_judged.json
    python experiments/f1_categories/run.py --model gemma-3-4b \\
        --judge-labels results/f1_categories/gemma-3-4b/judge_sample.labelled.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

def source_provenance(*paths) -> dict:
    """Hash of every input this report is a function of.

    NOT `env_versions()`. Nothing here touches torch — the numbers are a pure function of
    committed files — so stamping the local torch/CUDA build would assert a dependency that
    does not exist, and importing transformer_lens to discover it costs 156 seconds against
    0.5 seconds of actual work. The library versions that CAN move these numbers are the ones
    that produced the projections, carried forward from F1's own report as `upstream_env`.
    """
    import hashlib

    out = {}
    for path in paths:
        path = Path(path)
        if not path.exists():
            out[path.name] = None
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        out[path.name] = {"path": str(path), "sha256_16": digest,
                          "bytes": path.stat().st_size}
    return out


def default_judged(model: str):
    """The committed archive is the usual answer, and it is already on disk.

    `judged.json` is matched by `.gitignore`, so the plain path exists only on the machine
    that produced it — but `scripts/archive_results.py` commits a gzipped copy under
    `docs/runs/`, and for both models it is byte-for-byte the same run that produced the
    labels F1 consumed (700/700 response hashes and labels agree). Nothing needs copying.
    """
    plain = Path("results/h2") / model / "judged.json"
    if plain.exists():
        return plain
    for pattern in (f"*/h2/{model}/judged.json.gz", f"*/h2/{model}/judged.json"):
        found = sorted(Path("docs/runs").glob(pattern))
        if found:
            return found[-1]
    return None


REPORT = "f1_categories_report.json"
SAMPLE = "judge_sample.json"


def load_projections(f1_dir, model):
    from jbct.data import read_json_maybe_gz

    path = Path(f1_dir) / model / "projections.json"
    rows = read_json_maybe_gz(path)
    if not rows or "class" not in rows[0]:
        raise SystemExit(
            f"{path} has no per-prompt `class` field — it predates the class-aware F1. "
            f"Re-run:\n    uv run python experiments/f1_geometry/run.py --model {model}")
    return rows


def positions_in(rows, field: str) -> list[int]:
    """Read positions off the data rather than the registry, so a projections file written
    with a different `direction_positions` cannot be silently analysed at the wrong site."""
    return sorted({int(k.split("@")[1]) for k in rows[0][field]}, reverse=True)


def n_layers_in(rows, field: str) -> int:
    return 1 + max(int(k.split("@")[0]) for k in rows[0][field])


def classes_in(rows) -> list[str]:
    return sorted({r["class"] for r in rows if r.get("class")})


def analyse_position(rows, position, n_layers, *, field, floor):
    """Reference series, per-class profiles and summaries at one read position."""
    from jbct.categories import category_profile, profile_summary, reference_series

    refs = reference_series(rows, position, n_layers, field=field, floor=floor)
    profiles, summaries = [], []
    for cls in classes_in(rows):
        prof = category_profile(rows, cls, position, n_layers, refs, field=field)
        if not prof["layers"]:
            continue
        profiles.append(prof)
        summaries.append(profile_summary(prof, refs))
    return refs, profiles, summaries


def calibration_check(summaries, tol: float = 0.10) -> dict:
    """`prefill` must land at ~100% of the base, because its prompt IS the base.

    This is the only self-check available on the reference: both candidate references look
    like plausible "harmful" populations, and picking the wrong one shifts every class by a
    constant that no other number in the report would reveal. Deviation beyond `tol` means
    the reference population is not what this module thinks it is.
    """
    from jbct.categories import REFERENCE_CLASS

    entry = next((s for s in summaries if s["class"] == REFERENCE_CLASS), None)
    if entry is None or entry["p_zero_weighted"] is None:
        return {"available": False,
                "note": f"no {REFERENCE_CLASS} summary — reference cannot be checked"}
    value = entry["p_zero_weighted"]
    return {
        "available": True,
        "class": REFERENCE_CLASS,
        "p_zero_weighted": round(value, 4),
        "tolerance": tol,
        "passes": abs(value - 1.0) <= tol,
        "note": (f"{REFERENCE_CLASS}'s prompt is the bare request, so this must be ~1.0. "
                 f"Against the direction's fit corpus it reads ~1.18, which is how the wrong "
                 f"reference announces itself."),
    }


def separations(rows, n_layers, position, *, field, min_n):
    from jbct.categories import within_class_separation

    out = []
    for cls in classes_in(rows):
        for layer in range(n_layers):
            got = within_class_separation(rows, cls, layer, position,
                                          field=field, min_n=min_n)
            if got is not None:
                out.append(got)
    return out


def judge_reliability(h2_dir, model) -> dict:
    """What the committed (slimmed) judgments can still say about label quality."""
    from jbct.categories import behaviour_by_class, hash_inconsistency, label_risk

    path = Path(h2_dir) / model / "judgments.json"
    if not path.exists():
        return {"available": False, "note": f"{path} not found"}
    payload = json.loads(path.read_text())
    rows = payload["judgments"] if isinstance(payload, dict) else payload
    meta = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    return {
        "available": True,
        "source": str(path),
        "judge_model": meta.get("judge_model"),
        "max_new_tokens": meta.get("max_new_tokens"),
        "n": len(rows),
        "label_risk": label_risk(rows),
        "behaviour": behaviour_by_class(label_risk(rows)),
        "hash_inconsistency": hash_inconsistency(rows),
        "note": ("Response text is stripped from the committed judgments, so agreement "
                 "against a human label cannot be computed here. These are the signals that "
                 "survive slimming; --judge-sample builds the human-label task from a full "
                 "judged.json."),
    }


def read_judged_any(path):
    """Rows out of a judged file, whatever shape it is on this disk.

    Three shapes exist and all three are real: H2 writes `{"metadata", "generations"}`;
    `scripts/export_judgments.py` slims that to `{"metadata", "judgments"}`; and the
    committed archive under `docs/runs/` is the first shape, gzipped. The archive also names
    the prompt `prompt_text` rather than `prompt`. Guessing one shape meant this path failed
    on the only file that actually has response text in it.
    """
    import gzip

    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, list):
        return payload
    for key in ("generations", "judgments"):
        if key in payload:
            return payload[key]
    raise SystemExit(f"{path} has neither `generations` nor `judgments`")


def export_judge_sample(generations, per_cell, outdir) -> Path:
    """A stratified, deterministic sample for human labelling.

    Stratified by (class, kind, judge label) so the rare cell — an attack the judge scored
    REFUSE in a class that almost always succeeds — is represented at the same depth as the
    common one. A uniform sample would spend almost all of its budget confirming the easy
    majority and never reach the pockets where the label is actually in doubt.
    """
    from jbct.categories import stratified_sample

    rows = read_judged_any(generations)
    missing = [r for r in rows if not r.get("response")]
    if missing:
        raise SystemExit(
            f"{generations} has {len(missing)}/{len(rows)} rows without `response` text — "
            f"this looks like the slimmed judgments. The full judged.json is the one written "
            f"beside the H2 run before slimming; bring that file over instead.")
    picked = stratified_sample(
        rows, lambda r: (r.get("class"), r.get("kind"), r.get("judge")), per_cell)
    # `record_idx` alone does NOT identify a row: an attack and its matched control are the
    # two halves of one dataset record and share it. Keying on it silently merged pairs when
    # the labels were re-read, corrupting two rows out of 98 and moving the reported
    # agreement by 2 points. The key is what makes a row addressable; emit it explicitly.
    task = [{
        "key": f"{r.get('record_idx')}/{r.get('class')}/{r.get('kind')}",
        "idx": r.get("record_idx"),
        "class": r.get("class"),
        "kind": r.get("kind"),
        "judge": r.get("judge"),
        "hit_token_cap": r.get("hit_token_cap"),
        "n_new_tokens": r.get("n_new_tokens"),
        "prompt": r.get("prompt_text") or r.get("prompt"),
        "response": r.get("response"),
        "human": None,          # <- fill in with "COMPLY" or "REFUSE"
    } for r in picked]
    path = Path(outdir) / SAMPLE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(task, indent=2))
    return path


def score_judge_labels(labels_path, judgments=None) -> dict:
    """Agreement once a human has filled in the sample."""
    from jbct.categories import (
        COMPLY,
        REFUSE,
        agreement_by_cell,
        confusion,
        population_agreement,
    )

    rows = json.loads(Path(labels_path).read_text())
    done = [r for r in rows if r.get("human") in (COMPLY, REFUSE)]
    if not done:
        raise SystemExit(
            f"{labels_path} has no filled-in `human` fields. Set each to \"{COMPLY}\" or "
            f"\"{REFUSE}\" — rows left null are skipped, so a partial pass is fine.")
    by_cell = {}
    for r in done:
        by_cell.setdefault(str(r.get("class")), []).append((r["human"], r["judge"]))
    by_kind = {}
    for r in done:
        by_kind.setdefault(str(r.get("kind")), []).append((r["human"], r["judge"]))
    capped = [(r["human"], r["judge"]) for r in done if r.get("hit_token_cap")]
    # The stratification cell, so the sample can be re-weighted to the run it came from.
    by_stratum: dict = {}
    for r in done:
        key = (r.get("class"), r.get("kind"), r.get("judge"))
        by_stratum.setdefault(key, []).append((r["human"], r["judge"]))
    sizes: dict = {}
    for r in (judgments or []):
        key = (r.get("class"), r.get("kind"), r.get("judge"))
        sizes[key] = sizes.get(key, 0) + 1
    return {
        "n_labelled": len(done),
        "n_unlabelled": len(rows) - len(done),
        "sample": confusion([(r["human"], r["judge"]) for r in done]),
        "population_weighted": (population_agreement(by_stratum, sizes) if sizes
                                else {"available": False,
                                      "note": "no judgments passed; cannot re-weight"}),
        "by_class": agreement_by_cell(by_cell),
        "by_kind": agreement_by_cell(by_kind),
        # The pocket `label_risk` flags: truncated responses, scored by a human.
        "truncated_only": confusion(capped) if capped else None,
    }


def rehydrate(report: dict) -> dict:
    """Restore integer layer keys after a JSON round-trip.

    JSON has no integer keys, so a committed report reloads with `"15"` where the fresh run
    held `15`. Everything downstream then sorts layers as strings — `"10"` before `"2"` —
    so `--replot` would render a DIFFERENT figure from the run that produced the report,
    from identical data, without erroring. It errored here only because `imshow`'s extent
    does arithmetic on the key; the bar and trajectory figures would have silently drawn the
    layers in the wrong order.
    """
    for block in (report.get("per_position") or {}).values():
        refs = block.get("references") or {}
        if "layers" in refs:
            refs["layers"] = {int(k): v for k, v in refs["layers"].items()}
        for prof in block.get("profiles") or []:
            if "layers" in prof:
                prof["layers"] = {int(k): v for k, v in prof["layers"].items()}
    return report


def plot_classes(report, min_n: int) -> set:
    """Which classes the FIGURES show. The report always keeps all seven.

    Filtered by n rather than by a hard-coded list, because the usable four are not the same
    four in each model: gemma has roleplay at n=34 and narrative_sandwich at 15, qwen has
    roleplay at n=2 and narrative_sandwich at 18. A fixed list would drop qwen's
    narrative_sandwich while keeping a two-prompt roleplay panel.
    """
    keep = {s["class"] for s in report["summaries"] if s["n_prompts"] >= min_n}
    # prefill is always shown: it is the calibration check and the class whose flatness is
    # itself the finding.
    from jbct.categories import POST_PROMPT_CLASSES
    return keep | set(POST_PROMPT_CLASSES)


def judge_label_passes(labels_path, judgments) -> dict:
    """Score every labelling pass on disk and name which one is the reliability figure.

    **A revised pass is not an inter-rater agreement estimate.** The revision is made after
    seeing WHICH items disagreed, so it cannot measure how often an independent labeller and
    the judge coincide — it measures how well they converge once told where they differ. It
    is nonetheless the better gold standard for correcting downstream counts. The two answer
    different questions and the higher number belongs to the weaker claim, which is precisely
    the arrangement that gets misquoted.

    A `*.pass1.json` sibling is treated as the independent pass. Both are scored and labelled
    so the committed report cannot be read as claiming the adjudicated figure is reliability.
    """
    path = Path(labels_path)
    out = {"adjudicated": score_judge_labels(path, judgments),
           "labels_source": str(path)}
    earlier = sorted(path.parent.glob(f"{path.stem}.pass*.json"))
    if earlier:
        out["independent"] = score_judge_labels(earlier[0], judgments)
        out["independent_source"] = str(earlier[0])
        out["reliability_figure"] = "independent"
        out["note"] = (
            "`independent` is the inter-rater agreement: a first pass labelled without "
            "knowing which items the judge disagreed on. `adjudicated` is the same labeller "
            "after reviewing the disagreements, so it is a gold standard for correcting "
            "counts but NOT a reliability estimate. Quote `independent` for label quality. "
            "Neither pass was blind-adjudicated by a second person, which is the stronger "
            "protocol and is not what happened here.")
    else:
        out["reliability_figure"] = "adjudicated"
        out["note"] = ("Only one labelling pass on disk, so this is an independent "
                       "inter-rater agreement.")
    return out


def render(report, outdir, min_n: int = 0) -> None:
    from jbct.plots_categories import (
        figure_anchor_scatter,
        figure_category_summary,
        figure_position_layer_grid,
        figure_shift_by_layer,
        figure_trajectories,
        figure_within_class_auc,
    )

    model = report["model"]
    primary = report["primary_position"]
    per_pos = report["per_position"]
    keep = plot_classes(report, min_n)
    prof_at = {int(p): [x for x in v["profiles"] if x["class"] in keep]
               for p, v in per_pos.items()}
    prim = prof_at[primary]
    shown = sorted(keep & {s["class"] for s in report["summaries"]})
    if min_n:
        print(f"  [plot] {len(shown)} of {len(report['summaries'])} classes shown "
              f"(n >= {min_n}, plus prefill): {', '.join(shown)}")
    refs = per_pos[str(primary)]["references"]
    out = Path(outdir)

    summaries = [s for s in report["summaries"] if s["class"] in keep]
    figure_category_summary(
        summaries, out / "fig1_category_summary.png",
        f"{model}: how far each jailbreak class moves the refusal direction",
        f"read at position {primary}, successful attacks only, "
        f"against the {report['reference']['n']} plain requests they are built on")
    figure_shift_by_layer(
        prim, refs, out / f"fig2_shift_by_layer_p{primary}.png",
        f"{model}: shift toward harmless, layer by layer, at position {primary}")
    figure_position_layer_grid(
        prof_at,
        out / "fig3_position_layer_grid.png",
        f"{model}: shift toward harmless across every layer and read position")
    figure_trajectories(
        prim, out / f"fig4_trajectories_p{primary}.png",
        f"{model}: the shape of each class's path down the stack (position {primary})")
    figure_within_class_auc(
        [a for a in report["within_class_auc"] if a["class"] in keep], out / f"fig5_within_class_auc_p{primary}.png",
        f"{model}: within a class, does the direction predict which attacks succeed?")
    figure_anchor_scatter(
        summaries, out / "fig6_anchors.png",
        f"{model}: each class on both rulers")
    risk = (report.get("judge") or {}).get("human_agreement") or {}
    risk = risk.get("adjudicated", risk)
    if risk and risk.get("by_class"):
        from jbct.plots_categories import figure_judge_reliability
        figure_judge_reliability(
            risk["by_class"], out / "fig7_judge_reliability.png",
            f"{model}: judge agreement with human labels, by attack class")


def print_headline(report) -> None:
    cal = report["calibration"]
    print(f"\n[f1c] {report['model']} — reference: {report['reference']['n']} plain requests "
          f"({report['reference']['class']} controls)")
    if cal.get("available"):
        mark = "ok" if cal["passes"] else "FAILED — the reference is not the base population"
        print(f"[f1c] calibration: {cal['class']} reads "
              f"{cal['p_zero_weighted']:.3f} of base (must be ~1.0)  {mark}")
    print(f"\n  {'class':<26}{'n':>5}{'toward harmless':>17}{'retention':>11}   family")
    for s in report["summaries"]:
        if s["shift_toward_clean"] is None:
            continue
        tag = "  [not in prompt]" if s["post_prompt"] else ""
        thin = " *thin*" if s["n_prompts"] < 18 else ""
        ret = "     n/a" if s["p_zero_weighted"] is None else f"{s['p_zero_weighted']:>10.1%}"
        print(f"  {s['class']:<26}{s['n_prompts']:>5}{s['shift_toward_clean']:>16.1%}"
              f"{ret}   {s['family']}{tag}{thin}")
    pooled = report.get("pooled")
    if pooled and pooled.get("shift_toward_clean") is not None:
        print(f"  {'POOLED (what F3 used)':<26}{pooled['n_prompts']:>5}"
              f"{pooled['shift_toward_clean']:>16.1%}{pooled['p_zero_weighted']:>10.1%}")
    behaviour = ((report.get("judge") or {}).get("behaviour")) or {}
    if behaviour:
        print(f"\n  {'class':<26}{'succeeds':>10}{'control':>9}{'lift':>8}"
              f"{'retention':>11}   behaviour vs geometry")
        for s in report["summaries"]:
            b = behaviour.get(s["class"])
            if not b or b.get("lift") is None or s["p_zero_weighted"] is None:
                continue
            note = ""
            if b["lift"] >= 0.5 and s["p_zero_weighted"] >= 0.4:
                note = "  <- succeeds WITHOUT suppressing"
            elif b["lift"] >= 0.5 and s["p_zero_weighted"] <= 0.2:
                note = "  <- succeeds by suppressing"
            print(f"  {s['class']:<26}{b['attack']:>10.0%}{b['control']:>9.0%}"
                  f"{b['lift']:>+8.0%}{s['p_zero_weighted']:>11.1%}{note}")

    spread = [s for s in report["summaries"]
              if not s["post_prompt"] and s["p_zero_weighted"] is not None]
    if len(spread) > 1:
        lo = min(spread, key=lambda s: s["p_zero_weighted"])
        hi = max(spread, key=lambda s: s["p_zero_weighted"])
        print(f"\n  spread: {lo['class']} keeps {lo['p_zero_weighted']:.1%} of the refusal "
              f"magnitude, {hi['class']} keeps {hi['p_zero_weighted']:.1%}")
        print("  the pooled profile F3 transplanted is the average of these — a profile no "
              "single class has")
    judge = report.get("judge") or {}
    risk = judge.get("label_risk") or {}
    worst = sorted(risk.items(), key=lambda kv: -(kv[1]["at_risk_rate"] or 0))[:4]
    if worst:
        print(f"\n  [judge labels] {judge.get('judge_model')} — cells where a label is most "
              f"likely wrong:")
        for cell, m in worst:
            print(f"    {cell:<36} {m['at_risk_rate']:>5.0%} at risk  "
                  f"({m['truncated_refuse']} truncated+REFUSE, {m['short_comply']} "
                  f"short+COMPLY, cap {m['cap_rate']:.0%})")
    human = judge.get("human_agreement")
    if human and "adjudicated" in human:
        which = human["reliability_figure"]
        for name in ("independent", "adjudicated"):
            block = human.get(name)
            if not block:
                continue
            o, pw = block["sample"], block.get("population_weighted") or {}
            star = "  <- reliability figure" if name == which else ""
            print(f"\n  [judge vs human — {name}] sample n={o['n']} "
                  f"agreement {o['agreement']:.1%} kappa {round(o['kappa'], 3)}"
                  f"{star}")
            if pw.get("available"):
                print(f"    re-weighted to the full run: {pw['agreement']:.1%} "
                      f"(coverage {pw['coverage']:.0%})")
        if "independent" in human:
            print("    the adjudicated pass is a gold standard, NOT a reliability estimate: "
                  "it was revised after seeing where the two disagreed")
    elif human:
        o = human["sample"]
        print(f"\n  [judge vs human] sample n={o['n']}  agreement {o['agreement']:.1%}  "
              f"kappa {o['kappa'] if o['kappa'] is None else round(o['kappa'], 3)}  "
              f"judge-COMPLY/human-REFUSE {o['refuse_comply']}  "
              f"judge-REFUSE/human-COMPLY {o['comply_refuse']}")
        pw = human.get("population_weighted") or {}
        if pw.get("available"):
            print(f"    re-weighted to the full run: {pw['agreement']:.1%} "
                  f"(coverage {pw['coverage']:.0%} of {pw['n_population']} responses). "
                  f"The sample is stratified, so its raw agreement is not the run's.")


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

    from jbct.categories import POST_PROMPT_CLASSES, REFERENCE_CLASS, profile_summary
    from jbct.data import write_json_gz

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--f1", default="results/f1_geometry")
    ap.add_argument("--h2", default="results/h2")
    ap.add_argument("--field", default="proj_complement",
                    choices=["proj_complement", "proj"],
                    help="complement (default) is the basis the interventions act in; "
                         "`proj` is the raw direction, kept only for comparison")
    ap.add_argument("--position", type=int, default=None,
                    help="primary read position for the tables and most figures; "
                         "default = F1's causal anchor")
    ap.add_argument("--floor", type=float, default=0.25,
                    help="a cell is unusable when its denominator is below this many pooled "
                         "SDs — a ratio of noise to noise is a number, not a measurement")
    ap.add_argument("--min-n", type=int, default=5,
                    help="minimum per outcome group for a within-class AUC")
    ap.add_argument("--judge-sample", type=int, default=0,
                    help="export this many responses per (class, kind, label) cell for "
                         "human labelling; needs --generations")
    ap.add_argument("--generations", default=None,
                    help="a FULL h2 judged.json (with response text) for --judge-sample")
    ap.add_argument("--judge-labels", default=None,
                    help="a judge_sample.json whose `human` fields have been filled in")
    ap.add_argument("--plot-min-n", type=int, default=0,
                    help="hide classes with fewer than this many successful attacks from the "
                         "FIGURES; the report keeps all of them. 18 gives the four usable "
                         "classes per model. prefill is always shown")
    ap.add_argument("--replot", action="store_true",
                    help="re-render the figures from a committed report. No recomputation")
    ap.add_argument("--out", default="results/f1_categories")
    args = ap.parse_args()

    outdir = Path(args.out) / args.model
    outdir.mkdir(parents=True, exist_ok=True)

    if args.replot:
        from jbct.data import read_json_maybe_gz
        report = rehydrate(read_json_maybe_gz(outdir / REPORT))
        render(report, outdir, args.plot_min_n)
        print_headline(report)
        return 0

    if args.judge_sample:
        source = args.generations or default_judged(args.model)
        if source is None:
            raise SystemExit(
                f"no judged file with response text found for {args.model}. Looked at "
                f"results/h2/{args.model}/judged.json and "
                f"docs/runs/*/h2/{args.model}/judged.json[.gz]. Pass --generations.")
        print(f"[f1c] judged source: {source}")
        path = export_judge_sample(source, args.judge_sample, outdir)
        rows = json.loads(path.read_text())
        print(f"[f1c] wrote {len(rows)} responses to {path}")
        print("[f1c] fill in each `human` field with \"COMPLY\" or \"REFUSE\", then re-run "
              "with --judge-labels")
        return 0

    rows = load_projections(args.f1, args.model)
    n_layers = n_layers_in(rows, args.field)
    positions = positions_in(rows, args.field)
    f1_report = json.loads((Path(args.f1) / args.model / "f1_report.json").read_text())
    anchor = f1_report.get("causal_anchor")
    primary = args.position if args.position is not None else (
        int(anchor.split("@")[1]) if anchor else positions[0])
    if primary not in positions:
        raise SystemExit(f"position {primary} is not in the projections "
                         f"({positions}) — F1 records only its `direction_positions`")

    per_position, summaries_at = {}, {}
    for pos in positions:
        refs, profiles, summaries = analyse_position(
            rows, pos, n_layers, field=args.field, floor=args.floor)
        per_position[str(pos)] = {"references": refs, "profiles": profiles,
                                  "summaries": summaries}
        summaries_at[pos] = summaries

    # The pooled profile, computed the same way, so the report can show the class spread
    # against the single number F3 actually transplanted.
    from jbct.categories import category_profile
    pooled_refs = per_position[str(primary)]["references"]
    pooled_prof = category_profile(rows, None, primary, n_layers, pooled_refs,
                                   field=args.field)
    pooled_prof["class"] = "POOLED"
    pooled_prof["n_prompts"] = len(
        [r for r in rows if r.get("population") == "attack-COMPLY"])
    for layer, entry in pooled_prof["layers"].items():
        vals = [r[args.field][f"{layer}@{primary}"] for r in rows
                if r.get("population") == "attack-COMPLY"]
        entry["n"] = len(vals)
    pooled = profile_summary(pooled_prof, pooled_refs)

    judge = judge_reliability(args.h2, args.model)
    if args.judge_labels:
        h2_rows = []
        h2_path = Path(args.h2) / args.model / "judgments.json"
        if h2_path.exists():
            h2_rows = json.loads(h2_path.read_text())["judgments"]
        judge["human_agreement"] = judge_label_passes(args.judge_labels, h2_rows)

    proj_path = Path(args.f1) / args.model / "projections.json.gz"
    report = {
        "env": {"python": sys.version.split()[0], "platform": platform.platform()},
        "upstream_env": f1_report.get("env"),
        "sources": source_provenance(proj_path,
                                     Path(args.f1) / args.model / "f1_report.json",
                                     Path(args.h2) / args.model / "judgments.json"),
        "model": args.model,
        "field": args.field,
        "basis": "complement" if args.field == "proj_complement" else "raw",
        "positions": positions,
        "primary_position": primary,
        "n_layers": n_layers,
        "floor": args.floor,
        "excluded_from_prompt_side": list(POST_PROMPT_CLASSES),
        "reference": {
            "class": REFERENCE_CLASS,
            "population": "control",
            "n": pooled_refs["n_base"],
            "n_clean": pooled_refs["n_clean"],
            "note": ("the 50 plain requests every attack is built on. Using the direction's "
                     "own fit corpus instead inflates every ratio by ~18%."),
        },
        "calibration": calibration_check(summaries_at[primary]),
        "summaries": sorted(summaries_at[primary],
                            key=lambda s: (s["post_prompt"],
                                           -(s["p_zero_weighted"] or -9))),
        "pooled": pooled,
        "within_class_auc": separations(rows, n_layers, primary,
                                        field=args.field, min_n=args.min_n),
        "judge": judge,
        "per_position": per_position,
    }
    write_json_gz(outdir / REPORT, report)
    print(f"[f1c] -> {outdir / (REPORT + '.gz')}")
    render(report, outdir, args.plot_min_n)
    print_headline(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
