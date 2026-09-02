"""F3-transplant — impose a successful jailbreak's full per-layer refusal profile.

F3's single-layer clamps left one question open. Zeroing the refusal direction at the causal
anchor induces compliance (gemma 15@-2, all positions, target ~0: 0.813 from a 0.113
baseline); clamping it to the jailbroken level does not (target 339.8: 0.107). That reads as
"necessary but not sufficient" — except the test was not faithful. It moved ONE layer, while
a jailbreak reshapes the whole trajectory, and the one broad run imposed the position -2
value on every token.

So the literal F3 claim has never been tested at full fidelity. This does that: every layer
clamped, at the measured position, each to its own jailbroken mean along its own direction.

**This is not "alpha=1 again".** alpha=1 means something different at every layer. From F1's
per-layer complement means at gemma -2, the jailbroken projection as a share of harmful is
33% at L12, 18% at L13, **7% at L14**, then 50% at L15. The jailbreak's suppression is
BANDED — near-total just upstream of the causal anchor, partial elsewhere. A faithful
transplant reproduces that band; no single-site clamp can.

Arms:
  baseline    no intervention.
  null        every layer clamped to its own HARMFUL mean. Moves no mean, so it prices the
              act of clamping 30+ layers at once.
  transplant  every layer clamped to its own JAILBROKEN mean. The experiment.
  zero        every layer clamped to 0. The scope-matched ceiling — without it, a middling
              transplant cannot be told from "clamping many layers is just weak".
  sham_*      the same per-prompt per-layer shifts along a random direction.

Read it as `(transplant - null) / (zero - null)`: the share of the achievable effect that the
jailbroken profile captures.

**The sweep (`--scales`) is what turns the answer into a number.** The three arms above can
only say the jailbroken profile does little; they cannot say why. Adding points between
`null` and `zero` fits the threshold instead: compliance appears at retention <= X, and the
jailbreak reaches only 29% (gemma) / 47% (qwen). "Their suppression is too shallow to act
alone" is a claim; "the transplant was not significant" is an absence of one.

Every arm is filed by its magnitude-WEIGHTED RETENTION rather than by its scale, so both
scale axes and the transplant itself land on one comparable x. `--scale-axis retention`
scales the harmful profile uniformly (s IS the retention); `--scale-axis jailbreak`
interpolates toward, and past, the jailbreak's own banded shape.

Run:
    python experiments/f3_transplant/run.py --model gemma-3-4b --position -2 --n 150
    python experiments/f3_transplant/run.py --model gemma-3-4b --position -2 \
        --scales 0.7,0.5,0.35,0.2,0.1
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
from pathlib import Path

# The order arms run in. `baseline` first so a broken judge is caught before the expensive
# arms, then null/transplant/zero so the ratio's three terms land together.
PROFILES = ("null", "transplant", "zero")
ARMS = ("baseline", *PROFILES)

# Which population mean each profile clamps to. `zero` has no population — it is ablation.
PROFILE_POPULATION = {"null": "harmful", "transplant": "attack-COMPLY", "zero": None}


def site(layer: int, position: int) -> str:
    return f"{layer}@{position}"


def require_layers(f1_report: dict, position: int, layers) -> None:
    """Every layer must be present in F1's complement block, or fail naming what is missing.

    Silently skipping a layer would make the transplant a PARTIAL profile reported under a
    name that claims otherwise — the quiet-wrong-number failure this codebase specialises in.
    """
    comp = f1_report.get("per_site_complement") or {}
    missing = [site(layer, position) for layer in layers if site(layer, position) not in comp]
    if missing:
        raise SystemExit(
            f"F1 has no complement projections at {', '.join(missing[:6])}"
            f"{' ...' if len(missing) > 6 else ''}. Either the position was never measured "
            f"(F1 covers only its `direction_positions`) or the report predates the "
            f"two-basis F1. Re-run:\n"
            f"    uv run python experiments/f1_geometry/run.py --model <model>"
        )


def profile_targets(f1_report: dict, position: int, layers, profile: str) -> dict:
    """Per-layer target for one profile, from F1's committed per-population means."""
    if profile not in PROFILE_POPULATION:
        raise ValueError(f"unknown profile {profile!r}; have {sorted(PROFILE_POPULATION)}")
    pop = PROFILE_POPULATION[profile]
    if pop is None:
        return dict.fromkeys((int(x) for x in layers), 0.0)
    comp = f1_report["per_site_complement"]
    return {int(layer): float(comp[site(layer, position)][pop]["mean"]) for layer in layers}


# One definition, in the library, imported by both users. The per-class analysis places each
# attack class's own suppression on the SAME axis as this experiment's swept threshold; two
# copies free to drift would put them on different rulers without ever raising.
from jbct.categories import weighted_retention  # noqa: E402


def suppression_profile(targets: dict, layers) -> dict:
    """How much refusal magnitude the jailbroken profile leaves in place, vs the harmful one.

    Reported BEFORE any GPU time is spent, because it decides whether the experiment can
    answer anything: if the jailbroken profile is already ~0 everywhere then `transplant` and
    `zero` are the same intervention and `mediation_fraction` cannot separate them.

    **Magnitude-weighted, not a mean of per-layer ratios.** Early layers carry near-zero
    projections (qwen L9 harmful is 0.03), so a per-layer ratio there is a huge number
    computed from noise; averaging those gave 175% for qwen — an artifact, not suppression.
    The weighted figure answers the question that matters, "across the residual stream as a
    whole, what fraction of the refusal magnitude survives", and the median is reported
    alongside it as a shape check.
    """
    tr, nu = targets["transplant"], targets["null"]
    ratios = sorted(abs(tr[layer]) / abs(nu[layer])
                    for layer in layers if abs(nu[layer]) > 1e-9)
    return {
        "weighted": weighted_retention(tr, nu, layers),
        "median_per_layer": st.median(ratios) if ratios else None,
        "n_layers": len(layers),
    }


# How a sweep point's scalar is turned into a per-layer profile. Two axes, because they ask
# different questions and a single one would silently answer neither.
SCALE_AXES = ("retention", "jailbreak")


def scale_arm(scale: float) -> str:
    """Arm label for one sweep point.

    `%g` so that 0.50 and 0.5 name the SAME arm. Without it a resumed run written with a
    differently formatted scale lands in a second generations file, and the sweep gains a
    duplicate point at one retention that the curve then treats as two measurements.
    """
    return f"scale@{scale:g}"


def parse_scales(text: str) -> tuple[float, ...]:
    """Sweep points, de-duplicated by arm label, ordered from the null end to the zero end.

    Negatives are refused. On the `retention` axis a negative scale flips the sign of the
    refusal projection — a real and interesting arm, but `weighted_retention` takes an
    absolute value, so it would be filed at |s| alongside the positive arm of the same
    magnitude. Two different interventions at one x-coordinate is the quiet-wrong-number
    shape this file exists to avoid; an anti-refusal arm needs its own axis.
    """
    seen: set[str] = set()
    out: list[float] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = float(chunk)
        if value < 0:
            raise SystemExit(
                f"negative scale {value} — the retention axis is an absolute magnitude, so a "
                f"sign flip would be filed at |{value}| next to the positive arm. Run an "
                f"anti-refusal arm as its own experiment."
            )
        label = scale_arm(value)
        if label in seen:
            continue
        seen.add(label)
        out.append(value)
    return tuple(sorted(out, reverse=True))


def scaled_targets(targets: dict, layers, scale: float, axis: str = "retention") -> dict:
    """Per-layer targets for one sweep point.

    `retention` — ``target[L] = s * harmful[L]``. A uniform fraction of the harmful refusal
    magnitude at every layer. s=1 IS the `null` arm and s=0 IS the `zero` arm, so the sweep
    interpolates between the design's own two reference arms rather than between two new
    ones. Its weighted retention is exactly s, which is what makes a threshold quotable on
    the same ruler as the jailbreak's own 29% (gemma) / 47% (qwen).

    `jailbreak` — ``target[L] = harmful[L] + s * (jailbroken[L] - harmful[L])``. s=0 is
    `null`, s=1 is `transplant`, and s>1 extrapolates ALONG THE JAILBREAK'S OWN SHAPE. The
    jailbreak's suppression is banded — 7% of harmful at gemma L14 against 50% at L15 — so
    this asks whether the shape is right and merely too shallow. Uniform scaling cannot ask
    that question, and reading a uniform sweep as though it could would answer it wrongly.
    """
    if axis not in SCALE_AXES:
        raise ValueError(f"unknown scale axis {axis!r}; have {list(SCALE_AXES)}")
    null, tr = targets["null"], targets["transplant"]
    idx = [int(layer) for layer in layers]
    if axis == "retention":
        return {layer: scale * null[layer] for layer in idx}
    return {layer: null[layer] + scale * (tr[layer] - null[layer]) for layer in idx}


def mcnemar_exact(before: dict, after: dict) -> dict:
    """Exact two-sided McNemar over prompts present in BOTH arms.

    Every arm runs the same 150 prompts, so the comparison is paired and the unpaired
    heuristic in `specificity()` throws that information away. It cost a verdict: gemma's
    `zero` arm read "NOT specific" (+0.120 real against +0.053 sham) while the paired test
    on the same data gives 14 gained / 4 lost, p = 0.031.

    `gained` = refused before, complied after. Only discordant pairs carry information,
    which is the whole point of pairing — the ~130 prompts that refuse in both arms are
    noise in an unpaired proportion test and are correctly ignored here.
    """
    keys = sorted(set(before) & set(after))
    gained = sum(1 for k in keys if after[k] and not before[k])
    lost = sum(1 for k in keys if before[k] and not after[k])
    n = gained + lost
    if n == 0:
        p = 1.0
    else:
        tail = sum(math.comb(n, i) for i in range(min(gained, lost) + 1))
        p = min(1.0, 2 * tail / 2**n)
    return {"gained": gained, "lost": lost, "net": gained - lost,
            "n_pairs": len(keys), "p": round(p, 6)}


def mediation_ci(null: dict, transplant: dict, zero: dict,
                 n_boot: int = 4000, seed: int = 0) -> dict:
    """Bootstrap CI for the mediation fraction, resampling PROMPTS.

    The point estimate alone is not reportable: gemma's 36.8% carries a 95% CI of roughly
    [10%, 61%] and qwen's 8.3% spans zero. A ratio of two small differences over 150 prompts
    is far noisier than three decimal places of compliance rate suggest.

    Seeded locally rather than through the global RNG so a rescore is reproducible.
    """
    rng = random.Random(seed)
    keys = sorted(set(null) & set(transplant) & set(zero))

    def frac(sample):
        n = sum(null[k] for k in sample) / len(sample)
        t = sum(transplant[k] for k in sample) / len(sample)
        z = sum(zero[k] for k in sample) / len(sample)
        return None if abs(z - n) < 1e-9 else (t - n) / (z - n)

    point = frac(keys)
    boots = sorted(f for _ in range(n_boot)
                   if (f := frac([rng.choice(keys) for _ in keys])) is not None)
    if not boots or point is None:
        return {"point": point, "lo": None, "hi": None, "excludes_zero": False}
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]
    return {"point": round(point, 4), "lo": round(lo, 4), "hi": round(hi, 4),
            "excludes_zero": lo > 0, "n_boot": len(boots)}


def load_labels(outdir, arm: str) -> dict | None:
    """{prompt: complied} for one arm, from its committed generations."""
    path = Path(outdir) / f"{arm}_generations.json"
    if not path.exists():
        return None
    return {r["prompt"]: (r["judge"] == "COMPLY") for r in json.loads(path.read_text())}


def arm_labels(outdir) -> list[str]:
    """Profile arms present on disk — everything that is not the baseline or a sham.

    Derived from the files rather than from `PROFILES` so a scale sweep is picked up
    automatically, including by `--rescore` on a run whose report predates this function.
    A sweep arm silently absent from the paired table would read as a clean three-profile
    run rather than as a truncated sweep.
    """
    found = []
    for path in sorted(Path(outdir).glob("*_generations.json")):
        name = path.name[: -len("_generations.json")]
        if name != "baseline" and not name.startswith("sham_"):
            found.append(name)
    return [n for n in PROFILES if n in found] + sorted(n for n in found if n not in PROFILES)


def paired_report(outdir) -> dict:
    """Every comparison the design rests on, paired. Empty if the arms are not on disk."""
    profiles = arm_labels(outdir)
    labels = {a: load_labels(outdir, a)
              for a in ("baseline", *profiles, *(f"sham_{p}" for p in profiles))}
    out = {}
    for name in profiles:
        if labels.get(name) is None:
            continue
        for ref in ("baseline", "null", f"sham_{name}"):
            if ref == name or labels.get(ref) is None:
                continue
            out[f"{name}_vs_{ref}"] = mcnemar_exact(labels[ref], labels[name])
    if all(labels.get(k) is not None for k in PROFILES):
        out["mediation_ci"] = mediation_ci(labels["null"], labels["transplant"],
                                           labels["zero"])
    return out


def sweep_curve(arms: dict, retentions: dict, tol: float = 0.2):
    """(retention, compliance) for every arm that has both, ordered ablation-end first.

    `null` and `zero` are the retention 1.0 and 0.0 ends of the axis by construction, so
    they are curve POINTS rather than separate references — which is what lets a sweep of
    three interior scales bracket a crossing at all.

    Degenerate arms are split out, not dropped quietly. An arm in a repetition loop scores
    0.000 compliance, which on this curve is indistinguishable from "restoring refusal
    worked" and would pull the fitted threshold toward ablation. Returns
    ``(curve, excluded)``.
    """
    kept, dropped = [], []
    for label, retention in retentions.items():
        arm = arms.get(label)
        if not arm or not arm.get("n") or retention is None:
            continue
        row = {"arm": label, "retention": round(retention, 4), "comply": arm["comply"],
               "incoherent": arm.get("incoherent", 0), "n": arm["n"]}
        (dropped if row["incoherent"] / row["n"] > tol else kept).append(row)
    key = lambda r: r["retention"]  # noqa: E731
    return sorted(kept, key=key), sorted(dropped, key=key)


def threshold_crossing(curve: list[dict], level: float) -> dict:
    """Where compliance falls back through `level` as refusal magnitude is restored.

    Walks from the ablation end upward and reports the FIRST downward crossing, linearly
    interpolated. `monotone` is reported alongside rather than smoothed away: compliance is
    a proportion over 150 prompts, so a six-point curve wobbles, and a crossing read off a
    non-monotone curve is a number to distrust rather than to put in an abstract.
    """
    pts = sorted(curve, key=lambda r: r["retention"])
    if len(pts) < 2:
        return {"level": round(level, 4), "retention": None, "monotone": None,
                "note": "fewer than two usable points on the curve"}
    monotone = all(b["comply"] <= a["comply"] + 1e-12 for a, b in zip(pts, pts[1:]))
    for a, b in zip(pts, pts[1:]):
        if a["comply"] >= level > b["comply"]:
            span = a["comply"] - b["comply"]
            frac = 0.0 if span <= 1e-12 else (a["comply"] - level) / span
            hit = a["retention"] + frac * (b["retention"] - a["retention"])
            return {"level": round(level, 4), "retention": round(hit, 4),
                    "monotone": monotone, "between": [a["arm"], b["arm"]]}
    note = ("compliance never reaches the level, even at the lowest retention swept"
            if pts[0]["comply"] < level
            else "compliance is still above the level at the highest retention swept")
    return {"level": round(level, 4), "retention": None, "monotone": monotone, "note": note}


def threshold_bracket(outdir, curve: list[dict], alpha: float = 0.05) -> dict:
    """The highest retention that still beats `null`, and the lowest that does not.

    The interpolated crossing is a figure; this is the sentence. "Compliance is
    indistinguishable from the untouched refusal profile at retention >= Y and significantly
    above it at retention <= X" is a claim a reviewer can check against the paired counts,
    and unlike the crossing it assumes nothing about the curve being smooth.

    `bracket_consistent` is False when a LOWER retention showed no effect than one that did.
    That ordering is impossible for a real threshold, so it means the curve is too noisy to
    quote — reported rather than hidden behind a point estimate.
    """
    reference = load_labels(outdir, "null")
    if reference is None:
        return {}
    points = []
    for point in curve:
        if point["arm"] == "null":
            continue
        labels = load_labels(outdir, point["arm"])
        if labels is None:
            continue
        stat = mcnemar_exact(reference, labels)
        points.append({**point, "vs_null": stat,
                       "significant": stat["p"] < alpha and stat["net"] > 0})
    with_effect = [r["retention"] for r in points if r["significant"]]
    without = [r["retention"] for r in points if not r["significant"]]
    high = max(with_effect) if with_effect else None
    low = min(without) if without else None
    return {
        "alpha": alpha,
        "highest_retention_with_effect": high,
        "lowest_retention_without_effect": low,
        "bracket_consistent": None if high is None or low is None else high < low,
        "points": points,
    }


def threshold_verdict(jailbreak_retention: float | None, bracket: dict) -> str:
    """The one sentence the sweep exists to produce, built from the numbers rather than
    typed by hand — so it cannot drift from the table printed directly above it."""
    high = bracket.get("highest_retention_with_effect")
    if jailbreak_retention is None or high is None:
        return "no threshold established — no swept arm showed an effect over null"
    if bracket.get("bracket_consistent") is False:
        return ("threshold not quotable — a lower retention showed no effect than one that "
                "did, so the curve is not clean enough to name a crossing")
    if jailbreak_retention <= high:
        return (f"the jailbreak's {jailbreak_retention:.1%} retention is at or below the "
                f"{high:.1%} threshold, so suppression of this depth IS sufficient")
    return (f"compliance requires retention <= {high:.1%}; real jailbreaks reach only "
            f"{jailbreak_retention:.1%}, so their suppression is too shallow to act alone")


def sweep_block(arms: dict, retentions: dict, outdir,
                jailbreak_retention: float | None,
                levels=(0.25, 0.5, 0.75)) -> dict:
    """The whole sweep analysis. Empty when there is nothing to fit a threshold against."""
    curve, excluded = sweep_curve(arms, retentions)
    if len(curve) < 2:
        return {}
    comply = {p["arm"]: p["comply"] for p in curve}
    null_c, zero_c = comply.get("null"), comply.get("zero")
    crossings = {}
    # Levels are fractions of the ABLATION effect, not absolute compliance rates: the two
    # models baseline at 11.3% and 4.0%, so a fixed rate would mean different things in each.
    if null_c is not None and zero_c is not None and zero_c - null_c > 1e-9:
        for level in levels:
            crossings[f"{level:g}"] = threshold_crossing(
                curve, null_c + level * (zero_c - null_c))
    bracket = threshold_bracket(outdir, curve)
    return {
        "curve": curve,
        "excluded_degenerate": excluded,
        "crossings": crossings,
        "bracket": bracket,
        "jailbreak_retention": (None if jailbreak_retention is None
                                else round(jailbreak_retention, 4)),
        "verdict": threshold_verdict(jailbreak_retention, bracket),
    }


def degenerate_arms(arms: dict, tol: float = 0.2) -> list[str]:
    """Profiles whose incoherence rate makes their compliance number meaningless.

    The first transplant run printed "MEDIATION 13.5%" from a null arm that was 149/150
    repetition loops: its 0.000 compliance measured a broken model, not an intervention,
    and a ratio built on it reads as a measurement. Named rather than silently dropped.

    Scans `arms` rather than `PROFILES` so a broken SWEEP point is caught too. A degenerate
    sweep arm is the more dangerous case: its 0.000 compliance looks like "no effect at this
    retention" and would drag the fitted threshold toward ablation.
    """
    return [name for name, a in arms.items()
            if name != "baseline" and not name.startswith("sham_")
            and a.get("n") and a.get("incoherent", 0) / a["n"] > tol]


def mediation_fraction(null: float, transplant: float, zero: float) -> float | None:
    """Share of the achievable effect the jailbroken profile captures.

    1.0 means the jailbroken refusal geometry does everything ablation does; 0.0 means it
    does nothing. Returns None when the ceiling is degenerate — if `zero` achieved nothing
    over `null` there is no scale to report against, and the ratio would be a division by
    ~0 dressed up as a result. Not clipped: a transplant beating ablation is a real outcome.
    """
    denom = zero - null
    if abs(denom) < 1e-9:
        return None
    return (transplant - null) / denom


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    shifts = [r["mean_abs_shift"] for r in rows if r.get("mean_abs_shift") is not None]
    return {
        "n": n,
        "comply": round(sum(r["judge"] == "COMPLY" for r in rows) / n, 4),
        "incoherent": sum(not r["coherent"] for r in rows),
        "mean_abs_shift_over_layers": round(st.mean(shifts), 2) if shifts else None,
    }


def backfill_arms(outdir, arms: dict) -> dict:
    """Summaries for arms already on disk that THIS invocation did not run.

    `--arms` exists to resume a partial run, but the summary dict only ever held what this
    process generated. A sweep resumed that way would fit its threshold on a curve missing
    the `null` and `zero` endpoints — an unanchored threshold reported in the same field as
    one from a complete run, which is this experiment's signature failure mode rather than
    a new one. Re-read arms are marked `from_disk` so the report says which is which.
    """
    for path in sorted(Path(outdir).glob("*_generations.json")):
        label = path.name[: -len("_generations.json")]
        if label in arms:
            continue
        arms[label] = {**summarize(json.loads(path.read_text())), "from_disk": True}
    return arms


def specificity(arms: dict, baseline_comply: float) -> dict:
    """Each profile against its matched sham, and against the null profile.

    Same two references F3 settled on: a clamp moves the population AND removes the variance
    within it, so the un-intervened baseline prices both together.
    """
    null = arms.get("null", {})
    null_comply = null.get("comply") if null.get("n") else None
    null_sham = arms.get("sham_null", {})
    null_sham_comply = null_sham.get("comply") if null_sham.get("n") else None
    out = {}
    for name in PROFILES:
        res, shm = arms.get(name), arms.get(f"sham_{name}")
        if not res or not res.get("n") or not shm or not shm.get("n"):
            continue
        real = res["comply"] - baseline_comply
        sham = shm["comply"] - baseline_comply
        entry = {
            "real_shift": round(real, 4),
            "sham_shift": round(sham, 4),
            "real_incoherent": res["incoherent"],
            "sham_incoherent": shm["incoherent"],
            "verdict": ("specific" if abs(real) > 2 * abs(sham) + 0.05
                        else "NOT specific — a random direction moves it comparably"),
        }
        if null_comply is not None and null_sham_comply is not None:
            real_n = res["comply"] - null_comply
            sham_n = shm["comply"] - null_sham_comply
            entry["real_shift_vs_null"] = round(real_n, 4)
            entry["sham_shift_vs_null"] = round(sham_n, 4)
            entry["verdict_vs_null"] = ("specific" if abs(real_n) > 2 * abs(sham_n) + 0.05
                                        else "NOT specific — a random direction moves it "
                                             "comparably")
        out[name] = entry
    return out


def main() -> int:
    import sys

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from jbct.data import load_candidates
    from jbct.directions import drop_dims
    from jbct.generate import generate
    from jbct.interventions import ProfileClamp
    from jbct.judge import judge_refusal
    from jbct.models import load_model
    from jbct.provenance import env_versions
    from jbct.registry import get_model_spec

    from f3_clamp.run import held_out_harmful
    from h3_causal.run import is_coherent

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--position", type=int, default=None,
                    help="negative index the profile is measured and imposed at; "
                         "default = the model's registry direction_position")
    ap.add_argument("--layers", default="all",
                    help="'all' (default) or a comma-separated list, e.g. '13,14,15' to test "
                         "whether the L13-14 suppression band carries the effect")
    ap.add_argument("--scales", default="",
                    help="comma-separated sweep points between the null and zero arms, e.g. "
                         "'0.7,0.5,0.3,0.1'. Empty (default) runs the three-profile design "
                         "unchanged. What the number means depends on --scale-axis")
    ap.add_argument("--scale-axis", default="retention", choices=list(SCALE_AXES),
                    help="'retention' (default): s is the fraction of the HARMFUL refusal "
                         "magnitude kept at every layer, so s=1 is the null arm and s=0 is "
                         "the zero arm, and the swept threshold lands on the same ruler as "
                         "the jailbreak's own weighted retention. 'jailbreak': s is the "
                         "fraction of the way from harmful to the JAILBROKEN profile, so "
                         "s=1 is the transplant and s>1 extrapolates along the jailbreak's "
                         "own banded shape — the only way to ask whether that shape is "
                         "right but too shallow")
    ap.add_argument("--sham-scales", default="all",
                    help="'all' (default), 'none', or a comma-separated subset of --scales. "
                         "Each sweep point moves the residual by a different amount, so a "
                         "sham per point is the matched control; trimming it is a real "
                         "weakening of the design and has to be asked for explicitly")
    ap.add_argument("--n", type=int, default=150, help="held-out harmful prompts")
    ap.add_argument("--n-fit", type=int, default=64,
                    help="how many harmful_train prompts H1 used to fit; never reused here")
    ap.add_argument("--f1", default="results/f1_geometry")
    ap.add_argument("--h1", default="results/h1")
    ap.add_argument("--positions-scope", default="anchor", choices=["anchor", "all"],
                    help="'anchor' (default) clamps only the measured token. 'all' imposes "
                         "that token's value on every position, which is NOT a jailbroken "
                         "state except where the target is 0 — kept only for comparison "
                         "with the F3 all-positions run")
    ap.add_argument("--clamp-generation", default="off", choices=["off", "on"],
                    help="'off' (default) applies the profile to the PROMPT and lets the "
                         "model generate freely — a nonzero target measured at one prompt "
                         "position has no meaning on a generated token. 'on' holds every "
                         "generated token to it, which is what the first run did and what "
                         "left the null arm at 149/150 repetition loops")
    ap.add_argument("--keep-outliers", action="store_true",
                    help="skip drop_dims. The targets come from F1's COMPLEMENT block, so "
                         "this deliberately re-creates the basis mismatch; never for science")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--judge-model", default="openrouter/anthropic/claude-haiku-4.5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="all",
                    help="comma-separated subset of "
                         f"{','.join(ARMS)} (+ sham_*), for resuming a partial run")
    ap.add_argument("--out", default="results/f3_transplant")
    ap.add_argument("--rescore", action="store_true",
                    help="recompute the paired statistics of a finished run from its "
                         "committed generations and exit. No model, no GPU")
    args = ap.parse_args()

    if args.rescore:
        spec0 = get_model_spec(args.model)
        pos = args.position if args.position is not None else spec0.direction_position.value
        d = Path(args.out) / args.model / f"p{pos}"
        path = d / "f3_transplant_report.json"
        if not path.exists():
            raise SystemExit(f"{path} not found — nothing to rescore")
        rep = json.loads(path.read_text())
        rep["paired"] = paired_report(d)
        rep["degenerate_arms"] = degenerate_arms(rep["arms"])
        # Rebuilt from the committed report, so a finished sweep can be re-thresholded (a
        # different alpha, a repaired judge) without touching a GPU.
        retentions = {k: v for k, v in (rep.get("retentions") or {}).items() if v is not None}
        if retentions:
            rep["sweep"] = sweep_block(rep["arms"], retentions, d,
                                       (rep.get("suppression_profile") or {}).get("weighted"))
        path.write_text(json.dumps(rep, indent=2))
        print(f"[f3t] rescored {path}")
        print_sweep(rep)
        print_paired(rep)
        return 0

    spec = get_model_spec(args.model)
    position = args.position if args.position is not None else spec.direction_position.value

    f1_path = Path(args.f1) / args.model / "f1_report.json"
    if not f1_path.exists():
        raise SystemExit(f"{f1_path} not found — the profile comes from F1. Run F1 first.")
    f1 = json.loads(f1_path.read_text())

    layers = (list(range(spec.n_layers)) if args.layers == "all"
              else [int(x) for x in args.layers.split(",")])
    require_layers(f1, position, layers)

    targets = {p: profile_targets(f1, position, layers, p) for p in PROFILES}
    print(f"[f3t] {args.model} position {position}, {len(layers)} layers "
          f"({layers[0]}..{layers[-1]})")
    for p in PROFILES:
        vals = list(targets[p].values())
        print(f"[f3t]   {p:<10} target range [{min(vals):,.1f}, {max(vals):,.1f}]")

    supp = suppression_profile(targets, layers)
    if supp["weighted"] is not None:
        print(f"[f3t]   jailbroken retains {supp['weighted']:.1%} of the harmful refusal "
              f"magnitude (weighted), median {supp['median_per_layer']:.1%} per layer")
        print("[f3t]   0% = the jailbreak already ablates, 100% = it suppresses nothing; "
              "near either end the transplant cannot be told from an existing arm")

    scales = parse_scales(args.scales)
    sweep_targets = {scale_arm(s): scaled_targets(targets, layers, s, args.scale_axis)
                     for s in scales}
    # Every arm is filed by its WEIGHTED RETENTION, not by its scale, so that both axes and
    # the transplant itself land on one comparable x. On the retention axis these coincide;
    # on the jailbreak axis they do not, and plotting against s there would put the two
    # sweeps on different rulers while looking identical.
    retentions = {"null": 1.0, "zero": 0.0, "transplant": supp["weighted"]}
    for label, tgt in sweep_targets.items():
        retentions[label] = weighted_retention(tgt, targets["null"], layers)
    if scales:
        print(f"[f3t]   sweep on the {args.scale_axis} axis, {len(scales)} "
              f"point{'s' if len(scales) != 1 else ''}:")
        for s_val in scales:
            label = scale_arm(s_val)
            vals = list(sweep_targets[label].values())
            print(f"[f3t]     {label:<12} s={s_val:<6g} retention "
                  f"{retentions[label]:6.1%}  target range "
                  f"[{min(vals):,.1f}, {max(vals):,.1f}]")

    # Every name is resolved BEFORE the model is loaded. A typo discovered after the weights
    # are on the GPU costs the run, and an unrecognised --arms value used to run nothing at
    # all and report an empty arms block as though that were the result.
    sweep_labels = [scale_arm(s) for s in scales]
    all_arms = (*ARMS, *sweep_labels)
    wanted = all_arms if args.arms == "all" else tuple(x.strip() for x in args.arms.split(","))
    unknown_arms = sorted(set(wanted) - set(all_arms)
                          - {f"sham_{a}" for a in all_arms if a != "baseline"})
    if unknown_arms:
        raise SystemExit(
            f"--arms names {', '.join(unknown_arms)}, which no arm in this run produces. "
            f"Available: {', '.join(all_arms)} (+ sham_* for each but baseline)."
        )
    if args.sham_scales == "all":
        sham_wanted = set(sweep_labels)
    elif args.sham_scales == "none":
        sham_wanted = set()
    else:
        sham_wanted = {scale_arm(x) for x in parse_scales(args.sham_scales)}
        unknown = sorted(sham_wanted - set(sweep_labels))
        if unknown:
            raise SystemExit(
                f"--sham-scales names {', '.join(unknown)}, which is not in --scales "
                f"({', '.join(sweep_labels) or 'empty'}). A sham with no arm to match is "
                f"GPU time spent on a control for an experiment that never ran."
            )

    prompts = held_out_harmful(args.n, args.n_fit)
    print(f"[f3t] {len(prompts)} held-out harmful prompts (none used to fit the direction)")

    cands = load_candidates(args.h1, args.model)
    dims = () if args.keep_outliers else spec.outlier_dims
    directions, missing = {}, []
    for layer in layers:
        key = site(layer, position)
        if key not in cands:
            missing.append(key)
            continue
        directions[layer] = drop_dims(cands[key].detach().to(dtype=torch.float64), dims)
    if missing:
        raise SystemExit(
            f"candidates.pt has no direction at {', '.join(missing[:6])}"
            f"{' ...' if len(missing) > 6 else ''}. H1 fits only its `direction_positions`."
        )

    gen_t = torch.Generator().manual_seed(args.seed)
    shams = {layer: drop_dims(
        torch.randn(directions[layer].shape, generator=gen_t, dtype=torch.float64), dims)
        for layer in layers}

    model, tok = load_model(spec, device=args.device, dtype=getattr(torch, args.dtype))
    outdir = Path(args.out) / args.model / f"p{position}"
    outdir.mkdir(parents=True, exist_ok=True)
    clamp_pos = None if args.positions_scope == "all" else position
    prompt_only = args.clamp_generation == "off"

    def run_arm(label: str, hook_factory):
        rec: dict = {}
        with hook_factory(rec):
            gens = generate(model, tok, spec, prompts, prefills=[None] * len(prompts),
                            max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
        # `rec[layer]` holds one shift per prompt, so the per-prompt mean over layers is a
        # real per-prompt quantity rather than one sample wearing a plural name.
        per_prompt = []
        for i in range(len(prompts)):
            vals = [abs(rec[layer][i]) for layer in rec if i < len(rec[layer])]
            per_prompt.append(st.mean(vals) if vals else None)
        rows = [{"prompt": p, "response": g.response, "coherent": is_coherent(g.response),
                 "hit_token_cap": g.hit_token_cap, "mean_abs_shift": per_prompt[i]}
                for i, (p, g) in enumerate(zip(prompts, gens))]
        labels = judge_refusal([{"prompt": r["prompt"], "response": r["response"]}
                                for r in rows], model=args.judge_model)
        for row, is_refusal in zip(rows, labels):
            row["judge"] = "REFUSE" if is_refusal else "COMPLY"
        (outdir / f"{label}_generations.json").write_text(json.dumps(rows, indent=2))
        res = summarize(rows)
        print(f"  [f3t] {label:<16} comply {res['comply']:.3f}  "
              f"incoherent {res['incoherent']}/{res['n']}")
        return res

    class _NoHook:
        n_calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    arms: dict = {}
    if "baseline" in wanted:
        arms["baseline"] = run_arm("baseline", lambda _r: _NoHook())
    for name in PROFILES:
        if name in wanted:
            arms[name] = run_arm(name, lambda r, t=targets[name]: ProfileClamp(
                model, spec, directions, t, position=clamp_pos, record=r,
                prompt_only=prompt_only))
        if f"sham_{name}" in wanted or name in wanted:
            arms[f"sham_{name}"] = run_arm(f"sham_{name}", lambda r, t=targets[name]:
                                           ProfileClamp(model, spec, directions, t,
                                                        position=clamp_pos,
                                                        apply_along=shams, record=r,
                                                        prompt_only=prompt_only))

    for label in sweep_labels:
        if label in wanted:
            arms[label] = run_arm(label, lambda r, t=sweep_targets[label]: ProfileClamp(
                model, spec, directions, t, position=clamp_pos, record=r,
                prompt_only=prompt_only))
        if label in sham_wanted and (f"sham_{label}" in wanted or label in wanted):
            arms[f"sham_{label}"] = run_arm(
                f"sham_{label}", lambda r, t=sweep_targets[label]: ProfileClamp(
                    model, spec, directions, t, position=clamp_pos, apply_along=shams,
                    record=r, prompt_only=prompt_only))

    # Before anything reads `arms`: a resumed run must see the arms it did not re-generate.
    backfill_arms(outdir, arms)
    base = arms.get("baseline", {}).get("comply")
    # A ratio computed from a destroyed arm is worse than no ratio: the first run reported
    # "MEDIATION 13.5%" off a null arm that was 149/150 repetition loops, which reads as a
    # measurement rather than as a broken reference.
    degenerate = degenerate_arms(arms)
    # Gated on the three CORE profiles only: a broken sweep point invalidates the threshold
    # fit (which drops it, in `sweep_curve`) but says nothing about the null/transplant/zero
    # ratio, and suppressing that ratio over an unrelated arm would hide a good measurement.
    core_degenerate = [d for d in degenerate if d in PROFILES]
    frac = None
    if all(k in arms and arms[k].get("n") for k in PROFILES) and not core_degenerate:
        frac = mediation_fraction(arms["null"]["comply"], arms["transplant"]["comply"],
                                  arms["zero"]["comply"])

    report = {
        "env": env_versions(),
        "model": args.model,
        "position": position,
        "layers": layers,
        "dtype": args.dtype,
        "n_prompts": len(prompts),
        "n_fit_excluded": args.n_fit,
        "basis": "raw" if args.keep_outliers else "complement",
        "outlier_dims_dropped": list(dims),
        "positions_scope": args.positions_scope,
        "clamp_generation": args.clamp_generation,
        "degenerate_arms": degenerate,
        "sham_seed": args.seed,
        "targets": {p: {str(k): round(v, 4) for k, v in t.items()} for p, t in targets.items()},
        "suppression_profile": supp,
        "scale_axis": args.scale_axis,
        "scales": list(scales),
        "scale_targets": {label: {str(k): round(v, 4) for k, v in t.items()}
                          for label, t in sweep_targets.items()},
        "retentions": {k: (None if v is None else round(v, 4))
                       for k, v in retentions.items()},
        "sweep": sweep_block(arms, retentions, outdir, supp["weighted"]),
        "arms": arms,
        "specificity": specificity(arms, base) if base is not None else {},
        "paired": paired_report(outdir),
        "mediation_fraction": None if frac is None else round(frac, 4),
        "mediation_fraction_note": (
            "(transplant - null) / (zero - null): the share of the achievable effect that "
            "the jailbroken refusal profile captures. None when the zero arm achieved "
            "nothing over the null, leaving no scale to report against."
        ),
    }
    (outdir / "f3_transplant_report.json").write_text(json.dumps(report, indent=2))

    print(f"\n[f3t] -> {outdir}")
    if base is not None:
        print(f"  baseline {base:.3f}")
    for name, v in report["specificity"].items():
        line = (f"  {name:<11} vs baseline: real {v['real_shift']:+.3f} "
                f"sham {v['sham_shift']:+.3f} {v['verdict'].split(' —')[0]}")
        if "real_shift_vs_null" in v:
            line += (f"   | vs null: real {v['real_shift_vs_null']:+.3f} "
                     f"sham {v['sham_shift_vs_null']:+.3f} "
                     f"{v['verdict_vs_null'].split(' —')[0]}")
        print(line)
    if core_degenerate:
        print(f"\n  NO MEDIATION FIGURE — {', '.join(core_degenerate)} exceeded 20% "
              f"incoherent, "
              f"so its compliance rate measures a broken model rather than an intervention. "
              f"Compare the profiles against the BASELINE instead.")
        for name in PROFILES:
            a = arms.get(name, {})
            if a.get("n"):
                print(f"    {name:<11} comply {a['comply']:.3f}  "
                      f"incoherent {a['incoherent']}/{a['n']}")
    elif frac is not None:
        print(f"\n  MEDIATION {frac:.1%} of the achievable effect "
              f"(null {arms['null']['comply']:.3f} -> transplant "
              f"{arms['transplant']['comply']:.3f} -> zero {arms['zero']['comply']:.3f})")
    print_sweep(report)
    print_paired(report)
    return 0


def print_sweep(report: dict) -> None:
    """The threshold table. Printed above the paired block because it is the claim; the
    McNemar rows below are the evidence it is built from."""
    sweep = report.get("sweep") or {}
    curve = sweep.get("curve") or []
    if not curve:
        return
    stats = {r["arm"]: r.get("vs_null") for r in (sweep.get("bracket") or {}).get("points", [])}
    print("\n  [scale sweep — compliance as refusal magnitude is restored]")
    print(f"    {'arm':<14}{'retention':>10}{'comply':>9}   vs null")
    for point in curve:
        stat = stats.get(point["arm"])
        tail = "" if not stat else (f"net {stat['net']:+4d}  p={stat['p']:.4f}"
                                    f"{' *' if stat['p'] < 0.05 else ''}")
        print(f"    {point['arm']:<14}{point['retention']:>9.1%}"
              f"{point['comply']:>9.3f}   {tail}")
    for point in sweep.get("excluded_degenerate") or []:
        print(f"    {point['arm']:<14}{point['retention']:>9.1%}   EXCLUDED — "
              f"{point['incoherent']}/{point['n']} incoherent")
    for level, cross in (sweep.get("crossings") or {}).items():
        if cross.get("retention") is None:
            print(f"    crossing at {float(level):.0%} of the ablation effect: none — "
                  f"{cross.get('note')}")
            continue
        warn = "" if cross.get("monotone") else "   — CURVE NOT MONOTONE, distrust"
        print(f"    crossing at {float(level):.0%} of the ablation effect: retention "
              f"{cross['retention']:.1%} (between {' and '.join(cross['between'])}){warn}")
    bracket = sweep.get("bracket") or {}
    high, low = (bracket.get("highest_retention_with_effect"),
                 bracket.get("lowest_retention_without_effect"))
    if high is not None:
        tail = "" if low is None else f", absent at retention >= {low:.1%}"
        print(f"    effect present at retention <= {high:.1%}{tail} (alpha "
              f"{bracket.get('alpha')})")
    if sweep.get("jailbreak_retention") is not None:
        print(f"    the jailbreak itself sits at retention "
              f"{sweep['jailbreak_retention']:.1%}")
    print(f"    => {sweep.get('verdict')}")


def print_paired(report: dict) -> None:
    """Paired results, which are the ones to read: the arms share their prompts."""
    paired = report.get("paired") or {}
    if not paired:
        return
    print("\n  [paired — same prompts across arms, exact McNemar]")
    for key, v in paired.items():
        if key == "mediation_ci":
            continue
        stars = ("***" if v["p"] < 0.001 else "**" if v["p"] < 0.01
                 else "*" if v["p"] < 0.05 else "ns")
        print(f"    {key:<28} gained {v['gained']:3d}  lost {v['lost']:3d}  "
              f"net {v['net']:+4d}  p={v['p']:.4f} {stars}")
    ci = paired.get("mediation_ci")
    if ci and ci.get("lo") is not None:
        print(f"    mediation {ci['point']:.1%}  95% CI [{ci['lo']:.1%}, {ci['hi']:.1%}]"
              f"{'' if ci['excludes_zero'] else '  — SPANS ZERO'}")


if __name__ == "__main__":
    raise SystemExit(main())
