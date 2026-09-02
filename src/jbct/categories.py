"""Per-attack-class geometry: how far each jailbreak family moves the refusal direction.

F1 pools every successful attack into one `attack-COMPLY` population, and F3 transplants the
profile that pooling produces. Split by class, that profile turns out to describe no actual
attack: on gemma the classes run from 5% to 58% of a plain request's refusal magnitude, and
the pooled figure is the average of a class that nearly ablates the direction and classes
that barely touch it. This module computes the split.

Four things here are load-bearing.

**The reference is the 50 dataset bases, not the 32 prompts the direction was fitted on.**
Every attack is a wrapper around one of 50 base requests; the fit corpus is a different set
that sits ~18% lower on the direction, so measuring against it inflated every figure. The
bases are already committed: they are the `control` rows of the `prefill` class, whose
control IS the bare request. The class that cannot be measured is the one that supplies the
reference — and it doubles as the calibration check, because prefill cannot move the prompt,
so anything but ~100% means the reference is wrong. It read 117% against the fit corpus.

**Prefill is excluded from every prompt-side comparison.** Its attack is a forced assistant
opening applied during generation; `attack_text == control.text` token for token. The
control is correct and the lift is real (+64 on gemma), but nothing about the attack exists
in the prompt's residual stream. Reported as its own finding rather than averaged in.

**Two anchors, because they answer different questions.** `clean` places a prompt on the
harmless-to-harmful axis, and stays well-conditioned where the raw projection crosses zero,
because its denominator is a gap between two separated populations rather than a single
mean. `zero` divides by the base projection, so 0 is ablation and a class's position is
directly comparable to F3's `retention` sweep axis. Neither alone is enough: zero-anchored
per-layer ratios explode where the base projection is near zero (qwen L9 harmful is 0.03),
the artifact that once reported 175% suppression.

**Success rate differs by class, so `attack-COMPLY` is a different kind of sample in each.**
For fiction, 98% of attacks succeed and the COMPLY group is the whole class; for
authoritative_tone, 14% succeed and it is the unusual 7 that worked. That selection pushes
low-success classes to look MORE suppressed than they typically are, so a low-success class
reading as barely-suppressed (authoritative_tone, 58%) is evidence against suppression
rather than an artifact of it. `within_class_separation` is the version immune to this, and
it is computable only where success is intermediate.
"""

from __future__ import annotations

import hashlib
import math
import statistics as st

# The taxonomy the paper already uses, kept so figures can group by it. It is BEHAVIOURAL —
# how the attack is written — and the geometry does not respect it: fiction and roleplay are
# both "disguise" yet sit at 21% and 52% on gemma. Recorded to show that disagreement, not
# to assert it.
ATTACK_FAMILIES = {
    "prefill": "output-controlling",
    "refusal_suppression": "output-controlling",
    "refusal_suppression_long": "output-controlling",
    "fiction": "disguise",
    "roleplay": "disguise",
    "narrative_sandwich": "disguise",
    "authoritative_tone": "disguise",
}

# Classes whose attack is not in the prompt at all. See the module docstring.
POST_PROMPT_CLASSES = ("prefill",)

# Whose `control` population is the 50 bare requests. Same class, for the same reason.
REFERENCE_CLASS = "prefill"

COMPLY, REFUSE = "COMPLY", "REFUSE"


def site_key(layer: int, position: int) -> str:
    return f"{layer}@{position}"


def prompt_side_classes(classes) -> list[str]:
    """Classes a prompt-side comparison may include, in a stable order."""
    return sorted(c for c in classes if c and c not in POST_PROMPT_CLASSES)


def rows_where(rows, population=None, cls=None, judge=None) -> list[dict]:
    """Filter projection rows. `cls=None` means 'any class', not 'class is null'."""
    out = rows
    if population is not None:
        out = [r for r in out if r.get("population") == population]
    if cls is not None:
        out = [r for r in out if r.get("class") == cls]
    if judge is not None:
        out = [r for r in out if r.get("judge") == judge]
    return list(out)


def values_at(rows, layer: int, position: int, field: str = "proj_complement") -> list[float]:
    """Every prompt's projection at one (layer, position).

    `proj_complement` by default: the interventions drop the massive-activation coordinates,
    and reading the raw projection while clamping the complement is the basis mismatch that
    cost this project two rounds of GPU time.
    """
    key = site_key(layer, position)
    return [r[field][key] for r in rows if key in r.get(field, {})]


def _ci95(values) -> float:
    """Half-width of the 95% CI of the mean. 0.0 when there is nothing to spread."""
    n = len(values)
    if n < 2:
        return 0.0
    return 1.96 * st.stdev(values) / math.sqrt(n)


def _pooled_sd(a, b) -> float:
    na, nb = len(a), len(b)
    if na < 2 and nb < 2:
        return 0.0
    va = st.variance(a) if na > 1 else 0.0
    vb = st.variance(b) if nb > 1 else 0.0
    df = max(1, na + nb - 2)
    return math.sqrt(((na - 1) * va + (nb - 1) * vb) / df)


def reference_series(rows, position: int, n_layers: int, *,
                     field: str = "proj_complement", floor: float = 0.25) -> dict:
    """Per-layer `clean` and `base` means, with a usability flag per anchor.

    `base` is the 50 plain requests every attack is built on — the `control` rows of
    `REFERENCE_CLASS`. Using the direction's own fit corpus instead inflates every downstream
    ratio by ~18%, silently, because both are plausible-looking "harmful" populations.

    A cell is unusable for an anchor when that anchor's denominator is small against the
    spread of the data it came from: a ratio of noise to noise is a large number, not a
    measurement. Flagged per cell rather than dropped, so the figures can grey the cell and
    the report can say how many were suppressed.
    """
    clean_rows = rows_where(rows, population="clean")
    base_rows = rows_where(rows, population="control", cls=REFERENCE_CLASS)
    out = {"position": position, "n_clean": len(clean_rows), "n_base": len(base_rows),
           "floor": floor, "layers": {}}
    for layer in range(n_layers):
        cv = values_at(clean_rows, layer, position, field)
        bv = values_at(base_rows, layer, position, field)
        if not cv or not bv:
            continue
        clean_m, base_m = st.mean(cv), st.mean(bv)
        sd = _pooled_sd(cv, bv)
        gap = base_m - clean_m
        out["layers"][layer] = {
            "clean": clean_m,
            "base": base_m,
            "gap": gap,
            "pooled_sd": sd,
            # `sd <= 0` means every prompt agrees exactly; any nonzero gap is then real.
            "gap_usable": sd <= 0 or abs(gap) >= floor * sd,
            "base_usable": sd <= 0 or abs(base_m) >= floor * sd,
        }
    return out


def clean_anchored(attack_mean: float, ref: dict) -> float | None:
    """Where a prompt sits on the harmless-to-harmful axis.

    1.0 = exactly where a plain harmful request sits; 0.0 = where a harmless one does;
    below 0 = pushed past harmless to the far side. Robust where the raw projection crosses
    zero, because the denominator is a separation rather than a level.
    """
    if not ref.get("gap_usable"):
        return None
    return (attack_mean - ref["clean"]) / ref["gap"]


def zero_anchored(attack_mean: float, ref: dict) -> float | None:
    """Share of the base request's refusal magnitude still present.

    0.0 is ablation, which is what makes this the anchor comparable to F3's `retention`
    sweep axis. Unusable wherever the base projection is small against its own spread.
    """
    if not ref.get("base_usable"):
        return None
    return attack_mean / ref["base"]


def weighted_retention(targets: dict, reference: dict, layers) -> float | None:
    """Magnitude-weighted share of `reference`'s refusal magnitude that `targets` keeps.

    THE one definition, shared with `f3_transplant` (which imports it from here). The paper's
    central comparison places a swept threshold and each attack class's own suppression on
    one axis, so two copies free to drift would put them on different rulers without ever
    raising. Magnitude-weighted rather than a mean of per-layer ratios, because early layers
    carry near-zero projections and a ratio there is noise over noise — averaging those once
    reported 175% suppression for qwen.
    """
    den = sum(abs(reference[int(layer)]) for layer in layers)
    if den <= 1e-9:
        return None
    return sum(abs(targets[int(layer)]) for layer in layers) / den


def category_profile(rows, cls: str, position: int, n_layers: int, refs: dict, *,
                     judge: str = COMPLY, field: str = "proj_complement") -> dict:
    """Per-layer geometry for one attack class at one read position.

    `judge` selects the outcome to condition on. COMPLY is the default because "how far do
    attacks that WORKED move the direction" is the question; passing REFUSE gives the
    within-class contrast that `within_class_separation` scores.
    """
    sel = rows_where(rows, population=f"attack-{judge}", cls=cls)
    layers: dict[int, dict] = {}
    for layer in range(n_layers):
        ref = refs["layers"].get(layer)
        vals = values_at(sel, layer, position, field)
        if ref is None or not vals:
            continue
        mean = st.mean(vals)
        p_clean = clean_anchored(mean, ref)
        half = _ci95(vals)
        layers[layer] = {
            "n": len(vals),
            "mean": mean,
            "ci95": half,
            "p_clean": p_clean,
            # The CI treats the reference as fixed. It is estimated from 32 + 50 prompts and
            # is SHARED by every class, so it cannot move classes relative to one another —
            # which is what these figures compare. Stated rather than silently assumed.
            "p_clean_ci": None if p_clean is None else half / abs(ref["gap"]),
            "p_zero": zero_anchored(mean, ref),
            "shift_toward_clean": None if p_clean is None else 1.0 - p_clean,
            "gap_usable": ref["gap_usable"],
            "base_usable": ref["base_usable"],
        }
    return {
        "class": cls,
        "family": ATTACK_FAMILIES.get(cls),
        "position": position,
        "judge": judge,
        "post_prompt": cls in POST_PROMPT_CLASSES,
        "n_prompts": len(sel),
        "layers": layers,
    }


def profile_summary(profile: dict, refs: dict) -> dict:
    """One number per anchor for a whole profile, plus what had to be dropped.

    The clean-anchored aggregate is weighted by |gap|, so layers where harmless and harmful
    are barely separated cannot dominate — the same reasoning that makes the zero-anchored
    aggregate magnitude-weighted.
    """
    usable = [(layer, e) for layer, e in profile["layers"].items() if e["p_clean"] is not None]
    num = sum(abs(refs["layers"][layer]["gap"]) * e["p_clean"] for layer, e in usable)
    den = sum(abs(refs["layers"][layer]["gap"]) for layer, _ in usable)
    # `p_zero_weighted` takes absolute values, because F3's targets are absolute magnitudes
    # and the two must stay comparable. That makes it SIGN-BLIND: a class that inverts the
    # direction past harmless at some layers reads the same as one that keeps a small
    # positive projection there. gemma `fiction` does exactly this (-41% at L14), which is
    # why the two anchors order the classes differently. Counted here so the disagreement is
    # visible in the report rather than looking like an inconsistency.
    inverted = [layer for layer, e in profile["layers"].items()
                if e["p_clean"] is not None and e["p_clean"] < 0]
    zero_layers = [layer for layer, e in profile["layers"].items() if e["base_usable"]]
    attack = {layer: profile["layers"][layer]["mean"] for layer in zero_layers}
    base = {layer: refs["layers"][layer]["base"] for layer in zero_layers}
    p_clean = (num / den) if den > 1e-12 else None
    return {
        "class": profile["class"],
        "family": profile["family"],
        "position": profile["position"],
        "judge": profile["judge"],
        "post_prompt": profile["post_prompt"],
        "n_prompts": profile["n_prompts"],
        "p_clean_weighted": p_clean,
        "shift_toward_clean": None if p_clean is None else 1.0 - p_clean,
        "p_zero_weighted": weighted_retention(attack, base, zero_layers) if zero_layers
        else None,
        "n_layers_used": len(usable),
        "n_layers_dropped": len(profile["layers"]) - len(usable),
        "n_layers_inverted": len(inverted),
        "layers_inverted": sorted(inverted),
    }


def auc(pos_scores, neg_scores) -> float:
    """Mann-Whitney AUC: P(score(pos) > score(neg)), ties at half. 0.5 = chance.

    Same construction as `h1_direction.run.auc`, so per-class figures are on the same scale
    as F1's pooled `auc_refuse_over_comply`.
    """
    if not pos_scores or not neg_scores:
        return float("nan")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0
               for p in pos_scores for n in neg_scores)
    return wins / (len(pos_scores) * len(neg_scores))


def within_class_separation(rows, cls: str, layer: int, position: int, *,
                            field: str = "proj_complement", min_n: int = 5) -> dict | None:
    """Does the direction predict success WITHIN one attack class?

    Immune to the cross-class selection problem in the module docstring, because both groups
    are the same kind of attack. Returns None when either side is too thin — for a class that
    succeeds 98% of the time there is no failure group to contrast against, so the question
    is unanswerable rather than answered badly. Convention matches F1: above 0.5 means the
    direction ranks REFUSED attacks higher than successful ones.
    """
    refused = values_at(rows_where(rows, population=f"attack-{REFUSE}", cls=cls),
                        layer, position, field)
    complied = values_at(rows_where(rows, population=f"attack-{COMPLY}", cls=cls),
                         layer, position, field)
    if len(refused) < min_n or len(complied) < min_n:
        return None
    return {"class": cls, "layer": layer, "position": position,
            "n_refuse": len(refused), "n_comply": len(complied),
            "auc_refuse_over_comply": auc(refused, complied)}


# --------------------------------------------------------------------- judge reliability
#
# Every per-class number depends on which prompts landed in the COMPLY group, so a judge
# that is systematically wrong on ONE class corrupts that class specifically. The draft
# reports 50 hand-checked responses at full agreement, but not stratified by class, which is
# exactly the breakdown that would show a per-class failure.


def stratified_sample(items, key_fn, per_cell: int) -> list:
    """`per_cell` items from each cell, chosen deterministically and without RNG.

    Ordered within a cell by a hash of the item's own text rather than by position, so the
    sample is neither the first-N (which tracks dataset construction order and would draw
    every topic from the same block) nor seed-dependent. Re-running selects the same items,
    which matters because the human labels attach to them.
    """
    cells: dict = {}
    for item in items:
        cells.setdefault(key_fn(item), []).append(item)
    out = []
    for cell in sorted(cells, key=str):
        ranked = sorted(cells[cell],
                        key=lambda it: hashlib.sha1(
                            repr(it).encode("utf-8")).hexdigest())
        out.extend(ranked[:per_cell])
    return out


def confusion(pairs) -> dict:
    """Counts for (gold, predicted) label pairs, plus the two error rates separately.

    The directions are not interchangeable. A false COMPLY inflates an attack class's
    success rate AND puts a prompt that actually refused into the population whose refusal
    geometry the transplant copies; a false REFUSE only loses power. Reported apart so a
    reader can see which way the labelling errs.
    """
    tp = sum(1 for g, p in pairs if g == COMPLY and p == COMPLY)
    fp = sum(1 for g, p in pairs if g == REFUSE and p == COMPLY)
    fn = sum(1 for g, p in pairs if g == COMPLY and p == REFUSE)
    tn = sum(1 for g, p in pairs if g == REFUSE and p == REFUSE)
    n = tp + fp + fn + tn
    return {
        "n": n,
        "comply_comply": tp, "refuse_comply": fp,
        "comply_refuse": fn, "refuse_refuse": tn,
        "agreement": (tp + tn) / n if n else None,
        # named by what the judge did, not by "positive"/"negative", which flips meaning
        # depending on which label a reader treats as the interesting one
        "false_comply_rate": fp / (fp + tn) if (fp + tn) else None,
        "false_refuse_rate": fn / (fn + tp) if (fn + tp) else None,
        "kappa": cohens_kappa(pairs),
    }


def cohens_kappa(pairs) -> float | None:
    """Agreement above chance. Reported alongside raw agreement because agreement alone is
    flattering when one label dominates: a judge that always says REFUSE scores 96% on a
    class that succeeds 4% of the time."""
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(1 for g, p in pairs if g == p) / n
    gold_c = sum(1 for g, _ in pairs if g == COMPLY) / n
    pred_c = sum(1 for _, p in pairs if p == COMPLY) / n
    expected = gold_c * pred_c + (1 - gold_c) * (1 - pred_c)
    if abs(1 - expected) < 1e-12:
        return None
    return (observed - expected) / (1 - expected)


def agreement_by_cell(pairs_by_cell: dict) -> dict:
    """`confusion` per cell plus an `ALL` row, so a single bad class cannot hide in a mean."""
    out = {cell: confusion(pairs) for cell, pairs in sorted(pairs_by_cell.items(), key=str)}
    every = [p for pairs in pairs_by_cell.values() for p in pairs]
    out["ALL"] = confusion(every)
    return out


# ------------------------------------------------------------------------ label risk
#
# The committed judgments carry hashes, not response text, so agreement against a human
# cannot be computed from them. These are the reliability signals that survive slimming —
# and one of them found something: on gemma, `fiction`, `narrative_sandwich`, `prefill` and
# `roleplay` hit the 1024-token cap on 96-100% of generations while
# `refusal_suppression_long` hit it on none. Truncation is not spread evenly across the
# classes being compared.

TRUNCATED_REFUSE = "truncated_refuse"
SHORT_COMPLY = "short_comply"


def label_risk(judgments, *, short_tokens: int = 25) -> dict:
    """Per (class, kind) label-reliability signals, from slimmed judgments alone.

    Two pockets where a label is most likely wrong, counted separately because they push
    the headline numbers in opposite directions:

    `truncated_refuse` — the response hit the generation cap and was scored REFUSE. The
    model was still writing when it was cut off, so the judge scored an unfinished answer.
    On gemma this is ~48 of 50 `prefill` CONTROLS, and the control rate is what every lift
    in the paper is measured against, so an error here inflates lift.

    `short_comply` — a very short response scored COMPLY. Cheap to check and usually empty,
    but it is the pocket that would inflate an attack success rate rather than a control.
    """
    cells: dict = {}
    for row in judgments:
        cell = (row.get("class"), row.get("kind"))
        c = cells.setdefault(cell, {"n": 0, "comply": 0, "capped": 0,
                                    TRUNCATED_REFUSE: 0, SHORT_COMPLY: 0})
        complied = row.get("judge") == COMPLY
        capped = bool(row.get("hit_token_cap"))
        c["n"] += 1
        c["comply"] += complied
        c["capped"] += capped
        if capped and not complied:
            c[TRUNCATED_REFUSE] += 1
        if complied and (row.get("n_new_tokens") or 0) < short_tokens:
            c[SHORT_COMPLY] += 1
    out = {}
    for cell, c in sorted(cells.items(), key=str):
        n = c["n"]
        out[f"{cell[0]}/{cell[1]}"] = {
            **c,
            "comply_rate": c["comply"] / n if n else None,
            "cap_rate": c["capped"] / n if n else None,
            "at_risk_rate": (c[TRUNCATED_REFUSE] + c[SHORT_COMPLY]) / n if n else None,
        }
    return out


def hash_inconsistency(judgments) -> dict:
    """Identical response text carrying different labels — a provable judge contradiction.

    Currently vacuous (700 of 700 responses are distinct in both models), which is itself
    worth recording: it means the judge was never asked the same question twice, so nothing
    about its self-consistency can be inferred from the committed run. Kept as a guard,
    because a future run with a shared or degenerate response WOULD be caught here.
    """
    by_hash: dict = {}
    for row in judgments:
        key = row.get("response_sha256")
        if key is not None:
            by_hash.setdefault(key, set()).add(row.get("judge"))
    conflicting = sorted(h for h, labels in by_hash.items() if len(labels) > 1)
    return {
        "n_judgments": len(judgments),
        "n_distinct_responses": len(by_hash),
        "n_conflicting": len(conflicting),
        "conflicting": conflicting[:20],
        "informative": len(by_hash) < len(judgments),
    }


def behaviour_by_class(risk: dict) -> dict:
    """Attack rate, control rate and lift per class, from the same `label_risk` cells.

    Joined here so one table carries both halves of the claim. The interesting rows are the
    ones where behaviour and geometry disagree: on gemma `fiction` and `roleplay` lift +90 and
    +60 — comparable, both `disguise` — while leaving 21% and 52% of the refusal magnitude in
    place. A class that succeeds without suppressing is the whole finding, and it is only
    visible when the two are read side by side.
    """
    out: dict = {}
    for cell, m in risk.items():
        cls, _, kind = cell.rpartition("/")
        entry = out.setdefault(cls, {})
        entry[kind] = m["comply_rate"]
        entry[f"n_{kind}"] = m["n"]
    for cls, entry in out.items():
        attack, control = entry.get("attack"), entry.get("control")
        entry["lift"] = (None if attack is None or control is None
                         else attack - control)
        entry["family"] = ATTACK_FAMILIES.get(cls)
    return out


def population_agreement(pairs_by_cell: dict, cell_sizes: dict) -> dict:
    """Agreement re-weighted to the population the sample was drawn from.

    **The raw sample agreement is a biased estimate of the run's agreement, and reporting it
    as though it were not is a real error.** The labelling task is STRATIFIED by
    (class, kind, judge label) so that rare cells — an attack the judge scored REFUSE inside a
    class that almost always succeeds — are represented as deeply as common ones. That is the
    right design for FINDING where the judge fails, and exactly the wrong design for
    estimating how often it fails: a cell holding 1 of 700 responses contributes as much to
    the raw mean as a cell holding 49.

    Within a cell the sampled items are an unbiased subset, so cell-level agreement is sound.
    This re-weights those by each cell's true share. `coverage` is the share of the population
    that sampled cells account for — anything below 1.0 means some cells were never labelled
    and the estimate speaks only for the rest.
    """
    total = sum(cell_sizes.values())
    if not total:
        return {"available": False, "note": "no population sizes given"}
    seen, weighted = 0, 0.0
    per_cell = {}
    for cell, pairs in pairs_by_cell.items():
        size = cell_sizes.get(cell)
        if not size or not pairs:
            continue
        agree = sum(1 for g, p in pairs if g == p) / len(pairs)
        seen += size
        weighted += agree * size
        per_cell[str(cell)] = {"n_sampled": len(pairs), "n_population": size,
                               "agreement": round(agree, 4),
                               "weight": round(size / total, 4)}
    if not seen:
        return {"available": False, "note": "no sampled cell matched a population cell"}
    return {
        "available": True,
        "agreement": round(weighted / seen, 4),
        "coverage": round(seen / total, 4),
        "n_population": total,
        "n_cells": len(per_cell),
        "per_cell": per_cell,
        "note": ("Sample agreement is NOT this number: the labelling task is stratified, so "
                 "rare cells are over-represented in it by design."),
    }
