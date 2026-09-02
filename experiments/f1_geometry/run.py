"""F1 — refusal-direction geometry across prompt populations.

Foundational natural-state experiment: does the refusal direction's magnitude separate
harmful, jailbroken and clean prompts, and do jailbroken prompts sit near harmful ones?

The pieces existed and were never joined on one axis. H1 computes per-prompt projections and
throws them away (`h1_direction/run.py:254-258` saves only the tensors and the summary). The
H1xH2 chain projects jailbreak prompts but `chain_h1_h2/run.py:91` filters to
`kind == "attack"`, so harmful and harmless prompts are never scored beside them. Nothing in
the project has ever been plotted.

**No prompt in a figure was used to fit the direction.** The direction comes from H1's TRAIN
split; the harmful and clean populations are the held-out VAL split. Fit on everything and
the harmful/clean gap is guaranteed by construction, which a reviewer sees immediately.

**Five populations, not three.** H1's harmful prompts are bare instructions while attacks are
long elaborate framings, so a harmful-vs-attack gap partly measures prompt length. H2's
control is the SAME harmful request under weak framing, format-matched to the attacks, which
is the baseline that isolates harmfulness from style. Attacks split by judge because that is
where the hypothesis lives: if the direction mediates jailbreaking, COMPLY sits below REFUSE.

Run (RunPod — fp32 Gemma-3-4B is ~16 GB and will not fit the 5080, see `models.py:143`):
    python experiments/f1_geometry/run.py --model gemma-3-4b --dtype float32
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Re-exported: these live in jbct.data now, because chain_h1_h2, h3_causal and
# h3_correlation all read the plain path directly and hit the same failure.
from jbct.data import (  # noqa: F401
    read_json_maybe_gz,
    read_judged,
    resolve_judged,
    write_json_gz,
)

# Ordered for the legend and for every colour lookup: low end of the direction to high.
POPULATIONS = ("clean", "harmful", "control", "attack-COMPLY", "attack-REFUSE")


def prompt_relative_index(neg: int, n_prompt_tokens: int) -> int:
    """Absolute index of a NEGATIVE position, resolved against the PROMPT, not the sequence.

    H1 fitted at negative indices on bare prompts. A prefill appends to the assistant turn,
    so reading the same negative index off the full sequence lands inside the prefill string
    — `positions.py` records Gemma's `model` token moving from -2 to -20 that way, and 50 of
    the 350 attacks carry a prefill. Resolving against the prompt length keeps the index
    inside the prompt region, where appending cannot move it, and reproduces H1 exactly on
    the prompts that have no prefill.
    """
    if neg >= 0:
        raise ValueError(f"expected a negative index (H1's convention), got {neg}")
    idx = n_prompt_tokens + neg
    if idx < 0:
        raise ValueError(f"position {neg} falls outside a {n_prompt_tokens}-token prompt")
    return idx


def build_populations(judged: list[dict], harmful_val: list[str], harmless_val: list[str]):
    """One record per prompt to project, tagged with its population.

    Attacks the judge could not label are dropped rather than guessed — the same rule
    `chain_h1_h2` and `h3_correlation` apply.
    """
    recs: list[dict] = []
    for text in harmless_val:
        recs.append({"population": "clean", "prompt_text": text, "prefill": None,
                     "class": None, "base_id": None, "judge": None})
    for text in harmful_val:
        recs.append({"population": "harmful", "prompt_text": text, "prefill": None,
                     "class": None, "base_id": None, "judge": None})
    for g in judged:
        kind, judge = g.get("kind"), g.get("judge")
        if kind == "control":
            pop = "control"
        elif kind == "attack" and judge in ("COMPLY", "REFUSE"):
            pop = f"attack-{judge}"
        else:
            continue
        recs.append({
            "population": pop,
            # The control is a DIFFERENT prompt from its attack (same request, weak framing).
            "prompt_text": g["prompt_text"],
            "prefill": g.get("prefill"),
            "class": g.get("class"),
            "base_id": g.get("base_id"),
            "judge": judge,
        })
    return recs


def require_chat_formatted(prompts: list[str], role_marker: str) -> None:
    """Refuse to fit a direction on unformatted instruction text.

    `compute_direction_candidates` does NOT apply the chat template — `_mean_resids_at_
    positions` tokenizes whatever it is handed, and `h1_direction/run.py:138-140` formats
    the corpus itself before calling it. Passing the raw corpus instead fits the direction
    on bare instructions, where position -2 is an arbitrary word rather than the role
    marker, and then projects it at properly templated positions: fit in one space,
    measured in another. It produces no error, only wrong numbers — F1's first Gemma run
    read AUC 0.3457 at 13@-2 where H1 reads 1.0.
    """
    bad = [p for p in prompts if role_marker not in p]
    if bad:
        raise ValueError(
            f"{len(bad)}/{len(prompts)} fit prompts lack the role marker {role_marker!r}, so "
            f"they are not chat-formatted. Apply format_prompt before fitting. "
            f"First: {bad[0][:70]!r}"
        )


def require_disjoint(fit: list[str], evaluated: list[str]) -> None:
    """Hard guard on the circularity the whole design exists to avoid."""
    overlap = set(fit) & set(evaluated)
    if overlap:
        raise ValueError(
            f"{len(overlap)} prompt(s) both fit the direction and appear in the figures, so "
            f"the harmful/clean separation would be circular. First: {sorted(overlap)[0]!r}"
        )


def summarize(values: list[float]) -> dict:
    """n / mean / sd / median. `sd` is None below n=2 rather than 0.0, which would read as
    'perfectly consistent' instead of 'undefined'."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "median": None}
    mean = sum(values) / n
    sd = None
    if n >= 2:
        sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    s = sorted(values)
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return {"n": n, "mean": mean, "sd": sd, "median": median}


def outlier_energy(v, dims) -> float:
    """Fraction of a direction's squared norm carried by the massive-activation dims.

    Scale-invariant, so it describes the DIRECTION rather than the separation. The registry
    records 0.81 for gemma at L15 (`registry.py:125`), measured once at one layer and then
    assumed everywhere; recording it per (layer, position) is what makes the raw/complement
    divergence a measured quantity instead of an inference. Where this is ~0 the two bases
    coincide; where it is high they are nearly orthogonal and a target read in one basis is
    meaningless in the other.
    """
    if not dims:
        return 0.0
    import torch

    n2 = float(torch.linalg.vector_norm(v)) ** 2
    if n2 <= 0:
        return 0.0
    return float(sum(float(v[..., d]) ** 2 for d in dims) / n2)


def site_key(layer: int, pos: int) -> str:
    return f"{layer}@{pos}"


def parse_site(token: str) -> tuple[int, int]:
    """'21@-3' -> (21, -3). The position is not optional.

    Naming only the layer is the mistake this signature exists to prevent: the behavioural
    signal sits at a different token position from the causal anchor in BOTH models
    (gemma 21@-3 vs 15@-2, qwen 33@-5 vs 18@-1), so a layer-only argument silently draws
    the right depth at the wrong token.
    """
    if "@" not in token:
        raise ValueError(f"expected 'layer@position', e.g. '21@-3', got {token!r}")
    layer, pos = token.split("@", 1)
    return int(layer), int(pos)


def best_site(per_site: dict, metric: str, positions=None) -> str | None:
    """The site furthest from chance on `metric`, optionally restricted to some positions.

    Furthest from 0.5 rather than largest: an AUC of 0.20 is as informative as 0.80, it just
    ranks the groups the other way, and silently preferring the high tail would hide an
    inversion.
    """
    def eligible(k):
        if per_site[k].get(metric) is None:
            return False
        return positions is None or int(k.split("@")[1]) in positions

    keys = [k for k in per_site if eligible(k)]
    return max(keys, key=lambda k: abs(per_site[k][metric] - 0.5), default=None)


def causal_anchor_of(report: dict) -> str | None:
    """The site the project's attribution and steering work reads at.

    Falls back to the registry when the report predates the field, so figures redrawn from
    already-committed artifacts still get the contrast panel.
    """
    if report.get("causal_anchor"):
        return report["causal_anchor"]
    try:
        from jbct.registry import get_model_spec

        spec = get_model_spec(report["model"])
        return f"{spec.causal_layer.value}@{spec.direction_position.value}"
    except Exception:  # noqa: BLE001 — a missing anchor drops a panel, never the figure
        return None


def render_figures(report: dict, rows: list[dict], outdir: Path, positions=None) -> list[str]:
    """Draw every figure from the report and the per-prompt projections alone.

    Split out from `main` so figures can be redrawn without a GPU: the first Gemma and Qwen
    runs plotted only `spec.direction_position`, which left the strongest site — Gemma's
    `<start_of_turn>` and Qwen's -5 — in no figure at all. Re-reading the committed
    artifacts costs seconds; re-running the model costs a pod.
    """
    from jbct.plots import figure_distributions, figure_layer_profile, figure_pca, figure_sep

    model = report["model"]
    per_site = report["per_site"]
    n_layers = max(int(k.split("@")[0]) for k in per_site) + 1
    if positions is None:
        positions = report["positions"]["sites"]
    written = []

    # One profile and one separation curve PER POSITION — they are different questions, and
    # for both models the position carrying the behavioural signal is not the canonical one.
    for pos in positions:
        if not any(k.endswith(f"@{pos}") for k in per_site):
            continue
        p1 = outdir / f"fig1_layer_profile_pos{pos}.png"
        figure_layer_profile(per_site, n_layers, pos, p1,
                             title=f"{model}: projection by layer at position {pos}")
        p4 = outdir / f"fig4_separation_pos{pos}.png"
        figure_sep(per_site, n_layers, pos, p4,
                   title=f"{model}: separation vs depth at position {pos}")
        written += [p1.name, p4.name]

    # Distributions at the two sites worth reading: where the project currently anchors, and
    # where the behavioural signal actually is.
    anchor = causal_anchor_of(report)
    behavioural = best_site(per_site, "auc_refuse_over_comply")
    panels = [k for k in (anchor, behavioural) if k and k in per_site]
    if panels:
        p2 = outdir / "fig2_distributions.png"
        figure_distributions(rows, panels, p2, title=f"{model}: projection distributions")
        written.append(p2.name)

    # PCA needs the 2-D coordinates, which a run stores per row. Older artifacts predate
    # that, so skip rather than silently drawing something else.
    coords = rows[0].get("pca") if rows else None
    if isinstance(coords, dict):
        for key in coords:
            ev = (report.get("pca_explained") or {}).get(key, [0.0, 0.0])
            p3 = outdir / f"fig3_pca_{key.replace('@', 'p')}.png"
            figure_pca([r["pca"][key] for r in rows], [r["population"] for r in rows],
                       ev, p3, title=f"{model}: residual PCA at L{key}")
            written.append(p3.name)
    elif isinstance(coords, list):
        # Pre-multi-site artifacts stored a bare [x, y] at the causal anchor.
        pmeta = report.get("pca", {})
        p3 = outdir / "fig3_pca.png"
        figure_pca([r["pca"] for r in rows], [r["population"] for r in rows],
                   report.get("pca_explained", [0.0, 0.0]), p3,
                   title=f"{model}: residual PCA at "
                         f"L{pmeta.get('layer')}@{pmeta.get('position')}")
        written.append(p3.name)
    else:
        print("  [plot] no per-row pca coords in projections.json — skipping fig3 "
              "(re-run the model to regenerate it)")
    return written


def canonicalize_components(v):
    """Pin each component's sign so the scatter does not mirror between runs.

    SVD sign is arbitrary. Without this, two figures of identical data look like different
    results — the kind of silent non-reproducibility this project keeps paying for.
    """
    out = v.clone()
    for j in range(out.shape[1]):
        col = out[:, j]
        if col[int(col.abs().argmax())] < 0:
            out[:, j] = -col
    return out


def pca_2d(x):
    """(coords, explained_variance_ratio[:2]) via SVD. Deterministic, no sklearn, no seed."""
    import torch

    xc = x - x.mean(dim=0, keepdim=True)
    _u, s, vh = torch.linalg.svd(xc, full_matrices=False)
    comps = canonicalize_components(vh.T[:, :2])
    total = float((s**2).sum())
    ev = [float(s[i] ** 2) / total for i in range(min(2, len(s)))] if total > 0 else [0.0, 0.0]
    return xc @ comps, ev


def main() -> int:
    import sys

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from h1_direction.run import auc

    from jbct.data import load_direction_corpus
    from jbct.directions import compute_direction_candidates, drop_dims
    from jbct.models import HookManager, format_prompt, load_model, tokenize_prompt
    from jbct.registry import get_model_spec

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--h2", default="results/h2")
    ap.add_argument("--judged", default=None,
                    help="judged.json or .json.gz; default resolves the gz archive")
    ap.add_argument("--limit-per-population", type=int, default=None,
                    help="cap each population — for a plumbing smoke run, never for science")
    ap.add_argument("--n", type=int, default=64, help="fit prompts per class (H1's default)")
    ap.add_argument("--n-val", type=int, default=32, help="held-out prompts per class")
    # float32 to match H1. bf16 carries ~8 mantissa bits against residual norms of ~30,000,
    # and `models.py:137` records that directions from the two dtypes must never be compared.
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pca-sites", default=None,
                    help="sites for the 2-D scatter as 'layer@pos', comma-separated, e.g. "
                         "'15@-2,21@-3'. Default = the registry causal anchor alone. The "
                         "POSITION matters as much as the layer: the behavioural signal sits "
                         "at a different position from the anchor in both models")
    ap.add_argument("--figure-positions", default=None,
                    help="comma-separated positions to draw, e.g. '-2,-3'; default = all")
    ap.add_argument("--replot", action="store_true",
                    help="redraw figures from the committed report + projections. No model, "
                         "no GPU, seconds rather than a pod")
    ap.add_argument("--out", default="results/f1_geometry")
    args = ap.parse_args()

    figure_positions = ([int(p) for p in args.figure_positions.split(",")]
                        if args.figure_positions else None)

    if args.replot:
        outdir = Path(args.out) / args.model
        report = json.loads((outdir / "f1_report.json").read_text())
        rows = read_json_maybe_gz(outdir / "projections.json")
        names = render_figures(report, rows, outdir, positions=figure_positions)
        print(f"\n[f1] redrew {len(names)} figures in {outdir}")
        report_headline(report, outdir)
        return 0

    spec = get_model_spec(args.model)
    repo_root = Path(__file__).resolve().parents[2]
    judged_path = resolve_judged(args.judged, Path(args.h2), args.model, repo_root)
    print(f"[f1] judged generations: {judged_path}")
    judged = read_judged(judged_path)

    f_fit, k_fit = load_direction_corpus("train", n=args.n)
    f_val, k_val = load_direction_corpus("val", n=args.n_val)
    require_disjoint(fit=f_fit + k_fit, evaluated=f_val + k_val)
    records = build_populations(judged, harmful_val=f_val, harmless_val=k_val)
    require_disjoint(fit=f_fit + k_fit, evaluated=[r["prompt_text"] for r in records])
    if args.limit_per_population:
        seen: dict[str, int] = {}
        capped = []
        for r in records:
            n = seen.get(r["population"], 0)
            if n < args.limit_per_population:
                seen[r["population"]] = n + 1
                capped.append(r)
        print(f"[f1] SMOKE RUN: capped at {args.limit_per_population}/population "
              f"({len(records)} -> {len(capped)}). Numbers are not publishable.")
        records = capped

    counts = {p: sum(r["population"] == p for r in records) for p in POPULATIONS}
    print(f"[f1] fit on {len(f_fit)} harmful + {len(k_fit)} harmless (train split)")
    print(f"[f1] projecting {len(records)} held-out prompts: {counts}")

    dtype = getattr(torch, args.dtype)
    model, tok = load_model(spec, device=args.device, dtype=dtype)

    # The Arditi grid, fit on TRAIN only. Keyed "{layer}@{negative position}".
    # Format FIRST — `compute_direction_candidates` will not do it for us, and H1 formats
    # before calling it (`h1_direction/run.py:138-140`). See `require_chat_formatted`.
    fmt_fit = [format_prompt(spec, tok, t) for t in f_fit]
    fml_fit = [format_prompt(spec, tok, t) for t in k_fit]
    require_chat_formatted(fmt_fit + fml_fit, spec.role_marker)
    print(f"[f1] fitting directions at positions {list(spec.direction_positions)}")
    grid = compute_direction_candidates(model, tok, spec, fmt_fit, fml_fit)
    # Align BOTH sides at the boundary rather than relying on promotion — the failure
    # `chain_h1_h2.project_scalar` exists to prevent. The grid is accumulated in float64 but
    # on the MODEL's device, so a cpu residual dotted against it raises from inside torch
    # naming the dot, not the direction.
    units = {f"{k[0]}@{k[1]}":
             (v / torch.linalg.vector_norm(v)).detach().to(device="cpu", dtype=torch.float64)
             for k, v in grid.items()}
    # BOTH bases, every site. The raw direction is what F1 has always reported; the
    # complement (massive-activation dims zeroed) is the one every CAUSAL stage acts along
    # — `directions.py:99` and `registry.py:127` both say causal work must use it. Recording
    # only the raw projection is what let F3 read a target of -29,983 measured along r_raw
    # and impose it on <x, r_comp>, two vectors 64 degrees apart at gemma L15. The extra cost
    # is one dot product per site; there is no additional forward pass.
    units_complement = {
        key: drop_dims(grid[(int(key.split("@")[0]), int(key.split("@")[1]))],
                       spec.outlier_dims).detach().to(device="cpu", dtype=torch.float64)
        for key in units}
    energy = {key: round(outlier_energy(v, spec.outlier_dims), 6) for key, v in units.items()}
    n_layers = spec.n_layers
    sites = sorted(spec.direction_positions, reverse=True)  # e.g. (-2, -3, -5)

    # Residuals are captured DURING the forward pass, so the PCA sites have to be named up
    # front — they cannot be chosen from the per-site AUCs computed afterwards. Capturing
    # every site instead is not an option: 764 prompts x 102 sites x d_model x 8 bytes is
    # over a terabyte.
    pca_sites = ([parse_site(s) for s in args.pca_sites.split(",")] if args.pca_sites
                 else [(spec.causal_layer.value, spec.direction_position.value)])
    resid_rows: dict[str, list] = {site_key(*s): [] for s in pca_sites}
    print(f"[f1] PCA sites: {', '.join(resid_rows)}")

    out_rows = []
    for i, rec in enumerate(records):
        prompt = format_prompt(spec, tok, rec["prompt_text"])
        n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
        full = prompt + rec["prefill"] if rec["prefill"] else prompt
        ids = tokenize_prompt(tok, full, device=model.device)
        with HookManager(model, spec) as hm:
            model(**ids)
        row = {k: rec[k] for k in ("population", "class", "base_id", "judge")}
        row["n_prompt_tokens"] = n_prompt
        row["has_prefill"] = rec["prefill"] is not None
        proj: dict[str, float] = {}
        proj_complement: dict[str, float] = {}
        for neg in sites:
            idx = prompt_relative_index(neg, n_prompt)
            for layer in range(n_layers):
                key = f"{layer}@{neg}"
                if key not in units:
                    continue
                v = hm.residuals[layer][0, idx].to(device="cpu", dtype=torch.float64)
                proj[key] = float(v @ units[key])
                proj_complement[key] = float(v @ units_complement[key])
        row["proj"] = proj
        row["proj_complement"] = proj_complement
        for layer, pos in pca_sites:
            resid_rows[site_key(layer, pos)].append(
                hm.residuals[layer][0, prompt_relative_index(pos, n_prompt)]
                .to(device="cpu", dtype=torch.float64)
            )
        out_rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"  [f1] {i + 1}/{len(records)}")

    # ---- report -----------------------------------------------------------------
    by_pop = {p: [r for r in out_rows if r["population"] == p] for p in POPULATIONS}

    def score_site(key: str, field: str) -> dict | None:
        if not any(key in r[field] for r in out_rows):
            return None
        vals = {p: [r[field][key] for r in rows if key in r[field]]
                for p, rows in by_pop.items()}
        entry = {p: summarize(v) for p, v in vals.items()}
        # Does the direction separate harmful from clean HERE? (held out)
        if vals["harmful"] and vals["clean"]:
            entry["auc_harmful_over_clean"] = round(auc(vals["harmful"], vals["clean"]), 4)
        # ...and does it carry information about BEHAVIOUR? AUC > 0.5 means refused
        # attacks project higher, which is what mediation predicts.
        if vals["attack-REFUSE"] and vals["attack-COMPLY"]:
            entry["auc_refuse_over_comply"] = round(
                auc(vals["attack-REFUSE"], vals["attack-COMPLY"]), 4)
        return entry

    per_site: dict[str, dict] = {}
    per_site_complement: dict[str, dict] = {}
    for neg in sites:
        for layer in range(n_layers):
            key = f"{layer}@{neg}"
            raw_entry = score_site(key, "proj")
            if raw_entry is None:
                continue
            per_site[key] = raw_entry
            per_site_complement[key] = score_site(key, "proj_complement")

    outdir = Path(args.out) / args.model
    outdir.mkdir(parents=True, exist_ok=True)
    report = {
        "model": args.model,
        "design": (
            "direction fit on H1 TRAIN split; every projected prompt is held out. Five "
            "populations: clean/harmful are H1 val, control is the same harmful request "
            "under weak framing (NOT harmless), attacks split by H2 judge."
        ),
        "dtype": args.dtype,
        "n_fit": {"harmful": len(f_fit), "harmless": len(k_fit)},
        "counts": counts,
        "positions": {
            "convention": "negative index resolved against the PROMPT length, prefill-safe",
            "sites": list(sites),
            "n_prefilled_prompts": sum(r["has_prefill"] for r in out_rows),
        },
        "pca_sites": list(resid_rows),
        "causal_layer_provenance": [spec.causal_layer.value, spec.causal_layer.provenance],
        # Where the project's attribution and steering work currently reads. Named in the
        # report so a figure can contrast it against wherever the signal actually is.
        "causal_anchor": f"{spec.causal_layer.value}@{spec.direction_position.value}",
        "bases": {
            "proj": "raw diff-in-means direction, r/||r|| — what F1 has always reported",
            "proj_complement": (
                f"outlier dims {list(spec.outlier_dims)} zeroed then renormalized — the "
                "basis every causal stage (H3/F2/F3, H4 attribution) acts along"
            ),
        },
        "outlier_dims": list(spec.outlier_dims),
        # Per-site share of the raw direction's squared norm sitting on the outlier dims.
        # ~0 means the two bases coincide; high means a target read in one is meaningless
        # in the other, which is exactly what broke F3.
        "outlier_energy": energy,
        "per_site": per_site,
        "per_site_complement": per_site_complement,
    }

    # PCA coordinates ride in the rows so the scatter can be redrawn without the model.
    report["pca_explained"] = {}
    for key, mats in resid_rows.items():
        coords, ev = pca_2d(torch.stack(mats))
        for row, xy in zip(out_rows, coords.tolist()):
            row.setdefault("pca", {})[key] = xy
        report["pca_explained"][key] = ev
    (outdir / "f1_report.json").write_text(json.dumps(report, indent=2))
    # Compressed: carrying both bases put this file over the R23 5 MB cap. `f1_report.json`
    # stays plain — it is the small one, and the one worth diffing on the forge.
    written = write_json_gz(outdir / "projections.json", out_rows)
    stale = outdir / "projections.json"
    if stale.exists():
        print(f"[f1] NOTE {stale} is a pre-gz leftover and is no longer read. "
              f"Remove it (`git rm --cached` if tracked); {written.name} replaces it.")

    render_figures(report, out_rows, outdir, positions=figure_positions)
    report_headline(report, outdir)
    return 0


def report_headline(report: dict, outdir: Path) -> None:
    per_site = report["per_site"]
    print(f"\n[f1] -> {outdir}")
    for label, key in (("anchor    ", causal_anchor_of(report)),
                       ("behaviour ", best_site(per_site, "auc_refuse_over_comply")),
                       ("separation", best_site(per_site, "auc_harmful_over_clean"))):
        if not key or key not in per_site:
            continue
        e = per_site[key]
        print(f"  {label} {key:<8} AUC(harmful>clean)={e.get('auc_harmful_over_clean')} "
              f"AUC(refuse>comply)={e.get('auc_refuse_over_comply')}")

    # Does the geometry survive dropping the massive-activation dims? Printed next to the
    # raw numbers because the causal stages only ever act in the complement basis, so a
    # result that lives entirely in the raw basis cannot be the thing they intervene on.
    comp = report.get("per_site_complement") or {}
    if not comp:
        return
    print("  [complement basis: outlier dims dropped]")
    for label, key in (("anchor    ", causal_anchor_of(report)),
                       ("behaviour ", best_site(per_site, "auc_refuse_over_comply")),
                       ("separation", best_site(per_site, "auc_harmful_over_clean"))):
        if not key or not comp.get(key):
            continue
        c = comp[key]
        share = (report.get("outlier_energy") or {}).get(key)
        share_s = f"  outlier_energy={share:.3f}" if isinstance(share, (int, float)) else ""
        print(f"  {label} {key:<8} AUC(harmful>clean)={c.get('auc_harmful_over_clean')} "
              f"AUC(refuse>comply)={c.get('auc_refuse_over_comply')}{share_s}")


if __name__ == "__main__":
    raise SystemExit(main())
