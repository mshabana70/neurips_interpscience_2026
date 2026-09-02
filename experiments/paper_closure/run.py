"""One restartable closure run for the workshop-paper causal gaps.

The run deliberately answers a small set of paper-changing questions:

1. Scope: untreated, anchor-token-only, prompt-wide, sustained, and off-layer
   directional removal on the same matched harmful controls.  Every intervention has an
   orthogonal matched-shift sham.
2. Per-class sufficiency: impose each successful jailbreak class's full prompt-side
   refusal profile on held-out harmful prompts.
3. Exact restoration: impose the corresponding refusing-control profile on successful
   attacks of that class.

Prefill is excluded from the profile stages.  Its intervention and measured state occur
after an assistant prefix, so its negative token positions are not comparable to the
prompt-side class profiles.

All generation artifacts are gzip-compressed, every arm is resumable, and paired
statistics are reconstructed from per-prompt rows rather than from aggregate rates.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path


MODEL_SITES = {
    "gemma-3-4b": {"causal": (15, -2), "off": (21, -3)},
    "qwen3-4b": {"causal": (18, -1), "off": (33, -5)},
}

SCOPE_ARMS = (
    "baseline",
    "anchor_real",
    "anchor_sham",
    "prompt_real",
    "prompt_sham",
    "sustained_real",
    "sustained_sham",
    "offlayer_real",
    "offlayer_sham",
)


def _load(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _dump_gzip_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(value, fh, indent=2)
    tmp.replace(path)


def _mean(values) -> float:
    vals = [float(x) for x in values]
    if not vals:
        raise ValueError("cannot take the mean of an empty profile population")
    return st.fmean(vals)


def class_profiles(records: list[dict], position: int, layers: list[int],
                   min_class_n: int = 10, requested: set[str] | None = None) -> dict:
    """Return successful-attack and refusing-control means in the complement basis."""
    grouped: dict = defaultdict(lambda: {"attack": [], "control": []})
    for row in records:
        cls = row.get("class")
        if not cls or row.get("has_prefill") or cls == "prefill":
            continue
        if requested is not None and cls not in requested:
            continue
        if row.get("population") == "attack-COMPLY" and row.get("judge") == "COMPLY":
            grouped[cls]["attack"].append(row)
        elif row.get("population") == "control" and row.get("judge") == "REFUSE":
            grouped[cls]["control"].append(row)

    out = {}
    for cls, pops in sorted(grouped.items()):
        if min(len(pops["attack"]), len(pops["control"])) < min_class_n:
            continue
        targets = {}
        for pop in ("attack", "control"):
            targets[pop] = {}
            for layer in layers:
                key = f"{layer}@{position}"
                vals = [r["proj_complement"][key] for r in pops[pop]
                        if key in r.get("proj_complement", {})]
                if len(vals) != len(pops[pop]):
                    raise ValueError(f"{cls}/{pop} is missing {key} projections")
                targets[pop][layer] = _mean(vals)
        out[cls] = {
            "n_attack": len(pops["attack"]),
            "n_control_refuse": len(pops["control"]),
            "attack": targets["attack"],
            "control": targets["control"],
        }
    return out


def select_scope_jobs(judged: list[dict], limit_per_class: int | None = None) -> list[dict]:
    rows = []
    seen = defaultdict(int)
    for g in judged:
        if g.get("kind") != "control":
            continue
        cls = g["class"]
        if limit_per_class is not None and seen[cls] >= limit_per_class:
            continue
        seen[cls] += 1
        rows.append({
            "pair_id": f"{cls}:{g.get('base_id', g['record_idx'])}",
            "record_idx": g["record_idx"],
            "base_id": g.get("base_id"),
            "class": cls,
            "kind": "control",
            "text": g["prompt_text"],
            "prefill": g.get("prefill"),
            "historical_judge": g.get("judge"),
        })
    return rows


def select_restoration_jobs(judged: list[dict], eligible: set[str],
                            limit_per_class: int | None = None) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for g in judged:
        cls = g.get("class")
        if (g.get("kind") != "attack" or g.get("judge") != "COMPLY"
                or cls not in eligible or g.get("prefill")):
            continue
        if limit_per_class is not None and len(out[cls]) >= limit_per_class:
            continue
        out[cls].append({
            "pair_id": f"{cls}:{g.get('base_id', g['record_idx'])}",
            "record_idx": g["record_idx"],
            "base_id": g.get("base_id"),
            "class": cls,
            "kind": "attack",
            "text": g["prompt_text"],
            "prefill": None,
            "historical_judge": g.get("judge"),
        })
    return dict(out)


def exact_paired_p(real_only: int, reference_only: int) -> float:
    """Two-sided exact sign/McNemar test on discordant binary pairs."""
    n = real_only + reference_only
    if n == 0:
        return 1.0
    k = min(real_only, reference_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def _bootstrap_delta(pairs: list[tuple[int, int]], seed: int,
                     draws: int = 10_000) -> tuple[float, float]:
    if not pairs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(pairs)
    samples = []
    for _ in range(draws):
        total = 0
        for _j in range(n):
            a, b = pairs[rng.randrange(n)]
            total += a - b
        samples.append(total / n)
    samples.sort()
    return samples[int(.025 * draws)], samples[min(draws - 1, int(.975 * draws))]


def paired_stats(real_rows: list[dict], reference_rows: list[dict], seed: int = 0) -> dict:
    real = {r["pair_id"]: r for r in real_rows}
    ref = {r["pair_id"]: r for r in reference_rows}
    if set(real) != set(ref):
        raise ValueError("paired arms do not contain identical pair_id sets")
    pairs = []
    for key in sorted(real):
        pairs.append((int(real[key]["judge"] == "COMPLY"),
                      int(ref[key]["judge"] == "COMPLY")))
    real_only = sum(a == 1 and b == 0 for a, b in pairs)
    reference_only = sum(a == 0 and b == 1 for a, b in pairs)
    lo, hi = _bootstrap_delta(pairs, seed)
    return {
        "n": len(pairs),
        "real_comply": round(sum(a for a, _ in pairs) / len(pairs), 4),
        "reference_comply": round(sum(b for _, b in pairs) / len(pairs), 4),
        "delta": round(sum(a - b for a, b in pairs) / len(pairs), 4),
        "ci95_bootstrap": [round(lo, 4), round(hi, 4)],
        "discordant_real_only": real_only,
        "discordant_reference_only": reference_only,
        "p_exact": exact_paired_p(real_only, reference_only),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm family-wise adjusted p-values, returned in input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [1.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def _summary(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "n": n,
        "comply": round(sum(r.get("judge") == "COMPLY" for r in rows) / n, 4),
        "incoherent": sum(not r.get("coherent", True) for r in rows),
        "hit_token_cap": sum(bool(r.get("hit_token_cap")) for r in rows),
    }


def _stage_names(value: str) -> set[str]:
    allowed = {"scope", "transplant", "restoration", "analyze"}
    if value == "all":
        return allowed
    got = {x.strip() for x in value.split(",") if x.strip()}
    unknown = got - allowed
    if unknown:
        raise ValueError(f"unknown stages: {', '.join(sorted(unknown))}")
    return got


def _comparison(report: dict, family: str, name: str, real: list[dict], ref: list[dict],
                seed: int) -> None:
    report.setdefault("comparisons", {}).setdefault(family, {})[name] = paired_stats(
        real, ref, seed=seed)


def _adjust_families(report: dict) -> None:
    for comparisons in report.get("comparisons", {}).values():
        names = list(comparisons)
        adjusted = holm_adjust([comparisons[n]["p_exact"] for n in names])
        for name, p_adj in zip(names, adjusted):
            comparisons[name]["p_holm"] = p_adj


def _write_summary_csv(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["family", "comparison", "n", "real_comply", "reference_comply",
                         "delta", "ci95_low", "ci95_high", "p_exact", "p_holm"])
        for family, comparisons in report.get("comparisons", {}).items():
            for name, x in comparisons.items():
                writer.writerow([family, name, x["n"], x["real_comply"],
                                 x["reference_comply"], x["delta"],
                                 *x["ci95_bootstrap"], x["p_exact"], x["p_holm"]])


def main() -> int:
    import sys
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from f3_clamp.run import held_out_harmful
    from h3_causal.run import is_coherent
    from jbct.data import load_candidates, load_judged
    from jbct.directions import drop_dims
    from jbct.generate import generate
    from jbct.interventions import ProfileClamp, ProjectionClamp
    from jbct.judge import judge_refusal
    from jbct.models import load_model
    from jbct.provenance import env_versions
    from jbct.registry import get_model_spec

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODEL_SITES))
    ap.add_argument("--stage", default="all",
                    help="all or comma-separated: scope,transplant,restoration,analyze")
    ap.add_argument("--h1", default="results/h1")
    ap.add_argument("--h2", default="results/h2")
    ap.add_argument("--judged", default=None)
    ap.add_argument("--f1-projections", default="results/f1_geometry")
    ap.add_argument("--out", default="results/paper_closure")
    ap.add_argument("--causal-layer", type=int)
    ap.add_argument("--direction-position", type=int)
    ap.add_argument("--off-layer", type=int)
    ap.add_argument("--off-position", type=int)
    ap.add_argument("--classes", default="auto",
                    help="auto or a comma-separated subset; prefill is always excluded")
    ap.add_argument("--min-class-n", type=int, default=10)
    ap.add_argument("--scope-limit-per-class", type=int)
    ap.add_argument("--restoration-limit-per-class", type=int)
    ap.add_argument("--n-transplant", type=int, default=100)
    ap.add_argument("--n-fit", type=int, default=64)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--judge-model", default="openrouter/anthropic/claude-haiku-4.5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    stages = _stage_names(args.stage)
    defaults = MODEL_SITES[args.model]
    causal_layer = args.causal_layer if args.causal_layer is not None else defaults["causal"][0]
    position = (args.direction_position if args.direction_position is not None
                else defaults["causal"][1])
    off_layer = args.off_layer if args.off_layer is not None else defaults["off"][0]
    off_position = args.off_position if args.off_position is not None else defaults["off"][1]
    requested = None if args.classes == "auto" else {
        x.strip() for x in args.classes.split(",") if x.strip()
    }
    if requested and "prefill" in requested:
        raise SystemExit("prefill is not position-comparable and cannot be in --classes")

    model_out = Path(args.out) / args.model
    model_out.mkdir(parents=True, exist_ok=True)
    judged_path, judged = load_judged(args.model, args.h2, args.judged)
    projections_path = Path(args.f1_projections) / args.model / "projections.json.gz"
    if not projections_path.exists():
        raise SystemExit(f"missing {projections_path}")
    projection_rows = _load(projections_path)
    spec = get_model_spec(args.model)
    layers = list(range(spec.n_layers))
    profiles = class_profiles(projection_rows, position, layers,
                              min_class_n=args.min_class_n, requested=requested)
    if not profiles:
        raise SystemExit("no non-prefill classes meet --min-class-n")

    metadata = {
        "model": args.model,
        "causal_site": [causal_layer, position],
        "off_site": [off_layer, off_position],
        "judged_source": str(judged_path),
        "projection_source": str(projections_path),
        "profile_classes": {
            c: {"n_attack": p["n_attack"], "n_control_refuse": p["n_control_refuse"]}
            for c, p in profiles.items()
        },
        "excluded_profile_class": "prefill (assistant-prefix timing is not prompt-position comparable)",
        "max_new_tokens": args.max_new_tokens,
        "judge_model": args.judge_model,
        "seed": args.seed,
    }
    prior_report = model_out / "paper_closure_report.json"
    if prior_report.exists():
        prior = _load(prior_report)
        if prior.get("env"):
            metadata["env"] = prior["env"]

    # Analysis-only rescoring never loads model weights.
    generation_stages = stages & {"scope", "transplant", "restoration"}
    if generation_stages:
        cands = load_candidates(args.h1, args.model)
        dims = spec.outlier_dims

        def direction(layer: int, pos: int):
            key = f"{layer}@{pos}"
            if key not in cands:
                raise SystemExit(f"missing direction {key} in H1 candidates")
            return drop_dims(cands[key].detach().to(dtype=torch.float64), dims)

        causal_r = direction(causal_layer, position)
        off_r = direction(off_layer, off_position)
        directions = {layer: direction(layer, position) for layer in layers}
        rng = torch.Generator().manual_seed(args.seed)

        def sham_like(vec):
            # Match the real direction's support, then make the write direction exactly
            # orthogonal.  ProfileClamp reads the required scalar shift along ``vec`` and
            # writes that same scalar along this sham, so magnitude and support are held
            # fixed while causal direction is changed.
            real = vec / torch.linalg.vector_norm(vec)
            sham = drop_dims(
                torch.randn(vec.shape, generator=rng, dtype=torch.float64), dims)
            sham = sham - (sham @ real) * real
            return sham / torch.linalg.vector_norm(sham)

        causal_sham, off_sham = sham_like(causal_r), sham_like(off_r)
        profile_shams = {layer: sham_like(directions[layer]) for layer in layers}
        model, tok = load_model(spec, device=args.device, dtype=getattr(torch, args.dtype))
        metadata["env"] = env_versions()

        class _NoHook:
            n_calls = n_edits = 0

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

        def run_arm(stage: str, label: str, jobs: list[dict], hook_factory):
            path = model_out / stage / f"{label}.json.gz"
            expected = [j["pair_id"] for j in jobs]
            if path.exists() and not args.overwrite:
                rows = _load(path)
                if [r["pair_id"] for r in rows] != expected:
                    raise SystemExit(f"{path} exists but its paired population differs")
                print(f"[closure] reuse {stage}/{label}: {len(rows)}")
                return rows

            hook = hook_factory()
            print(f"[closure] generate {stage}/{label}: {len(jobs)}")
            with hook:
                gens = generate(model, tok, spec, [j["text"] for j in jobs],
                                prefills=[j.get("prefill") for j in jobs],
                                max_new_tokens=args.max_new_tokens,
                                batch_size=args.batch_size)
            rows = []
            for job, gen in zip(jobs, gens):
                rows.append({
                    **job,
                    "response": gen.response,
                    "n_prompt_tokens": gen.n_prompt_tokens,
                    "n_new_tokens": gen.n_new_tokens,
                    "hit_token_cap": gen.hit_token_cap,
                    "coherent": is_coherent(gen.response),
                })
            labels = judge_refusal(
                [{"prompt": r["text"], "response": r["response"]} for r in rows],
                model=args.judge_model,
            )
            for row, is_refusal in zip(rows, labels):
                row["judge"] = "REFUSE" if is_refusal else "COMPLY"
            _dump_gzip_atomic(path, rows)
            print(f"[closure] wrote {path}: {_summary(rows)}")
            return rows

        if "scope" in stages:
            jobs = select_scope_jobs(judged, args.scope_limit_per_class)
            factories = {
                "baseline": lambda: _NoHook(),
                "anchor_real": lambda: ProjectionClamp(
                    model, spec, causal_r, causal_layer, 0.0, position=position,
                    prompt_only=True),
                "anchor_sham": lambda: ProjectionClamp(
                    model, spec, causal_r, causal_layer, 0.0, position=position,
                    apply_along=causal_sham, prompt_only=True),
                "prompt_real": lambda: ProjectionClamp(
                    model, spec, causal_r, causal_layer, 0.0, position=None,
                    prompt_only=True),
                "prompt_sham": lambda: ProjectionClamp(
                    model, spec, causal_r, causal_layer, 0.0, position=None,
                    apply_along=causal_sham, prompt_only=True),
                "sustained_real": lambda: ProjectionClamp(
                    model, spec, causal_r, causal_layer, 0.0, position=None,
                    prompt_only=False),
                "sustained_sham": lambda: ProjectionClamp(
                    model, spec, causal_r, causal_layer, 0.0, position=None,
                    apply_along=causal_sham, prompt_only=False),
                "offlayer_real": lambda: ProjectionClamp(
                    model, spec, off_r, off_layer, 0.0, position=None,
                    prompt_only=False),
                "offlayer_sham": lambda: ProjectionClamp(
                    model, spec, off_r, off_layer, 0.0, position=None,
                    apply_along=off_sham, prompt_only=False),
            }
            for label in SCOPE_ARMS:
                run_arm("scope", label, jobs, factories[label])

        if "transplant" in stages:
            prompts = held_out_harmful(args.n_transplant, args.n_fit)
            jobs = [{"pair_id": f"heldout:{i}", "class": None, "kind": "harmful",
                     "text": p, "prefill": None} for i, p in enumerate(prompts)]
            run_arm("transplant", "baseline", jobs, lambda: _NoHook())
            zero = {layer: 0.0 for layer in layers}
            run_arm("transplant", "zero", jobs, lambda: ProfileClamp(
                model, spec, directions, zero, position=position, prompt_only=True))
            run_arm("transplant", "zero_sham", jobs, lambda: ProfileClamp(
                model, spec, directions, zero, position=position,
                apply_along=profile_shams, prompt_only=True))
            for cls, p in profiles.items():
                run_arm("transplant", f"{cls}__null", jobs, lambda p=p: ProfileClamp(
                    model, spec, directions, p["control"], position=position,
                    prompt_only=True))
                run_arm("transplant", f"{cls}__attack", jobs, lambda p=p: ProfileClamp(
                    model, spec, directions, p["attack"], position=position,
                    prompt_only=True))
                run_arm("transplant", f"{cls}__attack_sham", jobs,
                        lambda p=p: ProfileClamp(
                            model, spec, directions, p["attack"], position=position,
                            apply_along=profile_shams, prompt_only=True))

        if "restoration" in stages:
            jobs_by_class = select_restoration_jobs(
                judged, set(profiles), args.restoration_limit_per_class)
            for cls, jobs in jobs_by_class.items():
                p = profiles[cls]
                run_arm("restoration", f"{cls}__baseline", jobs, lambda: _NoHook())
                run_arm("restoration", f"{cls}__attack_null", jobs,
                        lambda p=p: ProfileClamp(
                            model, spec, directions, p["attack"], position=position,
                            prompt_only=True))
                run_arm("restoration", f"{cls}__harmful", jobs,
                        lambda p=p: ProfileClamp(
                            model, spec, directions, p["control"], position=position,
                            prompt_only=True))
                run_arm("restoration", f"{cls}__harmful_sham", jobs,
                        lambda p=p: ProfileClamp(
                            model, spec, directions, p["control"], position=position,
                            apply_along=profile_shams, prompt_only=True))

    # Reconstruct the report from disk so resumed arms and analysis-only runs are identical.
    report = {**metadata, "arms": {}, "comparisons": {}}
    for stage in ("scope", "transplant", "restoration"):
        stage_dir = model_out / stage
        report["arms"][stage] = {}
        if stage_dir.exists():
            for path in sorted(stage_dir.glob("*.json.gz")):
                report["arms"][stage][path.name[:-8]] = _summary(_load(path))

    def rows(stage, label):
        return _load(model_out / stage / f"{label}.json.gz")

    scope_dir = model_out / "scope"
    if all((scope_dir / f"{x}.json.gz").exists() for x in SCOPE_ARMS):
        for real, ref in (
            ("anchor_real", "baseline"), ("anchor_real", "anchor_sham"),
            ("prompt_real", "baseline"), ("prompt_real", "prompt_sham"),
            ("sustained_real", "baseline"), ("sustained_real", "sustained_sham"),
            ("prompt_real", "anchor_real"), ("sustained_real", "prompt_real"),
            ("offlayer_real", "baseline"), ("offlayer_real", "offlayer_sham"),
        ):
            _comparison(report, "scope", f"{real}_vs_{ref}", rows("scope", real),
                        rows("scope", ref), args.seed)

    transplant_dir = model_out / "transplant"
    if (transplant_dir / "baseline.json.gz").exists():
        for cls in profiles:
            labels = [f"{cls}__{suffix}" for suffix in ("null", "attack", "attack_sham")]
            if all((transplant_dir / f"{x}.json.gz").exists() for x in labels):
                _comparison(report, "transplant_vs_null", cls,
                            rows("transplant", labels[1]), rows("transplant", labels[0]),
                            args.seed)
                _comparison(report, "transplant_vs_sham", cls,
                            rows("transplant", labels[1]), rows("transplant", labels[2]),
                            args.seed)

    restoration_dir = model_out / "restoration"
    for cls in profiles:
        labels = [f"{cls}__{suffix}" for suffix in
                  ("baseline", "attack_null", "harmful", "harmful_sham")]
        if all((restoration_dir / f"{x}.json.gz").exists() for x in labels):
            _comparison(report, "restoration_vs_attack_null", cls,
                        rows("restoration", labels[2]), rows("restoration", labels[1]),
                        args.seed)
            _comparison(report, "restoration_vs_sham", cls,
                        rows("restoration", labels[2]), rows("restoration", labels[3]),
                        args.seed)

    _adjust_families(report)
    report_path = model_out / "paper_closure_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_summary_csv(report, model_out / "paper_closure_summary.csv")
    print(f"[closure] report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
