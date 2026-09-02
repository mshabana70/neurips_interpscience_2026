"""F3 — set a refused harmful prompt's refusal-direction magnitude to the jailbroken level.

This is the inverse of strong direction addition. Addition pushes the direction back
INTO prompts that had already jailbroken the model. F3 takes prompts the model refuses and
pulls the direction DOWN to the magnitude successful jailbreaks sit at, then asks whether
that alone induces compliance.

**Why a new intervention.** `DirectionalAblation` zeroes the component (target = 0) and
`ActivationAddition` moves by a fixed amount regardless of the starting point. Neither can
place a prompt at a *specified* magnitude. `ProjectionClamp` does: it reads the projection
and writes the difference.

**Basis.** The target and the clamped direction must be the same quantity, and `--basis`
picks both at once. The default `complement` reads F1's `proj_complement` and clamps the
outlier-dropped direction — the basis H3/F2 and H4 attribution already act in, and the one
`registry.py:127` says causal work must use. `raw` is self-consistent too, but writes to the
massive-activation dims, which `directions.py:99` records as collapsing the model; it is a
different experiment, and neither setting reproduces the original raw-target/complement-
direction mismatch, which is not worth being able to re-run.

**The premise is site-dependent, and this run tests that rather than assuming it.** The
figures below are from the RAW basis and are recorded here as history, not as the premise
the default run tests: bare harmful prompts sit ABOVE successful jailbreaks at gemma 21@-3
(+1.39 SD), qwen 33@-5 (+1.93) and qwen 18@-1 (+1.08), and BELOW at gemma 15@-2 (-0.57 SD).
That last verdict is 81% massive-activation structure and says nothing about the complement,
so `premise_holds` is re-read from whichever basis the run actually uses. gemma 15@-2 is
included either way: it is where F2 found the intervention works.

Arms:
  baseline      no intervention. These prompts have no H2 record, so their refusal rate has
                to be measured here rather than looked up.
  clamp@a{A}    target = mu_harmful + A * (mu_jailbroken - mu_harmful). A=1 is exactly
                "jailbroken level"; A=0 clamps to the harmful mean, which moves the mean
                nowhere and so isolates whether clamping ITSELF does anything.
  sham@a{A}     the same per-prompt shift along a RANDOM direction. Matched in magnitude,
                differing only in direction.

Run:
    python experiments/f3_clamp/run.py --model gemma-3-4b --site 21@-3 \
        --alphas 0,0.5,1.0,1.5 --n 150
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

ARMS_NOTE = "baseline + clamp/sham at each alpha"

# The target and the intervention MUST live in the same basis. They did not, and the run it
# cost is worth spelling out: F1 measured `proj` along the raw diff-in-means direction while
# this stage clamped along the complement (outlier dims dropped). At gemma 21@-3 that meant
# reading a target of -1,401 for a quantity whose true value was +2,194 — opposite signs, a
# -3,595 shift in the arm labelled "null". At 15@-2 the target was -29,983 against a
# complement projection of similar smallness, so every arm took a ~30,000 kick and returned
# 150/150 incoherent. One switch now picks both halves so they cannot drift apart again.
BASIS_FIELD = {"complement": "proj_complement", "raw": "proj"}


def basis_dims(outlier_dims: tuple, basis: str) -> tuple:
    """Dims to zero for a basis — the other half of the `BASIS_FIELD` choice."""
    return tuple(outlier_dims) if basis == "complement" else ()


def require_projection_field(projections: list[dict], field: str, site: str) -> None:
    """Fail loudly when F1's artifact predates the basis it is being asked for.

    Falling back to `proj` would silently reproduce the original bug, which is this repo's
    signature failure mode: a wrong number rather than an exception.
    """
    if not any(site in r.get(field, {}) for r in projections):
        raise SystemExit(
            f"projections.json carries no '{field}' at {site}. It predates the two-basis "
            f"F1. Re-run F1 to produce it:\n"
            f"    uv run python experiments/f1_geometry/run.py --model <model> "
            f"--pca-sites {site}\n"
            f"(--basis raw runs against the old field, but clamps the RAW direction — a "
            f"different experiment, not the mismatched original.)"
        )


def population_stats(projections: list[dict], site: str, field: str = "proj") -> dict:
    """Mean projection per population at one site, from F1's committed per-prompt values.

    `field` selects the basis; it is never inferred. See `BASIS_FIELD`.
    """
    out = {}
    for pop in ("harmful", "attack-COMPLY", "attack-REFUSE", "clean", "control"):
        vals = [r[field][site] for r in projections
                if r["population"] == pop and site in r.get(field, {})]
        if vals:
            out[pop] = {"n": len(vals), "mean": st.mean(vals),
                        "sd": st.pstdev(vals) if len(vals) > 1 else None}
    return out


def clamp_target(stats: dict, alpha: float) -> float:
    """Interpolate from the harmful mean toward the jailbroken mean.

    alpha=0 lands on the harmful mean (the population is already there, so this isolates the
    act of clamping from the act of moving); alpha=1 lands on the jailbroken mean, which is
    the experiment as specified.
    """
    lo, hi = stats["harmful"]["mean"], stats["attack-COMPLY"]["mean"]
    return lo + alpha * (hi - lo)


def premise_holds(stats: dict) -> bool:
    """Does the population actually sit ABOVE the jailbroken level at this site?

    If not, alpha=1 raises the magnitude rather than reducing it, and the arm is testing
    something other than what F3 describes. Reported, never silently corrected.

    **Basis-dependent, and read from whichever basis `stats` came from.** gemma 15@-2 fails
    the premise in the raw basis (harmful -29,983 vs jailbroken -28,400) but that number is
    81% massive-activation structure, so the raw verdict says nothing about the complement.
    """
    return stats["harmful"]["mean"] > stats["attack-COMPLY"]["mean"]


def held_out_harmful(n: int, n_fit: int) -> list[str]:
    """Harmful instructions that did NOT fit the direction.

    `harmful_val` holds only 39, so a run of any size has to reach into `harmful_train`
    beyond the first `n_fit` that H1 consumed. Those are genuinely unseen by the fit; the
    first `n_fit` are not, and are never returned.
    """
    from jbct.data import load_direction_corpus

    train, _ = load_direction_corpus("train")
    val, _ = load_direction_corpus("val")
    pool = val + train[n_fit:]
    if len(pool) < n:
        raise SystemExit(
            f"only {len(pool)} held-out harmful prompts available "
            f"({len(val)} val + {len(train)} train minus {n_fit} used to fit); asked for {n}"
        )
    return pool[:n]


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "comply": round(sum(r["judge"] == "COMPLY" for r in rows) / n, 4),
        "incoherent": sum(not r["coherent"] for r in rows),
        "mean_shift_applied": (round(st.mean([r["shift"] for r in rows if r.get("shift")]), 2)
                               if any(r.get("shift") for r in rows) else None),
    }


def _verdict(real: float, sham: float) -> str:
    return ("specific" if abs(real) > 2 * abs(sham) + 0.05
            else "NOT specific — a random direction moves it comparably")


def specificity(arms: dict, baseline_comply: float) -> dict:
    """Real clamp against its matched sham, per alpha, against TWO reference points.

    `vs_baseline` is the un-intervened rate. `vs_null` is the alpha=0 arm, and it is the one
    the design actually called for: this module's docstring specifies alpha=0 as the arm that
    "moves the mean nowhere and so isolates whether clamping ITSELF does anything".

    They differ because a clamp does two things at once — it moves the population AND it
    removes the variance within it. At gemma 15@-2 alpha=0 came in at 0.007 against a 0.113
    baseline: collapsing every prompt onto the harmful mean drags the naturally-low-projection
    prompts (the ones already complying) back up, so they start refusing. Measuring the other
    arms from the baseline therefore charges them for a homogenization cost that alpha=0 has
    already paid, and understates the effect of WHERE the target sits. Both are reported so
    the choice of reference is visible rather than assumed.
    """
    null_arm = arms.get("clamp@a0")
    null_comply = null_arm.get("comply") if null_arm and null_arm.get("n") else None
    null_sham = arms.get("sham@a0")
    null_sham_comply = null_sham.get("comply") if null_sham and null_sham.get("n") else None

    out = {}
    for label, res in arms.items():
        if not label.startswith("clamp@") or not res.get("n"):
            continue
        shm = arms.get(label.replace("clamp@", "sham@", 1))
        if not shm or not shm.get("n"):
            continue
        real = res["comply"] - baseline_comply
        sham = shm["comply"] - baseline_comply
        entry = {
            "real_shift": round(real, 4),
            "sham_shift": round(sham, 4),
            "real_incoherent": res["incoherent"],
            "sham_incoherent": shm["incoherent"],
            "verdict": _verdict(real, sham),
        }
        if null_comply is not None and null_sham_comply is not None:
            real_n = res["comply"] - null_comply
            sham_n = shm["comply"] - null_sham_comply
            entry["real_shift_vs_null"] = round(real_n, 4)
            entry["sham_shift_vs_null"] = round(sham_n, 4)
            entry["verdict_vs_null"] = _verdict(real_n, sham_n)
        out[label] = entry
    return out


def rescore(report: dict) -> dict:
    """Recompute the specificity block of a finished run. Pure arithmetic, no GPU.

    Every input lives in the report already, so a reference-point change never costs a
    re-run — which matters when one arm is 150 prompts x 1024 tokens.
    """
    base = report["arms"]["baseline"]["comply"]
    report["specificity"] = specificity(report["arms"], base)
    return report


def main() -> int:
    import sys

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from jbct.data import load_candidates, read_json_maybe_gz
    from jbct.directions import drop_dims
    from jbct.generate import generate
    from jbct.interventions import ProjectionClamp
    from jbct.judge import judge_refusal
    from jbct.models import load_model
    from jbct.provenance import env_versions
    from jbct.registry import get_model_spec

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from f1_geometry.run import parse_site
    from h3_causal.run import is_coherent

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--site", required=True, help="'layer@position', e.g. 21@-3")
    ap.add_argument("--alphas", default="0,0.5,1.0,1.5",
                    help="fractions of the harmful->jailbroken gap; 1.0 is 'jailbroken level'")
    ap.add_argument("--n", type=int, default=150, help="held-out harmful prompts")
    ap.add_argument("--n-fit", type=int, default=64,
                    help="how many harmful_train prompts H1 used to fit; never reused here")
    ap.add_argument("--f1", default="results/f1_geometry",
                    help="where F1's projections.json lives — supplies the target magnitudes")
    ap.add_argument("--h1", default="results/h1")
    ap.add_argument("--clamp-positions", default="anchor", choices=["anchor", "all"],
                    help="'anchor' (default) edits only the token the target was measured "
                         "at; 'all' overwrites every position, which is what the first run "
                         "did and what left gemma 15@-2 at 150/150 incoherent")
    ap.add_argument("--basis", default="complement", choices=["complement", "raw"],
                    help="'complement' (default) reads F1's proj_complement and clamps the "
                         "outlier-dropped direction — the basis every causal stage acts in. "
                         "'raw' reads proj and clamps the raw direction, which writes to the "
                         "massive-activation dims. Reading one basis and clamping the other "
                         "is what left gemma 15@-2 at 150/150 incoherent.")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--judge-model", default="openrouter/anthropic/claude-haiku-4.5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/f3_clamp")
    ap.add_argument("--rescore", action="store_true",
                    help="recompute the specificity block of an existing f3_report.json and "
                         "exit. No model, no GPU, no generation — the arms already hold "
                         "every number it needs")
    args = ap.parse_args()

    spec = get_model_spec(args.model)
    layer, position = parse_site(args.site)
    alphas = [float(a) for a in args.alphas.split(",")]

    if args.rescore:
        path = Path(args.out) / args.model / args.site.replace("@", "p") / "f3_report.json"
        if not path.exists():
            raise SystemExit(f"{path} not found — nothing to rescore")
        rep = rescore(json.loads(path.read_text()))
        path.write_text(json.dumps(rep, indent=2))
        print(f"[f3] rescored {path}")
        print_specificity(rep)
        return 0

    proj_path = Path(args.f1) / args.model / "projections.json"
    try:
        rows_f1 = read_json_maybe_gz(proj_path)
    except FileNotFoundError as exc:
        raise SystemExit(f"{exc} — F3's targets come from F1. Run F1 first.") from exc
    field = BASIS_FIELD[args.basis]
    dims = basis_dims(spec.outlier_dims, args.basis)
    require_projection_field(rows_f1, field, args.site)
    stats = population_stats(rows_f1, args.site, field=field)
    for pop in ("harmful", "attack-COMPLY"):
        if pop not in stats:
            raise SystemExit(f"F1 has no '{pop}' projections at {args.site}")
    holds = premise_holds(stats)
    gap = stats["harmful"]["mean"] - stats["attack-COMPLY"]["mean"]
    print(f"[f3] basis={args.basis} (target field '{field}', dims dropped {list(dims)})")
    print(f"[f3] {args.model} {args.site}: harmful mean {stats['harmful']['mean']:,.1f}, "
          f"jailbroken mean {stats['attack-COMPLY']['mean']:,.1f}, gap {gap:,.1f}")
    if not holds:
        print("[f3] NOTE the premise does NOT hold here: refused harmful prompts already sit "
              "BELOW the jailbroken level, so alpha=1 RAISES the magnitude. Recorded, not "
              "corrected — this site is a negative control.")

    prompts = held_out_harmful(args.n, args.n_fit)
    print(f"[f3] {len(prompts)} held-out harmful prompts (none used to fit the direction)")

    cands = load_candidates(args.h1, args.model)
    key = f"{layer}@{position}"
    if key not in cands:
        raise SystemExit(f"{key} not in candidates.pt; have {sorted(cands)[:8]}...")
    r_raw = cands[key].detach().to(dtype=torch.float64)
    # `dims` came from the SAME --basis switch that chose `field` above, so the vector we
    # clamp along and the numbers we clamp it to are guaranteed to be the same quantity.
    r_hat = drop_dims(r_raw, dims)
    gen_t = torch.Generator().manual_seed(args.seed)
    sham_hat = drop_dims(torch.randn(r_hat.shape, generator=gen_t, dtype=torch.float64), dims)

    model, tok = load_model(spec, device=args.device, dtype=getattr(torch, args.dtype))

    def run_arm(label, hook_factory):
        shifts: list[float] = []
        hook = hook_factory(shifts)
        with hook:
            gens = generate(model, tok, spec, prompts, prefills=[None] * len(prompts),
                            max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
        # `shifts` now carries one entry per prompt, recorded on each batch's prompt pass
        # and therefore aligned with `prompts`. The first run recorded a single scalar and
        # labelled it a mean.
        rows = [{"prompt": p, "response": g.response, "coherent": is_coherent(g.response),
                 "hit_token_cap": g.hit_token_cap,
                 "shift": shifts[i] if i < len(shifts) else None}
                for i, (p, g) in enumerate(zip(prompts, gens))]
        labels = judge_refusal([{"prompt": r["prompt"], "response": r["response"]}
                                for r in rows], model=args.judge_model)
        for row, is_refusal in zip(rows, labels):
            row["judge"] = "REFUSE" if is_refusal else "COMPLY"
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / f"{label}_generations.json").write_text(json.dumps(rows, indent=2))
        res = summarize(rows)
        print(f"  [f3] {label:<14} comply {res['comply']:.3f}  "
              f"incoherent {res['incoherent']}/{res['n']}")
        return res

    outdir = Path(args.out) / args.model / args.site.replace("@", "p")
    arms: dict = {}

    class _NoHook:
        n_calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    arms["baseline"] = run_arm("baseline", lambda _s: _NoHook())
    base_comply = arms["baseline"]["comply"]

    clamp_pos = None if args.clamp_positions == "all" else position
    for a in alphas:
        target = clamp_target(stats, a)
        arms[f"clamp@a{a:g}"] = run_arm(
            f"clamp@a{a:g}",
            lambda s, t=target: ProjectionClamp(model, spec, r_hat, layer=layer, target=t,
                                               position=clamp_pos, record=s))
        # EXACTLY matched: the shift is computed per prompt from r_hat and written along the
        # random direction, so the arms differ in direction alone. The first run applied one
        # prompt's shift as a constant to all of them.
        arms[f"sham@a{a:g}"] = run_arm(
            f"sham@a{a:g}",
            lambda s, t=target: ProjectionClamp(model, spec, r_hat, layer=layer, target=t,
                                                position=clamp_pos, apply_along=sham_hat,
                                                record=s))

    report = {
        "env": env_versions(),
        "model": args.model,
        "site": args.site,
        "layer": layer,
        "position": position,
        "dtype": args.dtype,
        "n_prompts": len(prompts),
        "n_fit_excluded": args.n_fit,
        "basis": args.basis,
        "target_field": field,
        "outlier_dims_dropped": list(dims),
        "population_stats_from_f1": {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                                         for kk, vv in v.items()} for k, v in stats.items()},
        "premise_holds": holds,
        "harmful_minus_jailbroken": round(gap, 4),
        "targets": {f"a{a:g}": round(clamp_target(stats, a), 4) for a in alphas},
        "sham_seed": args.seed,
        "clamp_positions": args.clamp_positions,
        "arms": arms,
        "specificity": specificity(arms, base_comply),
    }
    # The invariant that would have caught the basis mismatch on the first run: at alpha=0
    # the target IS the harmful mean, so the shift can only be as large as the difference
    # between F1's harmful sample and this run's held-out one. A shift several times the
    # harmful->jailbroken gap means the target and the projection are not the same quantity.
    a0 = arms.get("clamp@a0", {}).get("mean_shift_applied")
    if a0 is not None and abs(gap) > 0:
        ratio = abs(a0) / abs(gap)
        report["null_arm_shift_over_gap"] = round(ratio, 3)
        if ratio > 1.0:
            print(f"\n[f3] WARNING alpha=0 moved the projection by {a0:,.1f}, "
                  f"{ratio:.1f}x the harmful->jailbroken gap of {gap:,.1f}. That arm is "
                  f"supposed to be a near-null. Suspect the target and the clamped "
                  f"direction are in different bases (--basis {args.basis}).")

    (outdir / "f3_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n[f3] -> {outdir}")
    print(f"  basis {args.basis}  baseline comply {base_comply:.3f}  "
          f"(premise_holds={holds})")
    print_specificity(report)
    return 0


def print_specificity(report: dict) -> None:
    """Both reference points, side by side, so neither can be quoted without the other."""
    arms = report.get("arms", {})
    null_comply = arms.get("clamp@a0", {}).get("comply")
    for k, v in report.get("specificity", {}).items():
        line = (f"  {k:<14} vs baseline: real {v['real_shift']:+.3f} "
                f"sham {v['sham_shift']:+.3f} {v['verdict'].split(' —')[0]}")
        if "real_shift_vs_null" in v:
            line += (f"   | vs a0: real {v['real_shift_vs_null']:+.3f} "
                     f"sham {v['sham_shift_vs_null']:+.3f} "
                     f"{v['verdict_vs_null'].split(' —')[0]}")
        print(line)
    # A clamp removes within-population variance as well as moving the mean. When alpha=0
    # is far from the baseline it has measured that cost, and every other arm inherits it.
    base = arms.get("baseline", {}).get("comply")
    if null_comply is not None and base is not None and abs(null_comply - base) > 0.05:
        print(f"  NOTE alpha=0 sits {null_comply - base:+.3f} from the un-intervened "
              f"baseline, so clamping ITSELF moves compliance. Read the 'vs a0' column: it "
              f"is the one that isolates where the target sits.")


if __name__ == "__main__":
    raise SystemExit(main())
