"""H1 — refusal is a linear direction. Extract it, and prove the extraction is trustworthy.

Three outputs, in increasing order of what they prove:

1. **Separation curve** — ||mean(harmful) - mean(harmless)|| per layer, at the model's
   canonical position. Picks the max-separation layer (Refusal-Lens' `find_best_layer`).
2. **Arditi grid** — a candidate at each (layer, position) in `spec.direction_positions`,
   so a model whose `direction_position` is a PLACEHOLDER (Qwen) gets it swept rather than
   assumed. This is the provenance gate: placeholder -> verified.
3. **Held-out validation** — project the val split onto r-hat at the selected (layer, pos)
   and report AUC. A direction that doesn't separate unseen harmful/harmless is not a
   refusal direction, however clean its separation curve looks.

`--reconcile` additionally re-runs Gemma under Refusal-Lens' OLD conventions and reports how
the result relates to its published numbers. ADVISORY ONLY — matching the old repo 1-to-1 is
not the bar (that would mean reproducing its transformers version and its defects). What
validates H1 is the held-out AUC plus the recorded provenance.

Run:
    python experiments/h1_direction/run.py --model gemma-3-4b --n 64 --reconcile
    python experiments/h1_direction/run.py --model qwen3-4b --n 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Published Refusal-Lens Gemma-3-4B separations (n=64, fp32, pos=-2, hidden_states[L+1]).
# Source: scripts/pipeline_qwen/PIPELINE_PLAN.md:81 and scripts/pipeline/README.md.
# L33 is the post-final-RMSNorm value and is NOT comparable to our raw-hook convention.
RECONCILE_REFERENCE = {15: 3101.2, 32: 20873.2}
RECONCILE_MODEL = "gemma-3-4b"  # the only model with published reference separations
RECONCILE_N = 64  # the n those references were measured at; a different n will not match

# The golden vectors themselves (Refusal-Lens run_20260417_010035, reproduced bit-for-bit by
# three independent runs). See data/golden/gemma-3-4b/PROVENANCE.json.
RECONCILE_VECTORS = "data/golden/gemma-3-4b/refusal_lens_n64_unnormalized_r.pt"

# ADVISORY ONLY — reconciliation never gates the run.
#
# Reproducing the old repo's numbers 1-to-1 is NOT the bar; a consistent, correct methodology
# is. Exact agreement would require also reproducing its transformers version and its batched
# left-padded forward pass, and chasing that would mean preserving defects we deliberately
# fixed (double BOS, hidden_states last-layer norm). What actually validates H1 is the
# held-out AUC plus the provenance checks (position read, BOS count, dtype, layer count).
# These numbers are reported as context for interpreting cross-repo comparisons, nothing more.
RECONCILE_COS_REF = 0.999  # cosine at/above this means "same direction, different scale"


def separation_curve(directions) -> dict[int, float]:
    from jbct.directions import normalize_directions

    _unit, separation = normalize_directions(directions)
    return separation


def auc(pos_scores: list[float], neg_scores: list[float]) -> float:
    """Mann-Whitney AUC: P(score(harmful) > score(harmless)). 0.5 = chance."""
    wins = sum(
        1.0 if p > n else 0.5 if p == n else 0.0 for p in pos_scores for n in neg_scores
    )
    return wins / (len(pos_scores) * len(neg_scores))


def legacy_direction(model, tokenizer, spec, harmful, harmless, layers):
    """Deliberately reproduce Refusal-Lens' OLD conventions, returning unnormalized r.

    For reconciliation only.

    QUARANTINED here rather than in `jbct/` because two of these are defects we fixed:
      - reads `hidden_states[L+1]` (final layer is post-RMSNorm) instead of the raw hook
      - tokenizes with the default `add_special_tokens=True`, i.e. a DOUBLE BOS
    Do not import this into library code. It exists to show the delta is understood.
    """
    import torch

    from jbct.models import format_prompt

    def means(prompts):
        acc = {layer: None for layer in layers}
        for text in prompts:
            ids = tokenizer(format_prompt(spec, tokenizer, text), return_tensors="pt").to(
                model.device
            )  # add_special_tokens defaults True -> the old double BOS
            with torch.no_grad():
                out = model(**ids, output_hidden_states=True)
            for layer in layers:
                v = out.hidden_states[layer + 1][0, -2, :].to(torch.float64)  # old pos=-2
                acc[layer] = v.clone() if acc[layer] is None else acc[layer] + v
        return {layer: acc[layer] / len(prompts) for layer in layers}

    hf, hl = means(harmful), means(harmless)
    return {layer: hf[layer] - hl[layer] for layer in layers}  # UNnormalized r, as the golden file stores


def main() -> int:
    import torch

    from jbct.data import load_direction_corpus
    from jbct.directions import compute_direction_candidates, compute_directions
    from jbct.models import HookManager, format_prompt, load_model, tokenize_prompt
    from jbct.provenance import env_versions
    from jbct.registry import get_model_spec

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=64, help="prompts per class (old default: 64)")
    ap.add_argument("--n-val", type=int, default=32)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reconcile", action="store_true", help="Gemma-only sanity vs old numbers")
    ap.add_argument("--out", default="results/h1")
    args = ap.parse_args()

    if args.reconcile:
        # Both guards matter: the reference separations are Gemma's, measured at n=64. Run
        # them against another model or another n and the check "fails" for reasons that have
        # nothing to do with whether the codebase is correct — which is worse than no check.
        if args.model != RECONCILE_MODEL:
            raise SystemExit(
                f"--reconcile compares against published {RECONCILE_MODEL} separations; "
                f"there are no reference values for {args.model}"
            )
        if args.n != RECONCILE_N:
            raise SystemExit(
                f"--reconcile requires --n {RECONCILE_N} (the n the references were measured "
                f"at); got --n {args.n}"
            )

    spec = get_model_spec(args.model)
    dtype = getattr(torch, args.dtype)
    # fp32 is the methodology behind every verified Refusal-Lens number; bf16 directions are
    # self-consistent but NOT numerically comparable to them (audit F6).
    model, tok = load_model(spec, device=args.device, dtype=dtype)

    harmful, harmless = load_direction_corpus("train", n=args.n)
    fmt = [format_prompt(spec, tok, t) for t in harmful]
    fml = [format_prompt(spec, tok, t) for t in harmless]

    trace: dict = {}
    directions = compute_directions(model, tok, spec, fmt, fml, trace=trace)
    sep = separation_curve(directions)

    # Arditi grid -> best (layer, position), which is what promotes a placeholder position.
    grid = compute_direction_candidates(model, tok, spec, fmt, fml)
    grid_sep = {k: float(torch.linalg.vector_norm(v)) for k, v in grid.items()}
    best_lp = max(grid_sep, key=grid_sep.get)

    # Select by HELD-OUT AUC, not by ||r||.
    #
    # argmax of raw separation is not a refusal criterion — it is a depth criterion. Residual
    # norms grow monotonically with layer (Refusal-Lens: L15=3,101 -> L32=20,873, a 6.7x ramp
    # from depth alone), so argmax||r|| lands on the last layer for every model. Their runs
    # peaked at L32 only because the `hidden_states` convention post-RMSNorms the final entry
    # and collapsed L33 to 287; our raw hook has no such collapse, so nothing stops the walk.
    # AUC is rank-based and therefore scale-invariant, and selecting on held-out data is what
    # Arditi actually does (validation criterion, not magnitude).
    v_harmful, v_harmless = load_direction_corpus("val", n=args.n_val)
    unit = {k: v / torch.linalg.vector_norm(v) for k, v in grid.items()}

    def project_all(prompts):
        """One forward pass per prompt, projected at EVERY candidate (layer, position)."""
        out = {k: [] for k in unit}
        for text in prompts:
            ids = tokenize_prompt(tok, format_prompt(spec, tok, text), device=model.device)
            with HookManager(model, spec) as hm:
                model(**ids)
            for (layer, p), r_hat in unit.items():
                out[(layer, p)].append(float(hm.residuals[layer][0, p].to(torch.float64) @ r_hat))
        return out

    hs, ls = project_all(v_harmful), project_all(v_harmless)
    val_auc_grid = {k: auc(hs[k], ls[k]) for k in unit}
    best_lp = max(val_auc_grid, key=val_auc_grid.get)
    sel_layer, sel_pos = best_lp
    val_auc = val_auc_grid[best_lp]
    best_by_separation = max(grid_sep, key=grid_sep.get)

    report = {
        "model": args.model,
        "dtype": args.dtype,
        "n_train": args.n,
        "n_val": args.n_val,
        "provenance": {
            "env": env_versions(),
            "position_read": trace.get("positions"),
            "n_leading_bos": trace.get("n_leading_bos"),
            "model_dtype": trace.get("dtype"),
            "last_layer_convention": "raw residual (NOT post-final-RMSNorm)",
            "spec_direction_position": [
                spec.direction_position.value,
                spec.direction_position.provenance,
            ],
        },
        "selection_criterion": (
            "held-out validation AUC (rank-based, scale-invariant). NOT argmax||r||, which is "
            "dominated by residual-norm growth with depth and lands on the last layer."
        ),
        "arditi_grid_best": {"layer": sel_layer, "position": sel_pos, "val_auc": round(val_auc, 4)},
        "val_auc_at_selected": round(val_auc, 4),
        "val_auc_grid": {f"L{k[0]}@{k[1]}": round(v, 4) for k, v in sorted(val_auc_grid.items())},
        # Context only — the magnitude view, which is what the depth artifact shows up in.
        "separation_by_layer": {str(k): round(v, 4) for k, v in sep.items()},
        "best_layer_by_separation_CONTEXT_ONLY": {
            "layer": best_by_separation[0],
            "position": best_by_separation[1],
            "note": "expected to sit at/near the last layer; do not use for selection",
        },
        "arditi_grid_separation": {f"L{k[0]}@{k[1]}": round(v, 4) for k, v in grid_sep.items()},
        "registry_layers": {
            "causal_layer": [spec.causal_layer.value, spec.causal_layer.provenance],
            "separation_layer": [spec.separation_layer.value, spec.separation_layer.provenance],
        },
    }

    if args.reconcile:
        layers = sorted(RECONCILE_REFERENCE)
        legacy_r = legacy_direction(model, tok, spec, fmt, fml, layers)
        golden = torch.load(RECONCILE_VECTORS, weights_only=True)
        checks = {}
        for layer, expected in RECONCILE_REFERENCE.items():
            ours = legacy_r[layer]
            ref = golden[str(layer)].double()
            cos = float(
                torch.nn.functional.cosine_similarity(
                    ours.unsqueeze(0).cpu(), ref.unsqueeze(0), dim=1
                )
            )
            got = float(torch.linalg.vector_norm(ours))
            rel = abs(got - expected) / expected
            checks[f"L{layer}"] = {
                "cosine_vs_reference": round(cos, 6),
                "same_direction": cos >= RECONCILE_COS_REF,
                "separation_reference": expected,
                "separation_got": round(got, 2),
                "separation_rel_err": round(rel, 4),
            }
        report["reconciliation"] = {
            "status": "ADVISORY — does not gate the run",
            "conventions": "hidden_states[L+1], double BOS, pos=-2, fp32 (Refusal-Lens Stage 01)",
            "how_to_read": (
                "cosine near 1.0 means the same direction at a different scale, which is the "
                "expected outcome across transformers versions and batched-vs-single forwards. "
                "Absolute separation is not expected to match and is not required to."
            ),
            "reference_source": RECONCILE_VECTORS,
            "checks": checks,
        }

    outdir = Path(args.out) / args.model
    outdir.mkdir(parents=True, exist_ok=True)
    # Save on CPU: a tensor saved from CUDA carries cuda:0 into every consumer, which breaks
    # loading on a CPU-only box and invites silent device mismatches downstream.
    torch.save({str(k): v.cpu() for k, v in directions.items()}, outdir / "directions.pt")
    torch.save({f"{k[0]}@{k[1]}": v.cpu() for k, v in grid.items()}, outdir / "candidates.pt")
    (outdir / "h1_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    # H1 passes on held-out generalization + provenance, NOT on matching the old repo.
    ok = val_auc >= 0.9
    print(f"\n{'PASS' if ok else 'CHECK'} — best (L{sel_layer}, pos {sel_pos}), val AUC {val_auc:.3f}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
