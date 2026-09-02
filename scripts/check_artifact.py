"""Validate the anonymous artifact and optionally rebuild its paper figures."""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("gemma-3-4b", "qwen3-4b")
CLASSES = {
    "authoritative_tone",
    "fiction",
    "narrative_sandwich",
    "prefill",
    "refusal_suppression",
    "refusal_suppression_long",
    "roleplay",
}
FORBIDDEN_PUBLIC_KEYS = {"prompt", "prompt_text", "text", "prefill", "response"}
EXPECTED_FIGURES = {
    "fig1_method_overview.png",
    "fig2_refusal_trajectories.png",
    "fig3_scope_points.png",
    "fig4_causal_contrast.png",
    "fig1_direction_auc.png",
    "fig2_generation_validation.png",
    "fig3_jailbreak_dataset.png",
    "fig4_refusal_signal_by_class.png",
    "fig4_closure_scope.png",
    "fig5_closure_transplant.png",
    "fig6_closure_restoration.png",
    "fig6_sustained_addition.png",
    "figA1_gemma_causal_sweep.png",
    "figA2_qwen_causal_sweep.png",
    "figA3_jailbreak_examples.png",
    "figA4_intervention_examples.png",
}


def read_json(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def require(path: Path, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"missing required file: {path.relative_to(ROOT)}")


def forbidden_keys(value) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        found.update(FORBIDDEN_PUBLIC_KEYS.intersection(value))
        for child in value.values():
            found.update(forbidden_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(forbidden_keys(child))
    return found


def validate_inputs(errors: list[str]) -> None:
    dataset_path = ROOT / "data/jailbreaks/jailbreak_dataset.json"
    require(dataset_path, errors)
    if dataset_path.exists():
        records = read_json(dataset_path)["records"]
        for model in MODELS:
            selected = [row for row in records if row.get("model") in (None, model)]
            counts = Counter(row["class"] for row in selected)
            if set(counts) != CLASSES or any(counts[name] != 50 for name in CLASSES):
                errors.append(f"{model}: expected 50 dataset rows for each of seven classes")

    expected_corpus_sizes = {
        "harmful_train.json": 260,
        "harmful_val.json": 39,
        "harmless_train.json": 512,
        "harmless_val.json": 512,
    }
    for name, expected in expected_corpus_sizes.items():
        path = ROOT / "data/refusal_direction" / name
        require(path, errors)
        if path.exists() and len(read_json(path)) != expected:
            errors.append(f"{path.relative_to(ROOT)}: expected {expected} rows")


def validate_reports(errors: list[str]) -> None:
    expected_cells = {"gemma-3-4b": 102, "qwen3-4b": 108}
    h3_arms = {
        "ablate_control",
        "sham_control",
        "add_attack@c0.25",
        "add_attack@c0.5",
        "add_attack@c1",
        "sham_attack@c0.25",
        "sham_attack@c0.5",
        "sham_attack@c1",
    }
    for model in MODELS:
        paths = {
            "h1": ROOT / f"results/h1/{model}/h1_report.json",
            "h2": ROOT / f"results/h2/{model}/h2_report.json",
            "f1": ROOT / f"results/f1_geometry/{model}/f1_report.json",
            "f1_categories": ROOT
            / f"results/f1_categories/{model}/f1_categories_report.json.gz",
            "grid": ROOT / f"results/arditi_gap/{model}/proxy_grid_report.json",
            "validation": ROOT
            / f"results/arditi_gap/{model}/generation_validation_report.json",
            "h3": ROOT / f"results/h3_causal/{model}/h3_causal_report.json",
            "closure": ROOT
            / f"results/paper_closure/{model}/paper_closure_report.json",
        }
        for path in paths.values():
            require(path, errors)
        if any(not path.exists() for path in paths.values()):
            continue

        if read_json(paths["h1"]).get("model") != model:
            errors.append(f"{model}: H1 model field mismatch")
        if set(read_json(paths["h2"])["per_class"]) != CLASSES:
            errors.append(f"{model}: H2 class coverage mismatch")
        if read_json(paths["grid"]).get("n_cells") != expected_cells[model]:
            errors.append(f"{model}: incomplete Arditi causal grid")
        if set(read_json(paths["h3"])["arms"]) != h3_arms:
            errors.append(f"{model}: H3 arm coverage mismatch")
        if set(read_json(paths["closure"])["arms"]) != {
            "scope",
            "transplant",
            "restoration",
        }:
            errors.append(f"{model}: closure arm coverage mismatch")


def validate_public_labels(errors: list[str]) -> None:
    for experiment in ("arditi_gap", "h3_causal", "paper_closure"):
        for model in MODELS:
            path = ROOT / f"results/public_labels/{experiment}/{model}.json.gz"
            require(path, errors)
            if not path.exists():
                continue
            rows = read_json(path)
            if not rows:
                errors.append(f"{path.relative_to(ROOT)}: no rows")
                continue
            leaked = forbidden_keys(rows)
            if leaked:
                errors.append(
                    f"{path.relative_to(ROOT)}: forbidden text fields {sorted(leaked)}"
                )


def validate_figures(errors: list[str]) -> None:
    directory = ROOT / "paper/workshop_2026/figures"
    missing = EXPECTED_FIGURES.difference(path.name for path in directory.glob("*.png"))
    if missing:
        errors.append(f"missing paper figures: {', '.join(sorted(missing))}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild-figures",
        action="store_true",
        help="run the deterministic paper figure builder after validating inputs",
    )
    args = parser.parse_args()

    errors: list[str] = []
    validate_inputs(errors)
    validate_reports(errors)
    validate_public_labels(errors)
    validate_figures(errors)
    if errors:
        print("Artifact checks failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("Artifact checks passed.")
    if args.rebuild_figures:
        subprocess.run(
            [sys.executable, str(ROOT / "paper/workshop_2026/build_figures.py")],
            cwd=ROOT,
            check=True,
        )
        validate_errors: list[str] = []
        validate_figures(validate_errors)
        if validate_errors:
            for error in validate_errors:
                print(error, file=sys.stderr)
            return 1
        print(f"Rebuilt {len(EXPECTED_FIGURES)} paper figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
