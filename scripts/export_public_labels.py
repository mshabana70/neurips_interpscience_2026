"""Export label-level paper results without prompts or model response text.

The full experiment outputs contain harmful prompts and completions. This utility keeps
only identifiers, condition names, behavioral labels, coherence flags, and token counts.
It is useful after a fresh full run and was also used to prepare the anonymous artifact.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


MODELS = ("gemma-3-4b", "qwen3-4b")
SAFE_KEYS = (
    "pair_id",
    "record_idx",
    "base_id",
    "class",
    "kind",
    "historical_judge",
    "baseline_judge",
    "judge",
    "llm_refusal",
    "lexical_refusal",
    "coherent",
    "incoherent",
    "n_prompt_tokens",
    "n_new_tokens",
    "hit_token_cap",
)
FORBIDDEN_KEYS = {"prompt", "text", "prefill", "response"}


def read_json(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_gz(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8")
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            compressed.write(payload)


def safe_row(row: dict, *, source: str, arm: str, index: int, stage: str | None = None):
    output = {key: row[key] for key in SAFE_KEYS if key in row}
    output.update({"source": source, "arm": arm, "row_index": index})
    if stage is not None:
        output["stage"] = stage
    assert not FORBIDDEN_KEYS.intersection(output)
    return output


def export_arditi(source_root: Path, output_root: Path, model: str) -> int:
    rows = []
    directory = source_root / "results" / "arditi_gap" / model / "generations"
    for path in sorted(directory.glob("*.json.gz")):
        for index, row in enumerate(read_json(path)):
            rows.append(safe_row(row, source="arditi_gap", arm=path.name[:-8], index=index))
    write_json_gz(output_root / "arditi_gap" / f"{model}.json.gz", rows)
    return len(rows)


def export_h3(source_root: Path, output_root: Path, model: str) -> int:
    rows = []
    directory = source_root / "results" / "h3_causal" / model
    for path in sorted(directory.glob("*generations.json")):
        for index, row in enumerate(read_json(path)):
            rows.append(safe_row(row, source="h3_causal", arm=path.stem, index=index))
    write_json_gz(output_root / "h3_causal" / f"{model}.json.gz", rows)
    return len(rows)


def export_closure(source_root: Path, output_root: Path, model: str) -> int:
    rows = []
    directory = source_root / "results" / "paper_closure" / model
    for stage in ("scope", "transplant", "restoration"):
        for path in sorted((directory / stage).glob("*.json.gz")):
            for index, row in enumerate(read_json(path)):
                rows.append(
                    safe_row(
                        row,
                        source="paper_closure",
                        stage=stage,
                        arm=path.name[:-8],
                        index=index,
                    )
                )
    write_json_gz(output_root / "paper_closure" / f"{model}.json.gz", rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/public_labels")
    )
    args = parser.parse_args()
    for model in MODELS:
        counts = {
            "arditi_gap": export_arditi(args.source_root, args.output_root, model),
            "h3_causal": export_h3(args.source_root, args.output_root, model),
            "paper_closure": export_closure(args.source_root, args.output_root, model),
        }
        print(model, counts)


if __name__ == "__main__":
    main()
