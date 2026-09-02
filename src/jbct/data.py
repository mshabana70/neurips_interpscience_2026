"""Corpus loading for H1 (direction) and H2 (jailbreak viability).

H1 needs a genuinely *harmless* contrast set — Arditi's refusal_direction splits. Note this
is a different axis from H2's "control": an H2 control is the SAME harmful request under
neutral framing (does the attack framing carry the effect?), which is not a harmless prompt
and must never be substituted for one.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DIRECTION_CORPUS = REPO_ROOT / "data" / "refusal_direction"


def load_direction_corpus(
    split: str = "train", n: int | None = None, root: Path | None = None
) -> tuple[list[str], list[str]]:
    """``(harmful, harmless)`` instruction lists for diff-in-means.

    Selection is the first ``n`` of each file, matching Refusal-Lens' ``[:n_samples]``
    so a reconciliation run against its published separations is exact. Deterministic by
    construction — no shuffling, no seed to forget to record.
    """
    root = root or DIRECTION_CORPUS
    out = []
    for kind in ("harmful", "harmless"):
        recs = json.loads((root / f"{kind}_{split}.json").read_text())
        out.append([r["instruction"] for r in recs][:n])
    return out[0], out[1]


def load_attack_records(path: str | Path, model: str | None = None) -> list[dict]:
    """Attack records from the public jailbreak dataset JSON, filtered to one model.

    ``prefill`` records are model-SPECIFIC (one per model x base): running Gemma against
    Qwen's prefill string would silently measure the wrong thing, so records carrying a
    ``model`` field are dropped unless they match. Records with no ``model`` field are
    model-agnostic and always kept.
    """
    data = json.loads(Path(path).read_text())
    records = data["records"]
    if model is not None:
        records = [r for r in records if r.get("model") in (None, model)]
    return records


def resolve_judged(explicit, h2_root, model: str, repo_root: Path | None = None) -> Path:
    """Find H2's judged generations, preferring the committed gz archive.

    `judged.json` is matched by `.gitignore`'s `**/*judged.json`, so a fresh clone or a fresh
    pod has only the gzipped copy under `docs/runs/`. Every stage that consumed H2 read the
    plain path directly and so could only run on the machine that produced it. Reading the
    archive is also the better default: decompressing puts megabytes of by-construction-
    harmful completions back on disk in plain text, which is what the archive scheme exists
    to avoid.
    """
    if explicit:
        return Path(explicit)
    plain = Path(h2_root) / model / "judged.json"
    if plain.exists():
        return plain
    root = repo_root or REPO_ROOT
    archives = sorted(Path(root, "docs", "runs").glob(f"*/h2/{model}/judged.json.gz"))
    if archives:
        return archives[-1]
    raise SystemExit(
        f"no judged generations for {model}: looked at {plain} and "
        f"docs/runs/*/h2/{model}/judged.json.gz. Pass --judged explicitly."
    )


def write_json_gz(path, obj) -> Path:
    """Write ``<path>.gz``, compressed and byte-deterministic. Returns the path written.

    F1's `projections.json` doubled to ~5.7 MB (gemma) / ~6.0 MB (qwen) when it began
    carrying both bases, tripping the R23 size guard at 5 MB. `archive_results.PLAIN` had
    listed it as "small and worth reading directly on the forge" — true until it wasn't.
    gzip is **5.4x** on this content, which is headroom rather than a shave; rounding the
    floats would have bought 12% and left the next position or model to break it again.

    Determinism matters for the same reason it does in `scripts/archive_results.py`: `mtime`
    and the stored FNAME would otherwise make identical content produce differing bytes, so
    an unchanged rerun would show a diff that is not a result.
    """
    path = Path(path)
    dst = path if path.suffix == ".gz" else path.with_suffix(path.suffix + ".gz")
    dst.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=2).encode("utf-8")
    with open(dst, "wb") as fh, gzip.GzipFile(
        filename="", fileobj=fh, mode="wb", mtime=0
    ) as gz:
        gz.write(payload)
    return dst


def read_json_maybe_gz(path):
    """Read ``<path>.gz`` if present, else ``<path>``.

    The compressed copy wins deliberately: a stale plain file left over from a pre-gz run
    sitting next to a fresh archive would otherwise shadow it silently, which is this
    codebase's characteristic failure — a wrong number, not an error. Plain files stay
    readable so the pre-gz committed artifacts (`results/chain/*`, `docs/runs/*`) keep
    working.
    """
    path = Path(path)
    gz = path if path.suffix == ".gz" else path.with_suffix(path.suffix + ".gz")
    if gz.exists():
        with gzip.open(gz, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    if path.exists():
        return json.loads(path.read_text())
    raise FileNotFoundError(f"neither {gz} nor {path} exists")


def read_judged(path) -> list[dict]:
    """`generations` from a judged.json or judged.json.gz."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)["generations"]


def load_judged(model: str, h2_root="results/h2", explicit=None) -> tuple[Path, list[dict]]:
    """(path, generations) — the one call a consumer stage needs."""
    path = resolve_judged(explicit, h2_root, model)
    return path, read_judged(path)


def load_candidates(h1_root, model: str):
    """H1's direction grid, with an actionable message when it is absent.

    `candidates.pt` is matched by `.gitignore`'s `results/**/*.pt`, so it exists only on the
    machine that ran H1. Four stages load it and all four failed with a bare
    FileNotFoundError naming a path, which does not tell you that the fix is to re-run H1 —
    cheap, since the corpus it needs (`data/refusal_direction/`) IS committed.
    """
    import torch

    path = Path(h1_root) / model / "candidates.pt"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. It is gitignored (results/**/*.pt), so it lives only on the "
            f"machine that ran H1. Regenerate it here — the corpus is committed, so this is "
            f"~96 forward passes:\n"
            f"    uv run python experiments/h1_direction/run.py --model {model} "
            f"--dtype float32"
        )
    # map_location: H1 ran on GPU, so a saved tensor carries cuda:0 into every consumer.
    return torch.load(path, weights_only=True, map_location="cpu")
