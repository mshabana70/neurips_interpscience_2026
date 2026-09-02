"""Feature labels — the gap Refusal-Lens' dashboards never filled.

The transcoder repos ship per-feature detail (`features/index.json.gz` + `features/layer_N.bin`):
top-activating **examples** (tokens + per-token activations) and **top/bottom logits**. What they
do NOT ship is any human-readable label, so a graph node reads `L15/F8823` and the dashboard
tells you nothing about what it does. That is exactly why the old dashboards were hard to read.

Two labelling paths, deliberately layered:

- `heuristic_label` — deterministic, offline, free. Built from the tokens the feature fires
  hardest on plus the logits it promotes. Always available, never wrong in an interesting way,
  and reproducible: no model, no seed, no API.
- `llm_label` — a short description from an LLM shown the same evidence. Better prose, but it
  costs money, varies run to run, and can hallucinate. So it is opt-in, cached on disk, and
  **stored alongside the heuristic rather than replacing it**, with the evidence it was derived
  from — a label nobody can audit is worse than no label.

Only features that actually appear in a graph need labelling (a few thousand at most, versus
~131k per layer), so labelling is driven by the graph, not by the transcoder.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

MAX_LABEL_TOKENS = 6

# Bump when `heuristic_label`'s OUTPUT changes. Cached labels carry this, and a mismatch is
# treated as a cache miss — otherwise a scheme change leaves old labels in place forever,
# which is how a rebuilt dashboard ended up showing the retired `-> promotes: "ный"` format
# on some nodes and nothing on others.
HEURISTIC_VERSION = 2


def cantor_unpair(z: int) -> tuple[int, int]:
    """Invert circuit-tracer's `cantor_pairing(layer, feat_idx)`.

    A graph node stores `feature = (layer + feat) * (layer + feat + 1) // 2 + feat`, NOT the
    raw feature index — so looking a node's `feature` up directly in the transcoder's offset
    table silently misses every time (the paired value is far past the 16384 offsets, and the
    lookup just returns None). The frontend unpairs it in JS; this is the Python equivalent.
    """
    import math

    w = int((math.isqrt(8 * z + 1) - 1) // 2)
    t = w * (w + 1) // 2
    y = z - t
    return w - y, y


def decode_record(raw: bytes) -> dict:
    """Decode one feature record from a `layer_N.bin` span.

    Layout (verified against the real bytes, not inferred):

        fc 60 00 00 | 1f 8b 08 08 .. | "56.json\0" | <deflate stream>
        4-byte LE     gzip member with the FNAME flag set
        length

    The 4-byte length prefix is why `gzip.decompress(raw)` fails: the buffer does not start
    with the gzip magic. Raises with the leading bytes on failure — an earlier version
    returned None here, and that silent fallback disguised a decode bug as "this feature has
    no evidence" for every feature in every graph.
    """
    import struct

    if len(raw) > 6 and raw[4:6] == b"\x1f\x8b":
        (n,) = struct.unpack("<I", raw[:4])
        body = raw[4 : 4 + n] if 0 < n <= len(raw) - 4 else raw[4:]
        return json.loads(gzip.decompress(body))
    if raw[:2] == b"\x1f\x8b":  # bare gzip, no prefix
        return json.loads(gzip.decompress(raw))
    if raw[:1] in (b"{", b"["):  # plain JSON
        return json.loads(raw)
    raise ValueError(f"unrecognized feature-record framing; first bytes: {raw[:12].hex()}")


@dataclass
class FeatureRecord:
    """One transcoder feature's evidence, as stored in the repo."""

    layer: int
    index: int
    top_tokens: list[str] = field(default_factory=list)  # tokens at peak activation
    top_logits: list[str] = field(default_factory=list)
    bottom_logits: list[str] = field(default_factory=list)
    # Full activating examples: (str_tokens, per-token activations). Kept because a bag of
    # peak tokens loses the CONTEXT a feature fires in, which is most of the signal when
    # asking a model what the feature detects.
    examples: list[tuple[list[str], list[float]]] = field(default_factory=list)


def activating_examples(record: dict, max_examples: int = 20) -> list[tuple[list[str], list[float]]]:
    """(str_tokens, activations) per stored example, highest quantile first."""
    out = []
    for quantile in record.get("examples_quantiles", []):
        for ex in quantile.get("examples", []):
            acts, toks = ex.get("tokens_acts_list") or [], ex.get("tokens") or []
            if not acts or not toks:
                continue
            n = min(len(acts), len(toks))
            out.append((list(toks[:n]), [float(a) for a in acts[:n]]))
            if len(out) >= max_examples:
                return out
    return out


def peak_tokens(record: dict, max_examples: int = 32) -> list[str]:
    """The token at each example's ACTIVATION PEAK, not its centre.

    `train_token_ind` marks where the feature fired during training, but examples are stored as
    windows; taking the peak of `tokens_acts_list` describes what the feature responds to rather
    than where the window happens to be centred.
    """
    out = []
    for quantile in record.get("examples_quantiles", []):
        for ex in quantile.get("examples", [])[:max_examples]:
            acts, toks = ex.get("tokens_acts_list") or [], ex.get("tokens") or []
            if not acts or not toks:
                continue
            i = max(range(min(len(acts), len(toks))), key=lambda j: acts[j])
            out.append(toks[i])
    return out


# Tokens that carry no information about what a feature detects.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "that", "this", "for",
    "on", "with", "as", "at", "by", "be", "are", "was", "i", "you", "s", "t", "re", "m",
}


def context_tokens(record: FeatureRecord, window: int = 6, top: int = 4) -> list[str]:
    """Frequent content tokens NEAR each activation peak.

    Peak tokens alone are often uninformative — `","`, `"to"`, `"you"` — while the
    surrounding words carry the meaning ("let's be honest,", "I'm here to listen").
    Deterministic and offline, so this stays free.
    """
    counts: Counter = Counter()
    for tokens, acts in record.examples:
        if not acts:
            continue
        peak = max(range(len(acts)), key=lambda i: acts[i])
        lo, hi = max(0, peak - window), min(len(tokens), peak + window + 1)
        for t in tokens[lo:hi]:
            w = t.strip().strip(".,:;!?*#-—'\"()[]").lower()
            if len(w) > 2 and w not in _STOPWORDS and w.isalpha():
                counts[w] += 1
    return [w for w, _ in counts.most_common(top)]


def heuristic_label(record: FeatureRecord, include_logits: bool = False) -> str:
    """A deterministic label from the feature's own evidence.

    Reports the tokens the feature fires on plus the words that surround those firings.

    `top_logits` are OFF by default. A mid-layer feature's direct logit attribution is
    mediated through every later layer, so its "promotes" list is usually multilingual noise
    — measured on real Gemma L3-L7 features, it returned things like "eerder", "mulher", "ᴛ",
    "polytopes". Putting that in a label is worse than omitting it, because it reads as a
    finding. Enable only for late-layer features where the logit path is short.
    """
    fires = [t for t, _ in Counter(t.strip() for t in record.top_tokens if t.strip()).most_common(3)]
    parts = []
    if fires:
        parts.append("fires on: " + ", ".join(f'"{t}"' for t in fires))
    ctx = context_tokens(record)
    if ctx:
        parts.append("near: " + ", ".join(ctx))
    if include_logits:
        promotes = [t.strip() for t in record.top_logits[:3] if t.strip()]
        if promotes:
            parts.append("promotes: " + ", ".join(f'"{t}"' for t in promotes))
    return " | ".join(parts) if parts else f"L{record.layer}/F{record.index} (no evidence)"


class FeatureStore:
    """Random access into a transcoder repo's feature detail, by (layer, index).

    The `.bin` files are hundreds of MB per layer, so records are seeked-to via the offset
    index and never loaded wholesale.
    """

    def __init__(self, index: dict, resolve_bin):
        self._index = index
        self._resolve_bin = resolve_bin  # (filename) -> local path
        self._cache: dict[tuple[int, int], FeatureRecord] = {}
        self.decode_failures = 0

    @classmethod
    def from_hub(cls, hf_ref: str, subfolder: str | None = None):
        from huggingface_hub import hf_hub_download

        from .transcoders import parse_hf_ref

        repo_id, file_path, revision = parse_hf_ref(hf_ref)
        base = "/".join(p for p in (file_path, subfolder) if p) or None
        idx_name = f"{base}/features/index.json.gz" if base else "features/index.json.gz"
        with gzip.open(hf_hub_download(repo_id, idx_name, revision=revision)) as fh:
            index = json.load(fh)

        def resolve(filename: str) -> str:
            name = f"{base}/features/{filename}" if base else f"features/{filename}"
            return hf_hub_download(repo_id, name, revision=revision)

        return cls(index, resolve)

    def get(self, layer: int, index: int) -> FeatureRecord | None:
        key = (layer, index)
        if key in self._cache:
            return self._cache[key]
        entry = self._index.get(str(layer))
        if not entry:
            return None
        offsets = entry["offsets"]
        if index + 1 >= len(offsets):
            return None
        start, end = offsets[index], offsets[index + 1]
        if end <= start:  # zero-length span = a dead feature with no stored examples
            return None
        with open(self._resolve_bin(entry["filename"]), "rb") as fh:
            fh.seek(start)
            raw = fh.read(end - start)
        try:
            payload = decode_record(raw)
        except Exception as e:  # noqa: BLE001
            # Counted, not swallowed. A decode bug must not masquerade as missing data.
            self.decode_failures += 1
            if self.decode_failures <= 3:
                print(f"[features] decode failed L{layer}/F{index}: {type(e).__name__}: {e}")
            return None
        rec = FeatureRecord(
            layer=layer,
            index=index,
            top_tokens=peak_tokens(payload),
            top_logits=list(payload.get("top_logits") or []),
            bottom_logits=list(payload.get("bottom_logits") or []),
            examples=activating_examples(payload),
        )
        self._cache[key] = rec
        return rec


class LabelCache:
    """On-disk labels keyed by "layer/index", so a graph is labelled once, not per view."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data: dict = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except Exception:  # noqa: BLE001
                self.data = {}

    def key(self, layer: int, index: int) -> str:
        return f"{layer}/{index}"

    def get(self, layer: int, index: int) -> dict | None:
        return self.data.get(self.key(layer, index))

    def put(self, layer: int, index: int, **fields) -> None:
        self.data.setdefault(self.key(layer, index), {}).update(fields)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1, sort_keys=True))


def llm_label_prompt(record: FeatureRecord) -> str:
    """The evidence an LLM is shown. Stored with the label so it can be audited."""
    fires = ", ".join(f'"{t}"' for t, _ in Counter(record.top_tokens).most_common(10))
    promotes = ", ".join(f'"{t}"' for t in record.top_logits[:10])
    return (
        "A sparse transcoder feature in a language model. Describe what it detects in at most "
        f"{MAX_LABEL_TOKENS} words. Reply with the description only.\n\n"
        f"Tokens it activates most on: {fires or '(none)'}\n"
        f"Tokens it promotes in the output: {promotes or '(none)'}"
    )
