"""Environment provenance stamped into every experiment report.

Reports already carry `dtype`, the read position, and BOS count — the things that make two
directions incomparable *within* an environment. They did not carry the library versions,
which is what makes two results incomparable *across* one.

That gap became concrete when circuit-tracer's hard pin forced transformers 5.14.1 -> 4.57.3
(see docs/reorg/2026-07-28-transformers-downgrade-for-h4.md). That particular change turned
out to be numerically transparent — Gemma's H1 separations reproduced bit-identically across
all 34 layers — but we only know that because the old numbers were still on disk to compare
against. The next such change might not be transparent, and nothing in the reports would have
said so.

Cheap to record, impossible to reconstruct after the fact.
"""

from __future__ import annotations

import platform
import sys


def env_versions() -> dict:
    """Library versions that can move a number. Absent imports report None, never crash."""
    out: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for name, attr in (
        ("torch", "__version__"),
        ("transformers", "__version__"),
        ("transformer_lens", "__version__"),
        ("nnsight", "__version__"),
        ("huggingface_hub", "__version__"),
    ):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, attr, None)
        except Exception:  # noqa: BLE001 - a missing optional dep is data, not an error
            out[name] = None
    try:
        import torch

        out["cuda"] = torch.version.cuda
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        out["cuda"] = out["gpu"] = None
    return out
