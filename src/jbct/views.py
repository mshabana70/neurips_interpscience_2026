"""Saved dashboard views: pinned nodes, supernodes, label overrides, thresholds.

circuit-tracer's bundled frontend reads `pinnedIds`, `supernodes`, `clerps` and
`pruningThreshold` straight from the URL (same scheme Neuronpedia uses), so a curated subgraph
is pure client state — no compute, and nothing server-side to build. The gap is that it lives
only in the address bar: close the tab or rebuild the graphs and the curation is gone.

This makes a view a FILE. That matters for more than convenience:

- A supernode is a **hypothesis** ("these features are the refusal backbone"), and P2 tests it
  by steering exactly those features. The grouping therefore has to be a durable, diffable
  artifact that a steering run can read — not a URL someone pasted into Slack.
- Groupings are human judgement over labels that are currently UNSCORED, so a view records
  which graph and which label scheme it was made against. A grouping curated over one set of
  labels is not automatically valid over another.

Node ids are `{layer}_{feature}_{position}`, matching `create_graph_files`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

DEFAULT_BASE = "http://localhost:8032"


@dataclass
class GraphView:
    slug: str
    pinned: list[str] = field(default_factory=list)
    # [[label, node_id, node_id, ...], ...] — label first, matching the frontend's encoding
    supernodes: list[list[str]] = field(default_factory=list)
    clerps: list[list[str]] = field(default_factory=list)  # [[node_id, label], ...]
    pruning_threshold: float | None = None
    density_threshold: float | None = None
    note: str = ""
    meta: dict = field(default_factory=dict)

    def node_ids(self) -> list[str]:
        """Every node referenced, pinned or grouped — what a steering run acts on."""
        out = list(self.pinned)
        for group in self.supernodes:
            out.extend(group[1:])
        seen, unique = set(), []
        for n in out:
            if n not in seen:
                seen.add(n)
                unique.append(n)
        return unique

    def group_of(self, node_id: str) -> str | None:
        for group in self.supernodes:
            if node_id in group[1:]:
                return group[0]
        return None


def parse_node_id(node_id: str) -> tuple[int, int, int] | None:
    """`"15_8823_36"` -> (layer, feature, position). None if it is not a feature node."""
    parts = node_id.split("_")
    if len(parts) != 3 or not all(p.lstrip("-").isdigit() for p in parts):
        return None
    layer, feature, pos = (int(p) for p in parts)
    return layer, feature, pos


def parse_view_url(url: str) -> GraphView:
    """Reconstruct a view from a URL pasted out of the browser."""
    q = parse_qs(urlparse(url).query)

    def one(key, default=None):
        vals = q.get(key)
        return vals[0] if vals else default

    def as_json(key, default):
        raw = one(key)
        if not raw:
            return default
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return default

    def as_float(key):
        raw = one(key)
        try:
            return float(raw) if raw is not None else None
        except ValueError:
            return None

    pinned_raw = one("pinnedIds", "") or ""
    return GraphView(
        slug=one("slug", "") or "",
        pinned=[p for p in pinned_raw.split(",") if p],
        supernodes=[list(map(str, g)) for g in as_json("supernodes", [])],
        clerps=[list(map(str, c)) for c in as_json("clerps", [])],
        pruning_threshold=as_float("pruningThreshold"),
        density_threshold=as_float("densityThreshold"),
    )


def build_view_url(view: GraphView, base: str = DEFAULT_BASE) -> str:
    """A URL that reopens this view. `urlencode` handles the JSON/comma escaping."""
    params: dict = {"slug": view.slug}
    if view.pinned:
        params["pinnedIds"] = ",".join(view.pinned)
    if view.supernodes:
        params["supernodes"] = json.dumps(view.supernodes, separators=(",", ":"))
    if view.clerps:
        params["clerps"] = json.dumps(view.clerps, separators=(",", ":"))
    if view.pruning_threshold is not None:
        params["pruningThreshold"] = view.pruning_threshold
    if view.density_threshold is not None:
        params["densityThreshold"] = view.density_threshold
    return f"{base.rstrip('/')}/?{urlencode(params)}"


def apply_view_to_graph(graph: dict, view: GraphView) -> dict:
    """Bake a view into the graph JSON's `qParams`. Returns a drop report.

    This, not the URL, is the durable channel. `init-cg.js` seeds its visState from
    `data.qParams` before it ever looks at the query string, so a view written here loads
    every time the slug is opened. A URL only survives until the next click: every
    `util.params.set` calls `history.replaceState`, so picking another graph in the dashboard
    silently rewrites the query string and the curation is gone.

    Unknown node ids are DROPPED and reported rather than passed through. The frontend
    renders the subgraph as `nodes.filter(d => pinnedIds.includes(d.nodeId))` — an id that
    is not in this graph contributes nothing and raises no error, so a view pointing at the
    wrong slug looks exactly like a view that failed to load.
    """
    known = {n.get("node_id") for n in graph.get("nodes", [])}
    kept_pinned = [n for n in view.pinned if n in known]
    dropped = [n for n in view.pinned if n not in known]

    supernodes, groups = [], {}
    for group in view.supernodes:
        label, members = group[0], group[1:]
        kept = [m for m in members if m in known]
        groups[label] = {"kept": len(kept), "total": len(members)}
        if kept:  # an empty supernode renders as a bare label; omit it
            supernodes.append([label, *kept])
        dropped.extend(m for m in members if m not in known and m not in dropped)

    q = dict(graph.get("qParams") or {})
    q["pinnedIds"] = kept_pinned
    q["supernodes"] = supernodes
    q.setdefault("linkType", "both")
    q.setdefault("clickedId", "")
    q.setdefault("sg_pos", "")  # empty => the panel force-lays-out instead of restoring
    if view.clerps:
        q["clerps"] = json.dumps(view.clerps, separators=(",", ":"))
    if view.pruning_threshold is not None:
        q["pruningThreshold"] = view.pruning_threshold
    graph["qParams"] = q
    return {"pinned": len(kept_pinned), "dropped": sorted(dropped), "groups": groups}


def save_view(view: GraphView, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "slug": view.slug,
        "pinned": view.pinned,
        "supernodes": view.supernodes,
        "clerps": view.clerps,
        "pruning_threshold": view.pruning_threshold,
        "density_threshold": view.density_threshold,
        "note": view.note,
        # Which graph and label scheme this grouping was curated against. A supernode is a
        # hypothesis about specific features; it does not transfer to a regenerated graph
        # whose node set differs.
        "meta": view.meta,
    }, indent=2))
    return path


def load_view(path: str | Path) -> GraphView:
    d = json.loads(Path(path).read_text())
    return GraphView(
        slug=d.get("slug", ""),
        pinned=list(d.get("pinned", [])),
        supernodes=[list(g) for g in d.get("supernodes", [])],
        clerps=[list(c) for c in d.get("clerps", [])],
        pruning_threshold=d.get("pruning_threshold"),
        density_threshold=d.get("density_threshold"),
        note=d.get("note", ""),
        meta=d.get("meta", {}),
    )
