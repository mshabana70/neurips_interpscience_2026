"""Figures for the foundational experiments. Matplotlib, PNG, deterministic.

Design constraints this module encodes, so callers cannot get them wrong:

**One validated categorical palette, assigned by entity and never cycled.** The five
population slots below pass every hard gate on the adjacent pairlist at light surface
#fcfcfb (worst CVD dE 9.1, worst normal-vision dE 19.6). Three of the five sit below 3:1
contrast on that surface, which obligates relief: every figure carries a legend, and every
number behind every figure is written to `f1_report.json`, which is the table view.

**The scatter is faceted, not overplotted.** No five-hue selection clears the ALL-pairs
floors that a scatter demands — the best candidate still failed normal-vision dE at 13.7
against a floor of 15 — and 764 points in five colours on one axis is unreadable anyway. So
`figure_pca` draws small multiples: one panel per population against a recessive backdrop of
every other point.

**No randomness.** Strip plots use a deterministic golden-ratio offset rather than a jitter
draw, so re-running produces byte-comparable figures. A figure that moves between runs makes
identical data look like two different results.
"""

from __future__ import annotations

import math
from pathlib import Path

# Categorical slots 1-5, fixed order, assigned to populations by identity.
POPULATION_COLORS = {
    "clean": "#2a78d6",           # blue
    "harmful": "#eb6834",         # orange
    "control": "#1baf7a",         # aqua
    "attack-COMPLY": "#eda100",   # yellow
    "attack-REFUSE": "#e87ba4",   # magenta
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e4e3de"

_GOLDEN = 0.6180339887498949


def _style(ax):
    """Recessive axes and grid — the data carries the ink, not the frame."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)


def _new_fig(figsize):
    import matplotlib

    matplotlib.use("Agg")  # headless: RunPod has no display
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize, dpi=160, facecolor=SURFACE)
    return plt, fig


def _finish(plt, fig, path: Path, title: str, subtitle: str | None = None):
    """Header placed in INCHES, not figure fractions.

    Fractional offsets collapse as the figure grows: the distributions figure is 3x the
    height of the profile, so a fixed `y=0.985/0.945` pair that clears at 5 inches has the
    subtitle overprinting the title at 9. Reserving a fixed physical header and converting
    to fractions per figure keeps the spacing identical at every size.
    """
    h = fig.get_figheight()
    header_in = 0.62 if subtitle else 0.40
    fig.tight_layout(rect=(0, 0, 1, 1 - header_in / h))
    fig.text(0.012, 1 - 0.24 / h, title, color=INK, fontsize=13, fontweight="bold",
             ha="left", va="center")
    if subtitle:
        fig.text(0.012, 1 - 0.50 / h, subtitle, color=INK_SECONDARY, fontsize=9.5,
                 ha="left", va="center")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] -> {path}")


def _series(per_site: dict, n_layers: int, pos: int, pop: str):
    """(layers, means, half_width_of_95pct_CI) for one population down the stack."""
    xs, ys, es = [], [], []
    for layer in range(n_layers):
        entry = per_site.get(f"{layer}@{pos}")
        if not entry or pop not in entry:
            continue
        s = entry[pop]
        if s["n"] == 0 or s["mean"] is None:
            continue
        xs.append(layer)
        ys.append(s["mean"])
        es.append(1.96 * s["sd"] / math.sqrt(s["n"]) if s["sd"] and s["n"] > 1 else 0.0)
    return xs, ys, es


def _layer_center(per_site, pos, layer) -> float | None:
    """Unweighted mean of the population means at this layer.

    Unweighted on purpose: 350 controls would otherwise outvote 32 clean prompts and the
    centre would track the control arm rather than sitting between the populations.
    """
    means = []
    for pop in POPULATION_COLORS:
        s = (per_site.get(f"{layer}@{pos}") or {}).get(pop)
        if s and s.get("mean") is not None and s["n"]:
            means.append(s["mean"])
    return sum(means) / len(means) if means else None


def figure_layer_profile(per_site, n_layers, pos, path, title):
    """Mean projection vs depth, one line per population, CENTRED per layer.

    `proj = resid @ r-hat` is uncentred: every prompt carries a large shared component along
    r-hat — the same reason the H1xH2 chain's projections all sat near -250 to -360 rather
    than straddling zero. Raw, that shared trajectory reaches +-40,000 and the between-
    population differences of a few hundred are invisible. Subtracting the per-layer centre
    makes the y-axis 'distance from the population average', which is the quantity the
    question asks about. The RAW means are in `f1_report.json`; nothing is lost.
    """
    plt, fig = _new_fig((9, 5))
    ax = fig.add_subplot(111)
    _style(ax)
    centers = {layer: _layer_center(per_site, pos, layer) for layer in range(n_layers)}
    ax.axhline(0, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    for pop, color in POPULATION_COLORS.items():
        xs, ys, es = _series(per_site, n_layers, pos, pop)
        if not xs:
            continue
        ys = [y - (centers.get(x) or 0.0) for x, y in zip(xs, ys)]
        lo = [y - e for y, e in zip(ys, es)]
        hi = [y + e for y, e in zip(ys, es)]
        ax.fill_between(xs, lo, hi, color=color, alpha=0.16, linewidth=0, zorder=2)
        ax.plot(xs, ys, color=color, linewidth=2.0, label=pop, zorder=3)
    ax.set_xlabel("layer", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel(f"projection minus layer mean (position {pos})",
                  color=INK_SECONDARY, fontsize=10)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="best")
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)
    _finish(plt, fig, path, title,
            "centred per layer (raw means in f1_report.json); band = 95% CI. "
            "Direction fit on H1 train; every point held out.")


def figure_distributions(rows, site_keys, path, title):
    """Per-population strips at the called-out sites — overlap, not just means.

    Strips rather than violins because n ranges from 32 to 350; a KDE over 32 points invents
    a shape the data does not support.
    """
    plt, fig = _new_fig((9, 3.1 * max(1, len(site_keys))))
    pops = list(POPULATION_COLORS)
    for panel, key in enumerate(site_keys):
        ax = fig.add_subplot(len(site_keys), 1, panel + 1)
        _style(ax)
        labels = []
        for i, pop in enumerate(pops):
            vals = [r["proj"][key] for r in rows
                    if r["population"] == pop and key in r.get("proj", {})]
            if not vals:
                labels.append(pop)
                continue
            color = POPULATION_COLORS[pop]
            # Deterministic spread inside the row: golden-ratio walk, never a random draw.
            ys = [i + ((j * _GOLDEN) % 1.0 - 0.5) * 0.55 for j in range(len(vals))]
            ax.scatter(vals, ys, s=9, color=color, alpha=0.45, linewidths=0, zorder=2)
            mean = sum(vals) / len(vals)
            # 2px surface ring so the rule stays legible where it crosses dense points.
            ax.plot([mean, mean], [i - 0.36, i + 0.36], color=SURFACE, linewidth=4.4, zorder=3)
            ax.plot([mean, mean], [i - 0.36, i + 0.36], color=color, linewidth=2.4, zorder=4)
            # The mean rides in the row label, not beside the rule: a value printed at the
            # mean sits exactly where the rule is and the two overprint.
            labels.append(f"{pop}\nmean {mean:,.0f}  n={len(vals)}")
        ax.set_yticks(range(len(pops)))
        ax.set_yticklabels(labels, color=INK_SECONDARY, fontsize=9)
        ax.set_ylim(-0.7, len(pops) - 0.3)
        ax.set_xlabel(f"projection at {key}", color=INK_SECONDARY, fontsize=10)
        ax.invert_yaxis()
    _finish(plt, fig, path, title, "one dot per prompt; the vertical rule is the mean")


def figure_pca(coords, populations, explained, path, title):
    """Small multiples: each population against a recessive backdrop of every other point.

    Faceted rather than overplotted because a five-hue categorical set cannot clear the
    all-pairs colour floors a scatter requires, and 764 points in five colours would be
    unreadable even if it could.
    """
    plt, fig = _new_fig((12, 2.9))
    pops = list(POPULATION_COLORS)
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]

    def _lim(vs):
        """Never hand matplotlib a zero-width range — it warns and expands arbitrarily."""
        lo, hi = min(vs), max(vs)
        if lo == hi:
            pad = abs(lo) * 0.05 or 1.0
            return lo - pad, hi + pad
        return lo, hi

    xlim, ylim = _lim(xs), _lim(ys)
    for i, pop in enumerate(pops):
        ax = fig.add_subplot(1, len(pops), i + 1)
        _style(ax)
        ax.scatter(xs, ys, s=6, color=GRID, linewidths=0, zorder=1)
        sel = [(x, y) for x, y, p in zip(xs, ys, populations) if p == pop]
        if sel:
            ax.scatter([p[0] for p in sel], [p[1] for p in sel], s=11,
                       color=POPULATION_COLORS[pop], linewidths=0, alpha=0.75, zorder=3)
        ax.set_title(f"{pop}  (n={len(sel)})", color=INK, fontsize=9.5, pad=6)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if i == 0:
            ax.set_xlabel(f"PC1 ({explained[0] * 100:.0f}%)", color=INK_SECONDARY, fontsize=9)
            ax.set_ylabel(f"PC2 ({explained[1] * 100:.0f}%)", color=INK_SECONDARY, fontsize=9)
    _finish(plt, fig, path, title,
            "grey = every prompt; colour = this population. Shared axes across panels.")


def figure_sep(per_site, n_layers, pos, path, title):
    """Two AUCs on one axis — both are AUC, so this is one scale, never a dual axis.

    The point of putting them together: the direction can separate harmful from clean
    (upper line) at a depth where it carries no information about whether a jailbreak
    actually worked (lower line). 0.5 is chance, and it is drawn.
    """
    plt, fig = _new_fig((9, 4.6))
    ax = fig.add_subplot(111)
    _style(ax)
    ax.axhline(0.5, color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.text(0.2, 0.505, "chance", color=INK_MUTED, fontsize=8.5, va="bottom")
    for metric, color, label in (
        ("auc_harmful_over_clean", POPULATION_COLORS["harmful"], "harmful vs clean"),
        ("auc_refuse_over_comply", POPULATION_COLORS["attack-REFUSE"],
         "REFUSE vs COMPLY"),
    ):
        xs, ys = [], []
        for layer in range(n_layers):
            entry = per_site.get(f"{layer}@{pos}") or {}
            if entry.get(metric) is not None:
                xs.append(layer)
                ys.append(entry[metric])
        if not xs:
            continue
        ax.plot(xs, ys, color=color, linewidth=2.0, zorder=3)
        ax.scatter(xs[-1:], ys[-1:], s=26, color=color, linewidths=0, zorder=4)
        # Two series, so direct labels rather than a legend box.
        ax.text(xs[-1] + 0.4, ys[-1], label, color=color, fontsize=9.5, va="center")
    ax.set_xlabel("layer", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("AUC", color=INK_SECONDARY, fontsize=10)
    ax.set_ylim(0, 1)
    # Room on the right for the direct labels; without it the longer one runs off the edge.
    ax.set_xlim(-0.5, n_layers + 13)
    _finish(plt, fig, path, title,
            "above 0.5 = the direction ranks the first group higher; 0.5 = no information")


# ── H3 causal / F2 ────────────────────────────────────────────────────────────
# Two series only, and they are a matched pair: the real direction against a random vector
# of identical norm. Categorical slots 1 and 2, which clear the all-pairs colour floors.
REAL = "#2a78d6"
SHAM = "#eb6834"
# A dose that leaves this fraction of responses incoherent is not an intervention any more,
# it is damage — and a judge scoring gibberish returns a number that looks like a result.
INCOHERENT_INVALID = 0.20


def _arm_series(arms: dict, prefix: str, coeffs: list[float]):
    """(comply_after, incoherent_fraction) per coefficient for one addition arm."""
    after, incoh = [], []
    for c in coeffs:
        v = arms.get(f"{prefix}@c{c:g}") or {}
        n = v.get("n") or 0
        after.append(v.get("overall_comply_after") if n else None)
        incoh.append((v.get("incoherent", 0) / n) if n else None)
    return after, incoh


def figure_dose_response(sites: list[dict], path, title):
    """Comply rate vs dose, real direction against its matched sham, one panel per site.

    The sham line is the whole point. Without it a falling real curve reads as "the refusal
    direction suppresses jailbreaks"; with it you can see how much of the fall any vector of
    that norm would have produced, and at which dose the two converge.
    """
    plt, fig = _new_fig((10, 7))
    for i, s in enumerate(sites):
        ax = fig.add_subplot(2, 2, i + 1)
        _style(ax)
        coeffs = s["coeffs"]
        xs = list(range(len(coeffs)))
        real, real_inc = _arm_series(s["arms"], "add_attack", coeffs)
        sham, sham_inc = _arm_series(s["arms"], "sham_attack", coeffs)

        # Shade the doses where either arm has crossed into damage.
        for x, ri, si in zip(xs, real_inc, sham_inc):
            if max(ri or 0, si or 0) >= INCOHERENT_INVALID:
                ax.axvspan(x - 0.5, x + 0.5, color=INK_MUTED, alpha=0.12, linewidth=0, zorder=0)
        ax.axhline(1.0, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        ends = []
        for vals, inc, color, label in ((real, real_inc, REAL, "refusal direction"),
                                        (sham, sham_inc, SHAM, "random, same norm")):
            pts = [(x, v) for x, v in zip(xs, vals) if v is not None]
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linewidth=2.0,
                    zorder=3)
            for x, v in pts:
                damaged = (inc[x] or 0) >= INCOHERENT_INVALID
                ax.scatter([x], [v], s=42, zorder=4, linewidths=1.6,
                           facecolors=SURFACE if damaged else color, edgecolors=color)
            ends.append([pts[-1][0], pts[-1][1], color, label])
        # The two arms converge to the same value at high dose — that IS the finding — so
        # their end labels land on top of each other unless they are pushed apart.
        if len(ends) == 2 and abs(ends[0][1] - ends[1][1]) < 0.09:
            mid = (ends[0][1] + ends[1][1]) / 2
            order = sorted(ends, key=lambda e: e[1])
            order[0][1], order[1][1] = mid - 0.055, mid + 0.055
        for x, y, color, label in ends:
            ax.text(x + 0.12, y, label, color=color, fontsize=8.5, va="center", ha="left")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{c:g}" for c in coeffs])
        ax.set_xlim(-0.5, len(coeffs) - 0.5 + 1.5)
        ax.set_ylim(-0.05, 1.12)
        ax.set_title(f"{s['model']}  L{s['site']}", color=INK, fontsize=10, pad=6)
        if i >= 2:
            ax.set_xlabel("coefficient  (multiples of ‖r‖)", color=INK_SECONDARY, fontsize=9.5)
        if i % 2 == 0:
            ax.set_ylabel("compliance after", color=INK_SECONDARY, fontsize=9.5)
    _finish(plt, fig, path, title,
            "dashed line = compliance before (every prompt complied by construction). "
            f"Hollow markers and shading: ≥{INCOHERENT_INVALID:.0%} of responses incoherent.")


def figure_coherence(sites: list[dict], path, title):
    """Fraction of incoherent responses vs dose. Where this rises, the arm stops measuring."""
    plt, fig = _new_fig((10, 6))
    for i, s in enumerate(sites):
        ax = fig.add_subplot(2, 2, i + 1)
        _style(ax)
        coeffs = s["coeffs"]
        xs = list(range(len(coeffs)))
        ax.axhline(INCOHERENT_INVALID, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)),
                   zorder=1)
        for prefix, color, label in (("add_attack", REAL, "refusal direction"),
                                     ("sham_attack", SHAM, "random, same norm")):
            _after, inc = _arm_series(s["arms"], prefix, coeffs)
            pts = [(x, v) for x, v in zip(xs, inc) if v is not None]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linewidth=2.0,
                        marker="o", markersize=5, zorder=3)
                ax.text(pts[-1][0] + 0.12, pts[-1][1], label, color=color, fontsize=8.5,
                        va="center")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{c:g}" for c in coeffs])
        ax.set_xlim(-0.5, len(coeffs) - 0.5 + 1.5)
        ax.set_ylim(-0.05, 1.08)
        ax.set_title(f"{s['model']}  L{s['site']}", color=INK, fontsize=10, pad=6)
        if i >= 2:
            ax.set_xlabel("coefficient  (multiples of ‖r‖)", color=INK_SECONDARY, fontsize=9.5)
        if i % 2 == 0:
            ax.set_ylabel("fraction incoherent", color=INK_SECONDARY, fontsize=9.5)
    _finish(plt, fig, path, title,
            f"dashed line = the {INCOHERENT_INVALID:.0%} threshold above which a dose is "
            "damaging the model rather than steering it")


def figure_ablation(sites: list[dict], path, title):
    """Removing the direction from control prompts: does refusal break, and specifically?

    The mirror of the addition arm. Drawn as a slope so the before/after pair reads as one
    movement rather than two bars that have to be mentally subtracted.
    """
    plt, fig = _new_fig((9, 4.4))
    ax = fig.add_subplot(111)
    _style(ax)
    rows, labels = [], []
    for s in sites:
        for prefix, color, tag in (("ablate_control", REAL, "ablate r̂"),
                                   ("sham_control", SHAM, "ablate random")):
            v = s["arms"].get(prefix) or {}
            if not v.get("n"):
                continue
            rows.append((v["overall_comply_before"], v["overall_comply_after"], color))
            labels.append(f"{s['model'].split('-')[0]} L{s['site']}  {tag}")
    ys = list(range(len(rows)))
    for y, (before, after, color) in zip(ys, rows):
        ax.plot([before, after], [y, y], color=color, linewidth=2.0, zorder=2,
                solid_capstyle="round")
        ax.scatter([before], [y], s=30, facecolors=SURFACE, edgecolors=color, linewidths=1.6,
                   zorder=3)
        ax.scatter([after], [y], s=46, color=color, linewidths=0, zorder=4)
        ax.text(max(before, after) + 0.02, y, f"{after - before:+.3f}", color=INK_SECONDARY,
                fontsize=8.5, va="center")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 0.85)
    ax.set_xlabel("compliance on control prompts", color=INK_SECONDARY, fontsize=10)
    _finish(plt, fig, path, title,
            "hollow = before, filled = after. If refusal is mediated by the direction, "
            "removing it should raise compliance — and the random arm should not.")
