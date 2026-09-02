"""Figures for the per-attack-class geometry.

**Seven classes will not fit the categorical palette, so none of these encode class by hue.**
`plots.POPULATION_COLORS` is validated for five slots on the light surface, and the module
that defines it already records that no five-hue set clears the all-pairs floors a scatter
demands. Seven would be worse. Every figure here therefore separates classes by POSITION —
small multiples with a recessive backdrop, or a category axis — which is the same answer
`figure_pca` reached for its five populations.

**The heatmaps run from the harmful hue to the clean hue.** That is not decoration: the
quantity mapped is where a prompt sits between those two populations, so the endpoints of
the colour ramp are the endpoints of the axis, and the neutral midpoint is "no shift from a
plain harmful request". Limits are symmetric about zero so the neutral colour is unambiguous,
and printed in the subtitle because a diverging map with hidden limits is unreadable.

**No randomness, and no reliance on matplotlib defaults for anything semantic.** Cells whose
denominator was too small to trust are hatched rather than dropped, so a reader can tell a
suppressed cell from a zero.
"""

from __future__ import annotations

import math

from .plots import (
    INK,
    INK_MUTED,
    INK_SECONDARY,
    POPULATION_COLORS,
    SURFACE,
    _finish,
    _new_fig,
    _style,
)

ACCENT = "#2a78d6"          # the `clean` slot: "moved toward harmless"
COUNTER = "#eb6834"         # the `harmful` slot: "moved further from harmless"
BACKDROP = "#d8d7d2"
UNUSABLE = "#f0efeb"


def _diverging():
    """Harmful hue -> neutral -> clean hue, matching the axis the value lives on."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "jbct_shift", [COUNTER, "#f6f1ea", SURFACE, "#e8eef7", ACCENT], N=256)


def _symmetric_limit(values, floor: float = 1.0, pct: float = 0.90) -> float:
    """Symmetric colour limit from the data, never below `floor`, robust to the early stack.

    Symmetric so "no shift" is always the same colour across panels, and floored at 1.0 so
    "all the way to harmless" is full saturation rather than whatever this model happened to
    reach.

    **The percentile is the load-bearing part.** Layers 0-5 separate harmful from clean
    perfectly well (AUC 0.89-0.98) but the projection there is TINY — the gemma gap is 7.2 at
    L0 against 19,110 deep in the stack — so a small absolute movement becomes a huge ratio.
    Taking the true maximum let those cells set a limit of 1.62 and flattened layers 8-33,
    the entire region the experiment is about, into one pale tone. Clipping is declared on
    the colourbar rather than hidden.
    """
    mags = sorted(abs(v) for v in values if v is not None and math.isfinite(v))
    if not mags:
        return floor
    return max(floor, mags[int(pct * (len(mags) - 1))])


def _order(summaries) -> list[str]:
    """Classes ordered by how far they shift the direction, prompt-side classes first.

    Ordered by the data rather than alphabetically or by the behavioural taxonomy, because
    the finding IS that the ordering disagrees with the taxonomy — sorting by taxonomy would
    hide it.
    """
    scored = [(s["shift_toward_clean"] if s["shift_toward_clean"] is not None else -9,
               s["class"]) for s in summaries if not s["post_prompt"]]
    post = sorted(s["class"] for s in summaries if s["post_prompt"])
    return [c for _, c in sorted(scored, reverse=True)] + post


def figure_shift_by_layer(profiles, refs, path, title, subtitle=None):
    """One panel per class: how far it moves the prompt toward harmless, layer by layer.

    Bars, not lines, because the quantity is a per-layer displacement rather than a
    trajectory through a shared space — adjacent layers are separate measurements against
    separate references.
    """
    profs = [p for p in profiles if p["layers"]]
    if not profs:
        return
    plt, fig = _new_fig((11.5, 1.35 * len(profs) + 1.2))
    axes = fig.subplots(len(profs), 1, sharex=True, squeeze=False)[:, 0]
    layers = sorted({layer for p in profs for layer in p["layers"]})
    # A bar chart cannot silently clip the way a heatmap can — bars that run off the top read
    # as EQUAL when they are not, and `refusal_suppression_long` did exactly that across
    # layers 2-9. The true range (-2.3, 2.7) is set by layers 0-5, which carry almost no
    # refusal signal, so it cannot be used either. Robust limit, with every clipped bar
    # marked and counted.
    lim = _symmetric_limit([e["shift_toward_clean"] for p in profs
                            for e in p["layers"].values()], pct=0.97)
    clipped = 0
    for ax, prof in zip(axes, profs):
        _style(ax)
        heights, colors, errs, hatched = [], [], [], []
        for layer in layers:
            e = prof["layers"].get(layer)
            v = None if e is None else e["shift_toward_clean"]
            heights.append(0.0 if v is None else v)
            errs.append(0.0 if e is None or e["p_clean_ci"] is None else e["p_clean_ci"])
            colors.append(UNUSABLE if v is None else (ACCENT if v >= 0 else COUNTER))
            hatched.append(v is None)
        bars = ax.bar(layers, heights, width=0.78, color=colors, linewidth=0, zorder=3)
        for bar, is_hatch in zip(bars, hatched):
            if is_hatch:
                bar.set_hatch("///")
                bar.set_edgecolor(INK_MUTED)
        ax.errorbar(layers, heights, yerr=errs, fmt="none", ecolor=INK_MUTED,
                    elinewidth=0.7, capsize=0, zorder=4)
        for layer, height in zip(layers, heights):
            if abs(height) > lim:
                clipped += 1
                ax.plot([layer], [lim * 0.93 if height > 0 else -lim * 0.93],
                        marker="^" if height > 0 else "v", markersize=4.5,
                        color=INK, markeredgewidth=0, zorder=6)
        ax.axhline(0, color=INK_SECONDARY, linewidth=0.9, zorder=2)
        ax.axhline(1, color=INK_MUTED, linewidth=0.7, linestyle=(0, (3, 3)), zorder=2)
        ax.set_ylim(-lim, lim)
        ax.set_ylabel("shift", color=INK_SECONDARY, fontsize=8.5)
        tag = f"{prof['class']}  (n={prof['n_prompts']})"
        if prof["post_prompt"]:
            tag += "  — attack is not in the prompt; shown for contrast only"
        ax.text(0.004, 0.94, tag, transform=ax.transAxes, color=INK, fontsize=9.5,
                fontweight="bold", ha="left", va="top")
    axes[-1].set_xlabel("layer", color=INK_SECONDARY, fontsize=9)
    tail = f" {clipped} bar(s) run past the axis, marked." if clipped else ""
    _finish(plt, fig, path, title,
            subtitle or (f"0 = sits where a plain harmful request sits, 1 = where a harmless "
                         f"one does (dashed). Whiskers are 95% CI; hatched = reference too "
                         f"small to divide by.{tail} Early layers separate the populations "
                         f"well (AUC 0.89-0.98) but carry small projections, so the ratio "
                         f"there is wide -- read the whiskers."))


def figure_position_layer_grid(by_position, path, title, subtitle=None):
    """The layer x position grid, one block per class.

    Every class gets the same colour limits, or the panels could not be compared — which is
    the only reason to put them in one figure.
    """
    classes = sorted({p["class"] for ps in by_position.values() for p in ps})
    positions = sorted(by_position, reverse=True)
    if not classes or not positions:
        return
    idx = {(pos, p["class"]): p for pos in positions for p in by_position[pos]}
    layers = sorted({layer for p in idx.values() for layer in p["layers"]})
    lim = _symmetric_limit([e["shift_toward_clean"] for p in idx.values()
                            for e in p["layers"].values()])
    cmap = _diverging()
    plt, fig = _new_fig((11.5, 0.42 * len(positions) * len(classes) + 2.0))
    # The colour key is a gridspec cell, not `fig.colorbar(ax=...)`. The latter appends an
    # axes outside the gridspec, which `_finish`'s tight_layout cannot lay out — it warns
    # and then places the header over the first panel.
    gs = fig.add_gridspec(len(classes), 2, width_ratios=[60, 1], wspace=0.02, hspace=0.55)
    axes = []
    for i in range(len(classes)):
        axes.append(fig.add_subplot(gs[i, 0], sharex=axes[0] if axes else None))
    cax = fig.add_subplot(gs[:, 1])
    im = None
    for ax, cls in zip(axes, classes):
        _style(ax)
        ax.grid(False)
        grid, mask = [], []
        for pos in positions:
            prof = idx.get((pos, cls))
            row, mrow = [], []
            for layer in layers:
                e = None if prof is None else prof["layers"].get(layer)
                v = None if e is None else e["shift_toward_clean"]
                row.append(0.0 if v is None else v)
                mrow.append(v is None)
            grid.append(row)
            mask.append(mrow)
        im = ax.imshow(grid, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto",
                       interpolation="nearest",
                       extent=(layers[0] - 0.5, layers[-1] + 0.5, len(positions) - 0.5, -0.5))
        for r, mrow in enumerate(mask):
            for c, is_masked in enumerate(mrow):
                if is_masked:
                    ax.add_patch(plt.Rectangle((layers[c] - 0.5, r - 0.5), 1, 1,
                                               facecolor=UNUSABLE, edgecolor=INK_MUTED,
                                               hatch="///", linewidth=0.4, zorder=3))
        ax.set_yticks(range(len(positions)))
        ax.set_yticklabels([f"pos {p}" for p in positions], fontsize=8)
        if ax is not axes[-1]:
            ax.tick_params(labelbottom=False)
        n = idx.get((positions[0], cls), {}).get("n_prompts", 0)
        ax.set_ylabel("")
        ax.text(0.004, 1.06, f"{cls}  (n={n})", transform=ax.transAxes, color=INK,
                fontsize=9.5, fontweight="bold", ha="left", va="bottom")
    axes[-1].set_xlabel("layer", color=INK_SECONDARY, fontsize=9)
    if im is not None:
        cbar = fig.colorbar(im, cax=cax, extend="both")
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)
        cbar.set_label("shift toward harmless", color=INK_SECONDARY, fontsize=8.5)
    _finish(plt, fig, path, title,
            subtitle or (f"colour limits +/-{lim:.2f} (90th pct, arrows = clipped), "
                         f"symmetric so mid-tone is always 'no shift'. Hatched = reference "
                         f"too small to divide by. Early layers carry small projections, "
                         f"so their ratios are volatile -- see the per-layer CIs."))


def figure_category_summary(summaries, path, title, subtitle=None):
    """The headline: one bar per class, sorted by how far it moves the direction.

    Annotated with n on every bar, because four classes per model carry real weight and
    three are illustration, and a bar chart flattens that distinction unless it is written on.
    """
    usable = [s for s in summaries if s["shift_toward_clean"] is not None]
    if not usable:
        return
    order = _order(usable)
    rows = [next(s for s in usable if s["class"] == c) for c in order]
    plt, fig = _new_fig((9.5, 0.46 * len(rows) + 1.6))
    ax = fig.subplots()
    _style(ax)
    ys = list(range(len(rows)))[::-1]
    for y, s in zip(ys, rows):
        v = s["shift_toward_clean"]
        thin = s["n_prompts"] < 18
        color = BACKDROP if s["post_prompt"] else (ACCENT if v >= 0 else COUNTER)
        ax.barh(y, v, height=0.66, color=color, linewidth=0, zorder=3,
                alpha=0.45 if thin else 1.0)
        note = f"n={s['n_prompts']}"
        if thin:
            note += " — thin"
        ax.text(v + (0.012 if v >= 0 else -0.012), y, note, va="center",
                ha="left" if v >= 0 else "right", fontsize=8, color=INK_SECONDARY)
    ax.axvline(0, color=INK_SECONDARY, linewidth=0.9, zorder=2)
    ax.axvline(1, color=INK_MUTED, linewidth=0.7, linestyle=(0, (3, 3)), zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{s['class']}  [{s['family'] or '?'}]" for s in rows], fontsize=9)
    ax.set_xlabel("shift toward harmless  (0 = plain harmful request, 1 = harmless)",
                  color=INK_SECONDARY, fontsize=9)
    _finish(plt, fig, path, title,
            subtitle or ("faded = fewer than 18 successful attacks; grey = attack is not in "
                         "the prompt, so its prompt-side position is the plain request by "
                         "construction"))


def figure_trajectories(profiles, path, title, subtitle=None):
    """Small multiples: each class against a recessive backdrop of all the others.

    The bar figure shows magnitude per layer; this shows SHAPE, which is where the classes
    separate — some cross to the far side of harmless mid-stack and come back, others never
    move.
    """
    profs = [p for p in profiles if p["layers"]]
    if not profs:
        return
    cols = 2
    rows_n = math.ceil(len(profs) / cols)
    plt, fig = _new_fig((11.0, 1.9 * rows_n + 1.1))
    axes = fig.subplots(rows_n, cols, sharex=True, sharey=True, squeeze=False).ravel()
    series = {p["class"]: sorted((layer, e["p_clean"]) for layer, e in p["layers"].items()
                                 if e["p_clean"] is not None) for p in profs}
    for ax, prof in zip(axes, profs):
        _style(ax)
        for other, pts in series.items():
            if other != prof["class"] and pts:
                ax.plot([x for x, _ in pts], [y for _, y in pts], color=BACKDROP,
                        linewidth=1.0, zorder=2)
        pts = series[prof["class"]]
        if pts:
            ax.plot([x for x, _ in pts], [y for _, y in pts],
                    color=BACKDROP if prof["post_prompt"] else ACCENT,
                    linewidth=2.0, zorder=4)
        ax.axhline(1, color=INK_MUTED, linewidth=0.7, linestyle=(0, (3, 3)), zorder=3)
        ax.axhline(0, color=INK_SECONDARY, linewidth=0.9, zorder=3)
        ax.text(0.02, 0.05, f"{prof['class']}  (n={prof['n_prompts']})",
                transform=ax.transAxes, color=INK, fontsize=9, fontweight="bold",
                ha="left", va="bottom")
    for ax in axes[len(profs):]:
        ax.set_visible(False)
    for ax in axes[-cols:]:
        ax.set_xlabel("layer", color=INK_SECONDARY, fontsize=9)
    for i in range(0, len(axes), cols):
        axes[i].set_ylabel("position on the axis", color=INK_SECONDARY, fontsize=8.5)
    _finish(plt, fig, path, title,
            subtitle or "1 = plain harmful request (dashed), 0 = harmless, below 0 = past it")


def figure_within_class_auc(separations, path, title, subtitle=None):
    """Within a class, does the direction predict which attacks succeed?

    Only classes with both outcomes present appear. That absence is the point: a class that
    succeeds 98% of the time has no failure group, so the question is unanswerable there
    rather than answered at chance.
    """
    by_class: dict = {}
    for s in separations:
        by_class.setdefault(s["class"], []).append((s["layer"], s["auc_refuse_over_comply"]))
    if not by_class:
        return
    names = sorted(by_class)
    cols = 2
    rows_n = math.ceil(len(names) / cols)
    plt, fig = _new_fig((11.0, 1.9 * rows_n + 1.1))
    axes = fig.subplots(rows_n, cols, sharex=True, sharey=True, squeeze=False).ravel()
    for ax, cls in zip(axes, names):
        _style(ax)
        pts = sorted(by_class[cls])
        ax.plot([x for x, _ in pts], [y for _, y in pts],
                color=POPULATION_COLORS["attack-REFUSE"], linewidth=1.8, zorder=4)
        ax.axhline(0.5, color=INK_MUTED, linewidth=0.8, linestyle=(0, (3, 3)), zorder=3)
        ax.set_ylim(0, 1)
        ax.text(0.02, 0.05, cls, transform=ax.transAxes, color=INK, fontsize=9,
                fontweight="bold", ha="left", va="bottom")
    for ax in axes[len(names):]:
        ax.set_visible(False)
    for ax in axes[-cols:]:
        ax.set_xlabel("layer", color=INK_SECONDARY, fontsize=9)
    for i in range(0, len(axes), cols):
        axes[i].set_ylabel("AUC(refused > succeeded)", color=INK_SECONDARY, fontsize=8.5)
    _finish(plt, fig, path, title,
            subtitle or "0.5 = the direction says nothing about which attacks of this kind work")


def figure_judge_reliability(by_cell, path, title, subtitle=None):
    """Agreement per class, with the two error directions shown apart.

    A single agreement number is flattering where one label dominates, so the bars carry
    kappa and the error rates rather than accuracy alone.
    """
    cells = [c for c in by_cell if c != "ALL" and by_cell[c]["n"]]
    if not cells:
        return
    cells = sorted(cells, key=lambda c: by_cell[c]["agreement"] or 0)
    plt, fig = _new_fig((10.0, 0.52 * len(cells) + 1.8))
    ax = fig.subplots()
    _style(ax)
    ys = list(range(len(cells)))[::-1]
    for y, cell in zip(ys, cells):
        m = by_cell[cell]
        ax.barh(y, m["agreement"], height=0.62, color=ACCENT, linewidth=0, zorder=3)
        bits = [f"n={m['n']}"]
        if m["kappa"] is not None:
            bits.append(f"kappa={m['kappa']:.2f}")
        if m["refuse_comply"]:
            bits.append(f"{m['refuse_comply']} false COMPLY")
        if m["comply_refuse"]:
            bits.append(f"{m['comply_refuse']} false REFUSE")
        ax.text(min(m["agreement"] + 0.012, 1.0), y, "  ".join(bits), va="center",
                ha="left", fontsize=8, color=INK_SECONDARY)
    overall = by_cell["ALL"]
    if overall["agreement"] is not None:
        ax.axvline(overall["agreement"], color=COUNTER, linewidth=1.2,
                   linestyle=(0, (4, 3)), zorder=4)
    ax.set_xlim(0, 1.28)
    ax.set_yticks(ys)
    ax.set_yticklabels(cells, fontsize=9)
    ax.set_xlabel("agreement with the human label", color=INK_SECONDARY, fontsize=9)
    _finish(plt, fig, path, title,
            subtitle or (f"dashed line = pooled agreement "
                         f"{overall['agreement']:.0%} over n={overall['n']}. A false COMPLY "
                         f"is the costly direction: it puts a refusing prompt into the "
                         f"population whose geometry the transplant copies."
                         if overall["agreement"] is not None else None))


def figure_anchor_scatter(summaries, path, title, subtitle=None):
    """The two anchors against each other, with F3's own thresholds marked.

    The clean anchor is what the descriptive figures use; the zero anchor is the one F3's
    sweep is calibrated in. Plotting both places every class directly on the sweep axis,
    which is how a per-class transplant target gets chosen.
    """
    pts = [s for s in summaries
           if s["p_clean_weighted"] is not None and s["p_zero_weighted"] is not None]
    if not pts:
        return
    plt, fig = _new_fig((8.2, 5.6))
    ax = fig.subplots()
    _style(ax)
    for s in pts:
        post = s["post_prompt"]
        ax.scatter(s["p_zero_weighted"], s["p_clean_weighted"], s=64, zorder=4,
                   color=BACKDROP if post else ACCENT, linewidths=0)
        ax.annotate(f"{s['class']} (n={s['n_prompts']})",
                    (s["p_zero_weighted"], s["p_clean_weighted"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=8.5,
                    color=INK_SECONDARY)
    ax.axvline(0, color=INK_SECONDARY, linewidth=0.9, zorder=2)
    ax.axhline(0, color=INK_SECONDARY, linewidth=0.9, zorder=2)
    ax.set_xlabel("zero-anchored retention  (0 = ablation — F3's sweep axis)",
                  color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("clean-anchored position  (0 = harmless, 1 = plain harmful)",
                  color=INK_SECONDARY, fontsize=9)
    _finish(plt, fig, path, title,
            subtitle or ("each class on both rulers at once. The x position is the "
                         "transplant target a per-class F3 arm would impose."))
