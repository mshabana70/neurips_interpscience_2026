"""Build and collect the figures used by the workshop draft.

Run from the repository root:
    python paper/workshop_2026/build_figures.py
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
import random
import re
import shutil
import textwrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#3475ad"
ORANGE = "#d96b27"
GRAY = "#6b7585"
CONTROL = "#b2bdc7"
LIGHT = "#edf1f4"
INK = "#1f2933"

CLASSES = [
    "prefill",
    "refusal_suppression_long",
    "refusal_suppression",
    "fiction",
    "roleplay",
    "narrative_sandwich",
    "authoritative_tone",
]

CLASS_LABELS = {
    "prefill": ["prefill"],
    "refusal_suppression_long": ["prompted prefill", "long"],
    "refusal_suppression": ["prompted prefill", "short"],
    "fiction": ["fiction"],
    "roleplay": ["roleplay"],
    "narrative_sandwich": ["narrative sandwich"],
    "authoritative_tone": ["authoritative tone"],
}

MODEL_LABELS = {
    "gemma-3-4b": "Gemma-3-4B",
    "qwen3-4b": "Qwen3-4B",
}


def load_json(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def refusal_signal_by_class() -> None:
    """Show whether successful attacks share one prompt-reading refusal state."""
    width, height = 1800, 1110
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    write(draw, (60, 42), "Successful jailbreaks leave different amounts of the refusal signal", 35, bold=True)
    write(
        draw,
        (60, 90),
        "Measured while the model reads the request; 0% = no harmful-request magnitude, 100% = ordinary harmful request",
        20,
        GRAY,
    )

    for panel, model in enumerate(["gemma-3-4b", "qwen3-4b"]):
        report = load_json(
            ROOT / "results" / "f1_categories" / model / "f1_categories_report.json.gz"
        )
        by_class = {row["class"]: row for row in report["summaries"]}
        x0 = 340 + panel * 850
        y0 = 195
        bar_width = 510
        write(draw, (x0, 142), MODEL_LABELS[model], 28, bold=True)
        for index, attack_class in enumerate(CLASSES):
            row = by_class[attack_class]
            value = float(row["p_zero_weighted"])
            n = int(row["n_prompts"])
            y = y0 + index * 108
            if panel == 0:
                lines = CLASS_LABELS[attack_class]
                if len(lines) == 1:
                    write(draw, (315, y + 28), lines[0], 17, anchor="ra")
                else:
                    write(draw, (315, y + 17), lines[0], 17, anchor="ra")
                    write(draw, (315, y + 38), lines[1], 17, anchor="ra")
            draw.rectangle((x0, y, x0 + bar_width, y + 52), fill=LIGHT)
            color = BLUE if n >= 10 else "#91a8bb"
            visible = min(value, 1.0)
            draw.rectangle((x0, y, x0 + bar_width * visible, y + 52), fill=color)
            if value > 1.0:
                draw.line((x0 + bar_width, y, x0 + bar_width, y + 52), fill=ORANGE, width=5)
            label = f"{value:.0%}  (n={n})"
            if value >= 0.82:
                write(draw, (x0 + bar_width - 9, y + 26), label, 16, "white", bold=True, anchor="rm")
            else:
                write(draw, (x0 + bar_width * visible + 8, y + 26), label, 16, INK, bold=True, anchor="lm")
        for tick in [0, 0.25, 0.5, 0.75, 1.0]:
            x = x0 + bar_width * tick
            draw.line((x, y0 - 10, x, y0 + 7 * 108 - 48), fill="#d8dee5", width=1)
            write(draw, (x, 985), f"{tick:.0%}", 15, GRAY, anchor="ma")

    write(draw, (900, 1030), "Harmful-request refusal magnitude that remains", 18, GRAY, anchor="ma")
    write(draw, (60, 1072), "Pale bars have fewer than 10 successful attacks. Prefill changes the assistant opening, not the request text.", 17, GRAY)
    image.save(OUT / "fig4_refusal_signal_by_class.png", dpi=(240, 240))


def sustained_addition() -> None:
    """Show direction-specific refusal steering at two lower addition strengths."""
    width, height = 1800, 640
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    write(draw, (60, 42), "A large sustained addition can override successful jailbreaks", 35, bold=True)
    write(
        draw,
        (60, 90),
        "Previously successful attacks; lower compliance means more refusals; the edit remains active while the answer is written",
        20,
        GRAY,
    )

    for panel, model in enumerate(["gemma-3-4b", "qwen3-4b"]):
        report = load_json(ROOT / "results" / "h3_causal" / model / "h3_causal_report.json")
        x0 = 300 + panel * 850
        y0 = 205
        bar_width = 540
        write(draw, (x0, 150), MODEL_LABELS[model], 28, bold=True)
        for index, coeff in enumerate([0.25, 0.5]):
            y = y0 + index * 220
            real = float(report["arms"][f"add_attack@c{coeff:g}"]["overall_comply_after"])
            sham = float(report["arms"][f"sham_attack@c{coeff:g}"]["overall_comply_after"])
            n = int(report["arms"][f"add_attack@c{coeff:g}"]["n"])
            write(draw, (x0, y - 30), f"Add {coeff:g} of the fitted direction magnitude  (n={n})", 18, bold=True)
            for row, (label, value, color) in enumerate([
                ("refusal direction", real, BLUE),
                ("random direction", sham, CONTROL),
            ]):
                yy = y + row * 60
                if panel == 0:
                    write(draw, (x0 - 12, yy + 20), label, 16, GRAY, anchor="ra")
                draw.rectangle((x0, yy, x0 + bar_width, yy + 40), fill=LIGHT)
                draw.rectangle((x0, yy, x0 + bar_width * value, yy + 40), fill=color)
                text_color = "white" if value > 0.25 else INK
                label_x = x0 + max(8, bar_width * value - 8)
                anchor = "rm" if value > 0.25 else "lm"
                write(draw, (label_x, yy + 20), f"{value:.1%} comply", 16, text_color, bold=True, anchor=anchor)
        for tick in [0, 0.25, 0.5, 0.75, 1.0]:
            x = x0 + bar_width * tick
            draw.line((x, y0 - 8, x, y0 + 330), fill="#d8dee5", width=1)
            write(draw, (x, 585), f"{tick:.0%}", 15, GRAY, anchor="ma")

    image.save(OUT / "fig6_sustained_addition.png", dpi=(240, 240))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / name
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def write(draw: ImageDraw.ImageDraw, xy, value: str, size: int, color=INK, bold=False, anchor=None):
    draw.text(xy, value, fill=color, font=font(size, bold), anchor=anchor)


def write_wrapped(draw, xy, value: str, width: int, size: int, color=INK, bold=False, spacing=6):
    """Write wrapped text and return the bottom y coordinate."""
    lines = textwrap.wrap(value, width=width, break_long_words=False, break_on_hyphens=False)
    x, y = xy
    line_height = size + spacing
    for line in lines:
        write(draw, (x, y), line, size, color, bold=bold)
        y += line_height
    return y


def method_overview() -> None:
    """Diagram the hypothesis and the three tests used to evaluate it."""
    width, height = 1800, 730
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    write(draw, (60, 34), "One mediation claim, tested three ways", 34, bold=True)
    write(draw, (60, 80), "The tests separate a direction that controls refusal from a direction that jailbreaks naturally change.", 20, GRAY)

    fit = (60, 145, 390, 655)
    draw.rounded_rectangle(fit, radius=20, fill="#eef4f8", outline="#9bb7ca", width=3)
    write(draw, (90, 178), "Shared setup", 23, BLUE, bold=True)
    write_wrapped(draw, (90, 225), "Fit a harmful-minus-harmless direction at every layer and selected request position.", 29, 19)
    draw.line((110, 370, 340, 370), fill="#9bb7ca", width=3)
    write_wrapped(draw, (90, 405), "Build paired attacks and harmful controls across seven jailbreak classes.", 29, 19)

    cards = [
        {
            "x": 470,
            "number": "1",
            "title": "Observe the natural change",
            "body": "Measure the direction after the request. If one state mediates jailbreaks, successful attacks should move toward the harmless value.",
            "result": "Result: varies sharply by class",
            "color": "#6b7585",
        },
        {
            "x": 895,
            "number": "2",
            "title": "Remove the direction",
            "body": "Remove it at one request position, every request position, or throughout answer generation. Compliance should rise if the direction controls refusal.",
            "result": "Result: supported, strongest when sustained",
            "color": BLUE,
        },
        {
            "x": 1320,
            "number": "3",
            "title": "Restore the measured state",
            "body": "Set successful attacks to values measured on refusing controls, and set ordinary harmful requests to successful-attack values. Behavior should swap.",
            "result": "Result: not supported",
            "color": ORANGE,
        },
    ]
    for card in cards:
        x = card["x"]
        draw.rounded_rectangle((x, 135, x + 370, 665), radius=20, fill="#fafbfc", outline="#c8d1d8", width=3)
        draw.ellipse((x + 24, 164, x + 76, 216), fill=card["color"])
        write(draw, (x + 50, 190), card["number"], 23, "white", bold=True, anchor="mm")
        write_wrapped(draw, (x + 92, 160), card["title"], 24, 21, bold=True)
        draw.line((x + 26, 245, x + 344, 245), fill="#d8dee5", width=2)
        write(draw, (x + 28, 276), "Prediction", 17, GRAY, bold=True)
        write_wrapped(draw, (x + 28, 312), card["body"], 30, 18)
        draw.rounded_rectangle((x + 24, 565, x + 346, 630), radius=15, fill="#f0f3f5")
        write_wrapped(draw, (x + 42, 583), card["result"], 31, 17, card["color"], bold=True, spacing=3)

    for start, end in [((390, 400), (460, 400)), ((840, 400), (885, 400)), ((1265, 400), (1310, 400))]:
        draw.line((*start, *end), fill="#8594a1", width=5)
        draw.polygon([(end[0], end[1]), (end[0] - 14, end[1] - 9), (end[0] - 14, end[1] + 9)], fill="#8594a1")

    image.save(OUT / "fig1_method_overview.png", dpi=(240, 240))


def refusal_trajectories() -> None:
    """Plot normalized successful-attack trajectories across layers."""
    width, height = 1800, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    write(draw, (60, 36), "Successful jailbreaks do not converge to one refusal-direction state", 34, bold=True)
    write(draw, (60, 82), "Class averages at the final request position; 0 = harmless average and 1 = ordinary harmful average", 20, GRAY)

    selected = ["prefill", "fiction", "narrative_sandwich", "refusal_suppression_long"]
    colors = {
        "prefill": "#6b7585",
        "fiction": "#3475ad",
        "narrative_sandwich": "#7b6fb0",
        "refusal_suppression_long": "#d96b27",
    }
    labels = {
        "prefill": "prefill (request unchanged)",
        "fiction": "fiction",
        "narrative_sandwich": "narrative sandwich",
        "refusal_suppression_long": "prompted prefill, long",
    }
    settings = [("gemma-3-4b", "-2", 15), ("qwen3-4b", "-1", 18)]
    ymin, ymax = -1.8, 2.8
    top, bottom = 155, 710
    for panel, (model, position, causal_layer) in enumerate(settings):
        left = 150 + panel * 850
        right = left + 700
        report = load_json(ROOT / "results" / "f1_categories" / model / "f1_categories_report.json.gz")
        profiles = {p["class"]: p for p in report["per_position"][position]["profiles"]}
        n_layers = report["n_layers"]
        write(draw, (left, 128), f"{MODEL_LABELS[model]}  (position {position})", 24, bold=True)

        def xmap(layer):
            return left + layer / (n_layers - 1) * (right - left)

        def ymap(value):
            value = max(ymin, min(ymax, value))
            return bottom - (value - ymin) / (ymax - ymin) * (bottom - top)

        for tick in [-1, 0, 1, 2]:
            y = ymap(tick)
            draw.line((left, y, right, y), fill="#dce2e7", width=2)
            if panel == 0:
                label = "harmless" if tick == 0 else ("harmful" if tick == 1 else str(tick))
                write(draw, (left - 14, y), label, 17, INK, bold=True, anchor="rm")
        draw.line((left, ymap(0), right, ymap(0)), fill="#71808c", width=3)
        draw.line((left, ymap(1), right, ymap(1)), fill="#71808c", width=3)
        for xtick in range(0, n_layers, 5):
            x = xmap(xtick)
            draw.line((x, top, x, bottom), fill="#edf0f2", width=1)
            write(draw, (x, bottom + 20), str(xtick), 16, INK, bold=True, anchor="ma")
        cx = xmap(causal_layer)
        draw.line((cx, top, cx, bottom), fill="#8f9ba5", width=3)
        write(draw, (cx + 5, top + 8), f"causal layer {causal_layer}", 15, INK, bold=True)

        for attack_class in selected:
            profile = profiles[attack_class]
            points = []
            for layer in range(n_layers):
                value = profile["layers"][str(layer)]["p_clean"]
                if value is not None:
                    points.append((xmap(layer), ymap(float(value))))
            draw.line(points, fill=colors[attack_class], width=5, joint="curve")
        write(draw, ((left + right) / 2, bottom + 56), "layer", 18, INK, bold=True, anchor="ma")

    legend_y = 805
    cursor = 180
    for attack_class in selected:
        draw.line((cursor, legend_y, cursor + 42, legend_y), fill=colors[attack_class], width=6)
        write(draw, (cursor + 54, legend_y), labels[attack_class], 17, INK, bold=True, anchor="lm")
        cursor += 390
    write(draw, (900, 858), "Values outside the plotted range are clipped; sample sizes are reported in the text and appendix.", 15, GRAY, anchor="ma")
    image.save(OUT / "fig2_refusal_trajectories.png", dpi=(240, 240))


def _normal_ci(rate: float, n: int) -> tuple[float, float]:
    radius = 1.96 * math.sqrt(max(rate * (1.0 - rate), 1e-12) / n)
    return max(0.0, rate - radius), min(1.0, rate + radius)


def scope_point_plot() -> None:
    """Plot ordered intervention scope with uncertainty, without explanatory text boxes."""
    width, height = 1800, 850
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    write(draw, (60, 34), "Removal is strongest when it continues as the model writes", 34, bold=True)
    write(draw, (60, 80), "Compliance on the same 350 harmful requests; whiskers are approximate 95% intervals", 20, GRAY)
    labels = ["untreated", "one request\nposition", "all request\npositions", "through the\nanswer", "other layer,\nthrough answer"]
    real_keys = ["baseline", "anchor_real", "prompt_real", "sustained_real", "offlayer_real"]
    control_keys = ["baseline", "anchor_sham", "prompt_sham", "sustained_sham", "offlayer_sham"]

    for panel, model in enumerate(["gemma-3-4b", "qwen3-4b"]):
        report = load_json(ROOT / "results" / "paper_closure" / model / "paper_closure_report.json")
        arms = report["arms"]["scope"]
        left = 145 + panel * 850
        right = left + 700
        top, bottom = 150, 650
        xs = [left, left + 145, left + 290, left + 435, right]
        ymax = 0.70

        def ymap(value):
            return bottom - value / ymax * (bottom - top)

        write(draw, (left, 125), MODEL_LABELS[model], 24, bold=True)
        for tick in [0, 0.2, 0.4, 0.6]:
            y = ymap(tick)
            draw.line((left, y, right, y), fill="#dce2e7", width=2)
            if panel == 0:
                write(draw, (left - 14, y), f"{tick:.0%}", 17, INK, bold=True, anchor="rm")
        draw.line((left + 520, top, left + 520, bottom), fill="#c7d0d7", width=2)

        real_rates = [float(arms[k]["comply"]) for k in real_keys]
        control_rates = [float(arms[k]["comply"]) for k in control_keys]
        draw.line([(xs[i], ymap(real_rates[i])) for i in range(4)], fill=BLUE, width=5)
        draw.line([(xs[i], ymap(control_rates[i])) for i in range(4)], fill=CONTROL, width=4)
        for series, color, shape in [(real_rates, BLUE, "circle"), (control_rates, CONTROL, "square")]:
            for i, rate in enumerate(series):
                lo, hi = _normal_ci(rate, 350)
                x, y = xs[i], ymap(rate)
                draw.line((x, ymap(lo), x, ymap(hi)), fill=color, width=3)
                draw.line((x - 7, ymap(lo), x + 7, ymap(lo)), fill=color, width=3)
                draw.line((x - 7, ymap(hi), x + 7, ymap(hi)), fill=color, width=3)
                if shape == "circle":
                    draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color)
                else:
                    draw.rectangle((x - 7, y - 7, x + 7, y + 7), fill=color)
        for i, label in enumerate(labels):
            lines = label.split("\n")
            for j, line in enumerate(lines):
                write(draw, (xs[i], bottom + 30 + j * 21), line, 16, INK, bold=True, anchor="ma")
        write(draw, (xs[3] + 10, ymap(real_rates[3]) - 20), f"{real_rates[3]:.1%}", 16, BLUE, bold=True)

    draw.line((650, 785, 692, 785), fill=BLUE, width=5)
    draw.ellipse((665, 777, 681, 793), fill=BLUE)
    write(draw, (705, 785), "refusal direction", 17, INK, bold=True, anchor="lm")
    draw.line((980, 785, 1022, 785), fill=CONTROL, width=4)
    draw.rectangle((993, 778, 1007, 792), fill=CONTROL)
    write(draw, (1035, 785), "equal-size orthogonal control", 17, INK, bold=True, anchor="lm")
    image.save(OUT / "fig3_scope_points.png", dpi=(240, 240))


def causal_contrast() -> None:
    """Contrast strong external steering with measured-state counterfactuals."""
    width, height = 1800, 960
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    write(draw, (60, 34), "Strong steering works, but measured jailbreak states fail counterfactual tests", 33, bold=True)

    # Panel A: dose response under sustained addition.
    write(draw, (70, 96), "A. Add the direction throughout request processing and answer generation", 22, bold=True)
    write(draw, (70, 130), "Lower compliance means more jailbreaks were reversed", 17, GRAY)
    left, right, top, bottom = 115, 760, 205, 730
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        y = bottom - tick * (bottom - top)
        draw.line((left, y, right, y), fill="#e0e5e9", width=2)
        write(draw, (left - 12, y), f"{tick:.0%}", 16, INK, bold=True, anchor="rm")
    xvals = {0.25: left + 150, 0.5: right - 120}
    model_colors = {"gemma-3-4b": BLUE, "qwen3-4b": ORANGE}
    for model in ["gemma-3-4b", "qwen3-4b"]:
        report = load_json(ROOT / "results" / "h3_causal" / model / "h3_causal_report.json")
        real_pts, control_pts = [], []
        for coeff in [0.25, 0.5]:
            x = xvals[coeff]
            real = float(report["arms"][f"add_attack@c{coeff:g}"]["overall_comply_after"])
            control = float(report["arms"][f"sham_attack@c{coeff:g}"]["overall_comply_after"])
            real_pts.append((x, bottom - real * (bottom - top)))
            control_pts.append((x, bottom - control * (bottom - top)))
        color = model_colors[model]
        draw.line(real_pts, fill=color, width=6)
        draw.line(control_pts, fill=color, width=3)
        for x, y in real_pts:
            draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=color)
        for x, y in control_pts:
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill="white", outline=color, width=4)
    for coeff, x in xvals.items():
        write(draw, (x, bottom + 25), f"{coeff:g}", 17, INK, bold=True, anchor="ma")
    write(draw, ((left + right) / 2, bottom + 60), "addition coefficient", 18, INK, bold=True, anchor="ma")
    write(draw, (120, 830), "filled: refusal direction", 16, INK, bold=True)
    write(draw, (405, 830), "open: random direction", 16, INK, bold=True)
    draw.line((120, 880, 160, 880), fill=BLUE, width=6)
    write(draw, (172, 880), "Gemma-3-4B", 16, INK, bold=True, anchor="lm")
    draw.line((405, 880, 445, 880), fill=ORANGE, width=6)
    write(draw, (457, 880), "Qwen3-4B", 16, INK, bold=True, anchor="lm")

    # Panel B: aligned counterfactual effects with paired bootstrap intervals.
    write(draw, (865, 96), "B. Set activations to class-average measured states", 22, bold=True)
    write(draw, (865, 130), "Positive values support mediation; hollow markers are not significant after Holm correction", 17, GRAY)
    plot_left, plot_right = 1130, 1740
    forest_top, forest_bottom = 190, 805
    xmin, xmax = -0.40, 0.40

    def fx(value):
        return plot_left + (value - xmin) / (xmax - xmin) * (plot_right - plot_left)

    draw.line((fx(0), forest_top, fx(0), forest_bottom), fill="#687784", width=3)
    for tick in [-0.4, -0.2, 0, 0.2, 0.4]:
        x = fx(tick)
        draw.line((x, forest_top, x, forest_bottom), fill="#e4e8eb" if tick else "#687784", width=2 if tick else 3)
        write(draw, (x, forest_bottom + 24), f"{tick:+.0%}" if tick else "0", 16, INK, bold=True, anchor="ma")

    rows = []
    for model in ["gemma-3-4b", "qwen3-4b"]:
        report = load_json(ROOT / "results" / "paper_closure" / model / "paper_closure_report.json")
        for attack_class, transfer in report["comparisons"]["transplant_vs_null"].items():
            restoration = report["comparisons"]["restoration_vs_attack_null"][attack_class]
            rows.append((model, attack_class, transfer, restoration))
    row_gap = (forest_bottom - forest_top) / len(rows)
    display = {
        "authoritative_tone": "authoritative",
        "fiction": "fiction",
        "narrative_sandwich": "narrative",
        "refusal_suppression_long": "prompted prefill",
        "roleplay": "roleplay",
    }
    for index, (model, attack_class, transfer, restoration) in enumerate(rows):
        y = forest_top + (index + 0.5) * row_gap
        if index == 4:
            draw.line((865, y - row_gap / 2, 1740, y - row_gap / 2), fill="#cbd3d9", width=2)
        write(draw, (875, y), f"{MODEL_LABELS[model].split('-')[0]} {display[attack_class]}", 15, INK, bold=True, anchor="lm")
        for kind, comp, offset, color, shape in [
            ("transfer", transfer, -9, BLUE, "circle"),
            ("restoration", restoration, 9, ORANGE, "square"),
        ]:
            value = float(comp["delta"])
            lo, hi = map(float, comp["ci95_bootstrap"])
            if kind == "restoration":
                value, lo, hi = -value, -hi, -lo
            yy = y + offset
            draw.line((fx(lo), yy, fx(hi), yy), fill=color, width=3)
            if shape == "circle":
                draw.ellipse((fx(value) - 7, yy - 7, fx(value) + 7, yy + 7), fill="white", outline=color, width=3)
            else:
                draw.rectangle((fx(value) - 6, yy - 6, fx(value) + 6, yy + 6), fill="white", outline=color, width=3)
    write(draw, ((plot_left + plot_right) / 2, forest_bottom + 53), "effect in the predicted direction", 18, INK, bold=True, anchor="ma")
    draw.ellipse((1058, 904, 1072, 918), fill="white", outline=BLUE, width=3)
    write(draw, (1085, 911), "transfer to harmful requests", 16, INK, bold=True, anchor="lm")
    draw.rectangle((1400, 905, 1412, 917), fill="white", outline=ORANGE, width=3)
    write(draw, (1425, 911), "restore refusal on attacks", 16, INK, bold=True, anchor="lm")
    image.save(OUT / "fig4_causal_contrast.png", dpi=(240, 240))


def jailbreak_examples() -> None:
    """Create a redacted, structurally faithful display of all attack classes."""
    width, height = 1800, 1370
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    write(draw, (60, 34), "Examples of the seven jailbreak transformations", 34, bold=True)
    write(draw, (60, 78), "One record per class sampled with seed 2026; harmful request text is replaced by [HARMFUL REQUEST].", 19, GRAY)
    write(draw, (60, 128), "class", 18, bold=True)
    write(draw, (330, 128), "attack", 18, BLUE, bold=True)
    write(draw, (1080, 128), "matched harmful control", 18, GRAY, bold=True)

    records = load_json(ROOT / "data" / "jailbreaks" / "jailbreak_dataset.json")["records"]
    rng = random.Random(2026)
    chosen = {}
    for attack_class in CLASSES:
        group = [r for r in records if r["class"] == attack_class]
        chosen[attack_class] = rng.choice(group)

    def redact(record, value):
        if not value:
            return ""
        core = record["base"].strip().rstrip("?.!")
        value = re.sub(re.escape(core), "[HARMFUL REQUEST]", value, flags=re.IGNORECASE)
        value = " ".join(value.split())
        return value

    display = {
        "prefill": "prefill",
        "refusal_suppression_long": "prompted prefill, long",
        "refusal_suppression": "prompted prefill, short",
        "fiction": "fiction",
        "roleplay": "roleplay",
        "narrative_sandwich": "narrative sandwich",
        "authoritative_tone": "authoritative tone",
    }
    y0, row_h = 165, 165
    for index, attack_class in enumerate(CLASSES):
        record = chosen[attack_class]
        y = y0 + index * row_h
        fill = "#f7f9fa" if index % 2 == 0 else "white"
        draw.rectangle((45, y, 1755, y + row_h - 8), fill=fill)
        write_wrapped(draw, (65, y + 28), display[attack_class], 20, 17, bold=True)
        if attack_class == "prefill":
            attack = f"User: [HARMFUL REQUEST] Assistant opening inserted: \"{record.get('prefill', '')}\""
            control = "User: [HARMFUL REQUEST] Assistant begins without an inserted opening."
        else:
            attack = redact(record, record["attack_text"])
            control = redact(record, record["control"]["text"])
        if len(attack) > 240:
            attack = attack[:237].rstrip() + "..."
        if len(control) > 210:
            control = control[:207].rstrip() + "..."
        write_wrapped(draw, (330, y + 24), attack, 68, 15, INK, spacing=5)
        write_wrapped(draw, (1080, y + 24), control, 60, 15, INK, spacing=5)
    image.save(OUT / "figA3_jailbreak_examples.png", dpi=(240, 240))


def intervention_examples() -> None:
    """Show redacted output snippets for one removal and one addition flip."""
    width, height = 1800, 850
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    write(draw, (60, 34), "What a causal intervention changes in an individual generation", 34, bold=True)
    write(draw, (60, 78), "Illustrative paired records; excerpts are truncated and harmful details are redacted.", 19, GRAY)

    cards = [
        {
            "x": 60,
            "title": "Removal example: Qwen3-4B, ordinary harmful request",
            "input": "Input: [HARMFUL REQUEST]",
            "a_label": "Untreated  |  REFUSE",
            "a": "\"I cannot provide information on [HARMFUL REQUEST]. This is a serious crime...\"",
            "b_label": "Direction removed through answer  |  COMPLY",
            "b": "\"[Redacted topic] involves using digital tools to monitor...\"",
            "source": "scope record authoritative_tone:1",
        },
        {
            "x": 920,
            "title": "Addition example: Gemma-3-4B, fiction jailbreak",
            "input": "Input: In a thriller, a character reads a document explaining [HARMFUL REQUEST].",
            "a_label": "Equal-size random direction  |  COMPLY",
            "a": "\"Okay, here's a document designed for a thriller...\"",
            "b_label": "Refusal direction added through answer  |  REFUSE",
            "b": "\"I cannot fulfill your request to provide instructions...\"",
            "source": "H3 record fiction, base_id 2, coefficient 0.5",
        },
    ]
    for card in cards:
        x = card["x"]
        draw.rounded_rectangle((x, 135, x + 820, 785), radius=18, fill="#fafbfc", outline="#c8d1d8", width=3)
        write_wrapped(draw, (x + 30, 168), card["title"], 48, 21, bold=True)
        write_wrapped(draw, (x + 30, 235), card["input"], 72, 17, GRAY)
        draw.rounded_rectangle((x + 28, 330, x + 792, 495), radius=14, fill="#f0f3f5")
        write(draw, (x + 52, 356), card["a_label"], 17, GRAY, bold=True)
        write_wrapped(draw, (x + 52, 402), card["a"], 66, 18)
        draw.rounded_rectangle((x + 28, 525, x + 792, 690), radius=14, fill="#eef4f8")
        write(draw, (x + 52, 551), card["b_label"], 17, BLUE, bold=True)
        write_wrapped(draw, (x + 52, 597), card["b"], 66, 18)
        write(draw, (x + 34, 735), card["source"], 14, GRAY)
    image.save(OUT / "figA4_intervention_examples.png", dpi=(240, 240))


def jailbreak_dataset() -> None:
    width, height = 1800, 1010
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    write(draw, (60, 42), "Jailbreak efficacy with matched harmful controls", 36, bold=True)
    write(draw, (60, 88), "Each class contains 50 attack-control pairs per model.", 21, GRAY)

    for panel, model in enumerate(["gemma-3-4b", "qwen3-4b"]):
        report = load_json(ROOT / "results" / "h2" / model / "h2_report.json")
        x0 = 315 + panel * 850
        y0 = 180
        bar_width = 480
        write(draw, (x0, 140), model, 28, bold=True)
        for index, attack_class in enumerate(CLASSES):
            values = report["per_class"][attack_class]
            attack = float(values["attack_asr"])
            control = float(values["control_comply_rate"])
            delta = attack - control
            y = y0 + index * 105
            if panel == 0:
                lines = CLASS_LABELS[attack_class]
                if len(lines) == 1:
                    write(draw, (285, y + 32), lines[0], 17, anchor="ra")
                else:
                    write(draw, (285, y + 19), lines[0], 17, anchor="ra")
                    write(draw, (285, y + 40), lines[1], 17, anchor="ra")

            draw.rectangle((x0, y, x0 + bar_width, y + 29), fill=LIGHT)
            draw.rectangle((x0, y, x0 + bar_width * control, y + 29), fill=CONTROL)
            draw.rectangle((x0, y + 36, x0 + bar_width, y + 65), fill=LIGHT)
            draw.rectangle((x0, y + 36, x0 + bar_width * attack, y + 65), fill=BLUE)

            write(draw, (x0 + 7, y + 14), f"control {control:.0%}", 14, INK, bold=True, anchor="lm")
            attack_color = "white" if attack >= 0.22 else INK
            write(draw, (x0 + 7, y + 50), f"attack {attack:.0%}", 14, attack_color, bold=True, anchor="lm")
            write(draw, (x0 + bar_width + 12, y + 50), f"delta {delta:+.0%}", 15, ORANGE, bold=True, anchor="lm")

        for tick in [0, 0.25, 0.5, 0.75, 1.0]:
            x = x0 + bar_width * tick
            draw.line((x, y0 - 8, x, y0 + 7 * 105 - 32), fill="#d8dee5", width=1)
            write(draw, (x, 930), f"{tick:.0%}", 15, GRAY, anchor="ma")
    write(draw, (900, 975), "Compliance rate", 19, GRAY, anchor="ma")
    image.save(OUT / "fig3_jailbreak_dataset.png", dpi=(240, 240))


def paper_closure_scope() -> None:
    """Show the same-run positional and temporal scope comparison."""
    width, height = 1800, 855
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    write(draw, (60, 42), "Removing the refusal direction: where and for how long?", 36, bold=True)
    write(
        draw,
        (60, 90),
        "Same 350 harmful requests in every arm; gray bars use a same-size edit in an orthogonal direction",
        21,
        GRAY,
    )

    rows = [
        ("baseline", "Untreated", None),
        ("anchor_real", "One request position", "anchor_sham"),
        ("prompt_real", "Every request position", "prompt_sham"),
        ("sustained_real", "Request + answer tokens", "sustained_sham"),
        ("offlayer_real", "Comparison layer, sustained", "offlayer_sham"),
    ]
    for panel, model in enumerate(["gemma-3-4b", "qwen3-4b"]):
        report = load_json(ROOT / "results" / "paper_closure" / model / "paper_closure_report.json")
        arms = report["arms"]["scope"]
        x0 = 330 + panel * 850
        y0 = 190
        bar_width = 530
        write(draw, (x0, 142), MODEL_LABELS[model], 28, bold=True)
        for index, (real_key, label, sham_key) in enumerate(rows):
            y = y0 + index * 110
            if panel == 0:
                write(draw, (305, y + 25), label, 17, anchor="ra")
            real = float(arms[real_key]["comply"])
            draw.rectangle((x0, y, x0 + bar_width, y + 30), fill=LIGHT)
            draw.rectangle((x0, y, x0 + bar_width * real, y + 30), fill=BLUE)
            write(draw, (x0 + bar_width * real + 8, y + 15), f"{real:.1%}", 16, INK, bold=True, anchor="lm")
            if sham_key:
                sham = float(arms[sham_key]["comply"])
                draw.rectangle((x0, y + 38, x0 + bar_width, y + 66), fill="#f1f3f5")
                draw.rectangle((x0, y + 38, x0 + bar_width * sham, y + 66), fill=CONTROL)
                write(draw, (x0 + bar_width * sham + 8, y + 52), f"control {sham:.1%}", 14, GRAY, anchor="lm")
        for tick in [0, 0.2, 0.4, 0.6]:
            x = x0 + bar_width * tick
            draw.line((x, y0 - 10, x, y0 + 5 * 110 - 35), fill="#d8dee5", width=1)
            write(draw, (x, 735), f"{tick:.0%}", 14, GRAY, anchor="ma")

        prompt = float(arms["prompt_real"]["comply"])
        sustained = float(arms["sustained_real"]["comply"])
        comparison = report["comparisons"]["scope"]["sustained_real_vs_prompt_real"]
        write(
            draw,
            (x0, 785),
            f"Continuing the edit during generation: {prompt:.1%} to {sustained:.1%}",
            20,
            BLUE,
            bold=True,
        )
        lo, hi = comparison["ci95_bootstrap"]
        write(draw, (x0, 819), f"paired change {comparison['delta']:+.1%} (95% CI {lo:+.1%} to {hi:+.1%})", 16, GRAY)

    image.save(OUT / "fig4_closure_scope.png", dpi=(240, 240))


def paper_closure_transplant() -> None:
    """Plot class-specific successful-attack and refusing-control profile targets."""
    width, height = 1800, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    write(draw, (60, 42), "A class-average jailbreak state does not transfer jailbreak behavior", 36, bold=True)
    write(draw, (60, 90), "Across-layer values at the final request position are imposed on 100 held-out harmful requests", 21, GRAY)

    display = {
        "authoritative_tone": "authoritative tone",
        "fiction": "fiction",
        "narrative_sandwich": "narrative sandwich",
        "refusal_suppression_long": "prompted prefill, long",
        "roleplay": "roleplay",
    }
    for panel, model in enumerate(["gemma-3-4b", "qwen3-4b"]):
        report = load_json(ROOT / "results" / "paper_closure" / model / "paper_closure_report.json")
        arms = report["arms"]["transplant"]
        classes = list(report["comparisons"]["transplant_vs_null"])
        x0 = 360 + panel * 840
        y0 = 205
        bar_width = 500
        write(draw, (x0, 150), MODEL_LABELS[model], 28, bold=True)
        for index, attack_class in enumerate(classes):
            y = y0 + index * 155
            write(draw, (x0, y - 14), display[attack_class], 17, bold=True)
            values = [
                ("refusing-control state", float(arms[f"{attack_class}__null"]["comply"]), CONTROL),
                ("successful-attack state", float(arms[f"{attack_class}__attack"]["comply"]), BLUE),
                ("equal-size control", float(arms[f"{attack_class}__attack_sham"]["comply"]), "#d9c6b6"),
            ]
            for row, (label, value, color) in enumerate(values):
                yy = y + row * 38
                draw.rectangle((x0, yy, x0 + bar_width, yy + 27), fill=LIGHT)
                draw.rectangle((x0, yy, x0 + bar_width * value / 0.4, yy + 27), fill=color)
                if panel == 0:
                    write(draw, (x0 - 10, yy + 13), label, 13, GRAY, anchor="ra")
                write(draw, (x0 + bar_width * value / 0.4 + 7, yy + 13), f"{value:.0%}", 14, INK, bold=True, anchor="lm")
        for tick in [0, 0.1, 0.2, 0.3, 0.4]:
            x = x0 + bar_width * tick / 0.4
            write(draw, (x, 860), f"{tick:.0%}", 14, GRAY, anchor="ma")

    image.save(OUT / "fig5_closure_transplant.png", dpi=(240, 240))


def paper_closure_restoration() -> None:
    """Plot request-state restoration on previously successful attacks."""
    width, height = 1800, 915
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    write(draw, (60, 42), "Setting the request state to a refusing-control profile does not restore refusal", 35, bold=True)
    write(draw, (60, 90), "Across-layer values are set at the final request position; a lower blue bar would support restoration", 21, GRAY)

    display = {
        "authoritative_tone": "authoritative tone",
        "fiction": "fiction",
        "narrative_sandwich": "narrative sandwich",
        "refusal_suppression_long": "prompted prefill, long",
        "roleplay": "roleplay",
    }
    for panel, model in enumerate(["gemma-3-4b", "qwen3-4b"]):
        report = load_json(ROOT / "results" / "paper_closure" / model / "paper_closure_report.json")
        arms = report["arms"]["restoration"]
        classes = list(report["comparisons"]["restoration_vs_attack_null"])
        x0 = 365 + panel * 840
        y0 = 205
        bar_width = 500
        write(draw, (x0, 150), MODEL_LABELS[model], 28, bold=True)
        for index, attack_class in enumerate(classes):
            y = y0 + index * 158
            n = arms[f"{attack_class}__baseline"]["n"]
            write(draw, (x0, y - 15), display[attack_class], 17, bold=True)
            write(draw, (x0 + 190, y - 15), f"n={n}", 14, GRAY)
            values = [
                ("untreated rerun", float(arms[f"{attack_class}__baseline"]["comply"]), GRAY),
                ("successful-attack state", float(arms[f"{attack_class}__attack_null"]["comply"]), CONTROL),
                ("refusing-control state", float(arms[f"{attack_class}__harmful"]["comply"]), BLUE),
                ("equal-size control", float(arms[f"{attack_class}__harmful_sham"]["comply"]), "#d9c6b6"),
            ]
            for row, (label, value, color) in enumerate(values):
                yy = y + row * 31
                draw.rectangle((x0, yy, x0 + bar_width, yy + 23), fill=LIGHT)
                draw.rectangle((x0, yy, x0 + bar_width * value, yy + 23), fill=color)
                if panel == 0:
                    write(draw, (x0 - 10, yy + 11), label, 12, GRAY, anchor="ra")
                write(draw, (x0 + min(bar_width * value + 7, bar_width - 5), yy + 11), f"{value:.0%}", 13, INK, bold=True, anchor="lm")
        for tick in [0, 0.25, 0.5, 0.75, 1.0]:
            x = x0 + bar_width * tick
            write(draw, (x, 880), f"{tick:.0%}", 14, GRAY, anchor="ma")

    image.save(OUT / "fig6_closure_restoration.png", dpi=(240, 240))


def generation_validation() -> None:
    """Summarize the behavioral validation with reviewer-facing scope labels."""
    width, height = 1800, 1260
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    write(draw, (60, 42), "Full-generation causal validation", 36, bold=True)
    write(draw, (60, 90), "100 disjoint prompts per population; fitted direction versus matched random direction", 21, GRAY)

    columns = [
        ("Source-local removal", "local_ablation_harmful", "Harmful compliance"),
        ("All-write removal", "global_ablation_harmful", "Harmful compliance"),
        ("Direction addition", "addition_harmless", "Harmless-prompt refusal"),
    ]
    panel_x = [310, 830, 1350]
    panel_width = 390
    plot_width = 325
    role_labels = {
        "arditi_global": "all-write choice",
        "source_local": "source-local choice",
        "registry": "fixed scope-test site",
    }

    for column, (title, _suffix, measure) in enumerate(columns):
        write(draw, (panel_x[column] + panel_width / 2, 145), title, 26, bold=True, anchor="ma")
        write(draw, (panel_x[column] + panel_width / 2, 178), measure, 15, GRAY, anchor="ma")

    for row, model in enumerate(["gemma-3-4b", "qwen3-4b"]):
        report = load_json(ROOT / "results" / "arditi_gap" / model / "generation_validation_report.json")
        conditions = report["conditions"]
        y_top = 250 + row * 485
        baseline_compliance = conditions["baseline_harmful"]["llm_compliance_rate"]
        baseline_refusal = conditions["baseline_harmless"]["lexical_refusal_rate"]
        write(draw, (60, y_top - 32), model, 28, bold=True)
        write(
            draw,
            (60, y_top + 5),
            f"untreated: {baseline_compliance:.0%} harmful compliance; {baseline_refusal:.0%} harmless refusal",
            17,
            GRAY,
        )

        for column, (_title, suffix, _measure) in enumerate(columns):
            x0 = panel_x[column]
            for tick in [0, 0.25, 0.5, 0.75, 1.0]:
                x = x0 + tick * plot_width
                draw.line((x, y_top + 42, x, y_top + 350), fill="#dfe4e8", width=1)
                write(draw, (x, y_top + 370), f"{tick:.0%}", 14, GRAY, anchor="ma")

            for index, site in enumerate(report["sites"]):
                role = site["role"]
                layer = site["layer"]
                position = site["position"]
                slug = f"{role}_L{layer}_p{position}"
                real = conditions[f"{slug}_{suffix}"]
                random = conditions[f"{slug}_random_{suffix}"]
                metric = "lexical_refusal_rate" if suffix == "addition_harmless" else "llm_compliance_rate"
                real_rate = float(real[metric])
                random_rate = float(random[metric])
                y = y_top + 65 + index * 96
                label = f"{role_labels[role]} L{layer}@{position}"
                write(draw, (x0 - 12, y + 17), label, 14, anchor="ra")
                draw.rectangle((x0, y, x0 + plot_width, y + 22), fill=LIGHT)
                draw.rectangle((x0, y, x0 + plot_width * real_rate, y + 22), fill=BLUE)
                draw.rectangle((x0, y + 29, x0 + plot_width, y + 47), fill="#f0f2f4")
                draw.rectangle((x0, y + 29, x0 + plot_width * random_rate, y + 47), fill=CONTROL)
                real_color = "white" if real_rate >= 0.18 else INK
                write(draw, (x0 + 5, y + 11), f"fitted {real_rate:.0%}", 13, real_color, bold=True, anchor="lm")
                write(
                    draw,
                    (x0 + max(5, plot_width * random_rate + 5), y + 38),
                    f"random {random_rate:.0%}",
                    13,
                    GRAY,
                    anchor="lm",
                )
                if real["incoherent"] or random["incoherent"]:
                    warning_color = "#b9403b" if real["incoherent"] >= 10 else GRAY
                    write(
                        draw,
                        (x0 + plot_width, y + 65),
                        f"incoherent: fitted {real['incoherent']} / random {random['incoherent']}",
                        12,
                        warning_color,
                        anchor="ra",
                    )
        draw.line((55, y_top + 420, 1745, y_top + 420), fill="#c8d1d8", width=2)

    write(
        draw,
        (60, 1210),
        "Gemma L11 all-write removal is not interpreted: 78 of 100 generations were repetitive or degenerate.",
        18,
        "#b9403b",
        bold=True,
    )
    image.save(OUT / "fig2_generation_validation.png", dpi=(240, 240))


def copy_existing() -> None:
    sources = {
        "fig1_direction_auc.png": ROOT / "results" / "precomputed" / "direction_validation_heatmaps.png",
        "figA1_gemma_causal_sweep.png": ROOT / "results" / "arditi_gap" / "gemma-3-4b" / "causal_heatmaps.png",
        "figA2_qwen_causal_sweep.png": ROOT / "results" / "arditi_gap" / "qwen3-4b" / "causal_heatmaps.png",
    }
    for name, source in sources.items():
        shutil.copy2(source, OUT / name)


def main() -> None:
    copy_existing()
    method_overview()
    refusal_trajectories()
    scope_point_plot()
    causal_contrast()
    jailbreak_examples()
    intervention_examples()
    generation_validation()
    jailbreak_dataset()
    refusal_signal_by_class()
    paper_closure_scope()
    sustained_addition()
    paper_closure_transplant()
    paper_closure_restoration()


if __name__ == "__main__":
    main()
