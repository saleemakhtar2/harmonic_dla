"""Generate a non-technical, plain-English guide to the harmonic-DLA paper."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer

FIGURES = (
    "growing_pile.png",
    "safety_fence.png",
    "lost_direction.png",
    "moved_centre.png",
    "batch_checkpoints.png",
)

NAVY = "#173F5F"
BLUE = "#20639B"
TEAL = "#3CAEA3"
ORANGE = "#F6A01A"
RED = "#ED553B"
PALE_BLUE = "#E5EEF5"
PALE_TEAL = "#D9F0ED"
PALE_ORANGE = "#FFF2D9"


def report_text() -> str:
    """Return the reader-language contract used by the report."""
    return """
    A tiny story about a growing pile. A random wanderer helps grow a branching
    pile. A safety fence keeps the journey manageable. A gold-standard return
    rule uses the direction in which the wanderer left. A random restart can
    forget that helpful clue. The paper's key idea moves the fence to a better
    balance point and checks that point in batches. The gold-standard rule is
    the best choice when it is available. The method does not promise an
    identical final picture. The paper in one minute: improve a useful shortcut
    without pretending it replaces the gold-standard method.
    """


def _save(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _pile(axis: Any, *, offset_x: float = 0.0, scale: float = 1.0) -> None:
    points = [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 2),
        (4, 2),
        (1, -1),
        (1, -2),
    ]
    for x, y in points:
        axis.add_patch(
            Circle(
                (offset_x + scale * x, scale * y),
                0.36 * scale,
                facecolor=TEAL,
                edgecolor="white",
                lw=1.2,
            )
        )


def _plot_growing_pile(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
    _pile(axis, offset_x=-1.2)
    path_points = np.asarray(
        [(-4.0, 2.8), (-3.1, 2.5), (-2.0, 2.9), (-1.3, 2.2), (-0.4, 2.5), (0.4, 1.9), (1.0, 1.5)]
    )
    axis.plot(path_points[:, 0], path_points[:, 1], color=ORANGE, lw=2.2, ls="--")
    axis.scatter(
        path_points[0, 0], path_points[0, 1], s=95, color=ORANGE, edgecolor="white", zorder=4
    )
    axis.annotate(
        "a wandering dot",
        xy=path_points[0],
        xytext=(-4.6, 3.8),
        arrowprops={"arrowstyle": "->", "color": ORANGE},
        color=NAVY,
        fontsize=11,
    )
    axis.annotate(
        "it sticks when it reaches the pile",
        xy=(1.0, 1.5),
        xytext=(2.0, 3.2),
        arrowprops={"arrowstyle": "->", "color": NAVY},
        color=NAVY,
        fontsize=11,
    )
    axis.text(
        1.0,
        -3.0,
        "Repeat this many times and a branching shape appears.",
        ha="center",
        color=NAVY,
        fontsize=12,
        fontweight="bold",
    )
    axis.set_xlim(-5.2, 6.0)
    axis.set_ylim(-3.5, 4.5)
    axis.set_aspect("equal")
    axis.axis("off")
    _save(figure, path)


def _plot_safety_fence(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
    _pile(axis, offset_x=-1.8, scale=0.72)
    axis.add_patch(Circle((0.0, 0.0), 3.0, fill=False, ec=BLUE, lw=2.5))
    axis.add_patch(Circle((0.0, 0.0), 4.35, fill=False, ec=ORANGE, lw=2.8, ls="--"))
    axis.annotate(
        "inner starting ring",
        xy=(0.0, 3.0),
        xytext=(3.5, 2.6),
        arrowprops={"arrowstyle": "->", "color": BLUE},
        color=NAVY,
        fontsize=11,
    )
    axis.annotate(
        "safety fence",
        xy=(0.0, 4.35),
        xytext=(3.3, 4.2),
        arrowprops={"arrowstyle": "->", "color": ORANGE},
        color=NAVY,
        fontsize=11,
        fontweight="bold",
    )
    axis.text(
        0.0,
        -5.4,
        "When a wanderer reaches the fence, the computer needs a practical next step.",
        ha="center",
        color=NAVY,
        fontsize=12,
        fontweight="bold",
    )
    axis.set_xlim(-5.2, 5.6)
    axis.set_ylim(-5.8, 5.4)
    axis.set_aspect("equal")
    axis.axis("off")
    _save(figure, path)


def _plot_lost_direction(path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.5), constrained_layout=True)
    for axis, title, remembered in zip(
        axes, ("Keep the clue", "Forget the clue"), (True, False), strict=True
    ):
        _pile(axis, offset_x=-1.2, scale=0.62)
        axis.add_patch(Circle((0.0, 0.0), 3.2, fill=False, ec=BLUE, lw=2.2))
        exit_point = (3.2, 0.7)
        axis.scatter(*exit_point, s=70, color=RED, zorder=4)
        if remembered:
            arrow = FancyArrowPatch(
                exit_point,
                (2.25, 0.55),
                connectionstyle="arc3,rad=-0.35",
                arrowstyle="->",
                mutation_scale=14,
                color=TEAL,
                lw=2.5,
            )
            axis.add_patch(arrow)
            caption = "The return uses the side where the wanderer left."
        else:
            axis.scatter(-2.5, 1.8, s=65, color=ORANGE, zorder=4)
            axis.annotate(
                "fresh random start",
                xy=(-2.5, 1.8),
                xytext=(-3.5, 3.1),
                arrowprops={"arrowstyle": "->", "color": ORANGE},
                color=NAVY,
                fontsize=9,
            )
            caption = "A random restart forgets which side mattered."
        axis.set_title(title, color=NAVY, fontsize=13, fontweight="bold")
        axis.text(0.0, -4.05, caption, ha="center", color=NAVY, fontsize=10, wrap=True)
        axis.set_xlim(-4.2, 4.2)
        axis.set_ylim(-4.5, 4.0)
        axis.set_aspect("equal")
        axis.axis("off")
    _save(figure, path)


def _plot_moved_centre(path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.7), constrained_layout=True)
    for axis, title, centre, color, note in zip(
        axes,
        ("An arbitrary centre", "A better balance point"),
        ((0.0, 0.0), (0.75, 0.25)),
        (RED, TEAL),
        (
            "The fence is not lined up with the lopsided pile.",
            "The fence follows where the shortcut tends to attach.",
        ),
        strict=True,
    ):
        _pile(axis, offset_x=-1.45, scale=0.68)
        axis.add_patch(Circle(centre, 3.05, fill=False, ec=color, lw=2.7))
        axis.scatter(*centre, s=85, color=color, edgecolor="white", zorder=4)
        axis.set_title(title, color=NAVY, fontsize=13, fontweight="bold")
        axis.text(0.0, -4.0, note, ha="center", color=NAVY, fontsize=10, wrap=True)
        axis.set_xlim(-4.3, 4.3)
        axis.set_ylim(-4.5, 4.2)
        axis.set_aspect("equal")
        axis.axis("off")
    _save(figure, path)


def _plot_batch_checkpoints(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 3.4), constrained_layout=True)
    axis.axis("off")
    checkpoints = (1.0, 3.2, 5.4, 7.6)
    for index, x in enumerate(checkpoints, start=1):
        axis.add_patch(
            plt.Rectangle((x - 0.72, 1.25), 1.44, 1.15, facecolor=PALE_BLUE, edgecolor=BLUE, lw=1.6)
        )
        axis.text(
            x,
            1.82,
            f"check {index}",
            ha="center",
            va="center",
            color=NAVY,
            fontsize=11,
            fontweight="bold",
        )
        if index < len(checkpoints):
            axis.annotate(
                "",
                xy=(x + 1.4, 2.72),
                xytext=(x + 0.8, 2.72),
                arrowprops={"arrowstyle": "->", "lw": 2, "color": TEAL},
            )
            axis.text(x + 1.1, 2.92, "reuse for a while", color=TEAL, fontsize=8.5, ha="center")
    axis.text(
        4.3,
        3.15,
        "Checking the balance point in batches",
        ha="center",
        color=NAVY,
        fontsize=15,
        fontweight="bold",
    )
    axis.text(
        4.3,
        0.45,
        "The paper shows the balance point cannot race away after one new dot is added.",
        ha="center",
        color=NAVY,
        fontsize=11,
    )
    axis.set_xlim(0.0, 8.7)
    axis.set_ylim(0.0, 3.7)
    _save(figure, path)


def _make_figures(directory: Path) -> dict[str, Path]:
    paths = {name: directory / name for name in FIGURES}
    _plot_growing_pile(paths["growing_pile.png"])
    _plot_safety_fence(paths["safety_fence.png"])
    _plot_lost_direction(paths["lost_direction.png"])
    _plot_moved_centre(paths["moved_centre.png"])
    _plot_batch_checkpoints(paths["batch_checkpoints.png"])
    return paths


def _page_decor(canvas: Any, _document: Any) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#D9E2EC"))
    canvas.line(0.55 * inch, height - 0.35 * inch, width - 0.55 * inch, height - 0.35 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#718096"))
    canvas.drawString(0.55 * inch, 0.25 * inch, "harmonic-dla | a friendly guide")
    canvas.drawRightString(width - 0.55 * inch, 0.25 * inch, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def _styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="Cover",
            parent=styles["Title"],
            fontSize=26,
            leading=31,
            alignment=TA_CENTER,
            textColor=colors.HexColor(NAVY),
            spaceAfter=15,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subtitle",
            parent=styles["Normal"],
            fontSize=12,
            leading=17,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4A5568"),
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H1",
            parent=styles["Heading1"],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor(NAVY),
            spaceBefore=6,
            spaceAfter=9,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H1Long",
            parent=styles["Heading1"],
            fontSize=16,
            leading=20,
            textColor=colors.HexColor(NAVY),
            spaceBefore=6,
            spaceAfter=9,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontSize=10.3,
            leading=15.3,
            alignment=TA_LEFT,
            spaceAfter=9,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Callout",
            parent=styles["BodyText"],
            fontSize=11.0,
            leading=16,
            alignment=TA_CENTER,
            textColor=colors.HexColor(NAVY),
            borderColor=colors.HexColor(TEAL),
            borderWidth=0.8,
            borderPadding=10,
            backColor=colors.HexColor("#EFFAF8"),
            spaceBefore=8,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontSize=8.2,
            leading=11.3,
            textColor=colors.HexColor("#4A5568"),
            spaceAfter=4,
        )
    )
    return styles


def build_report(output: Path, figure_dir: Path) -> None:
    """Build the six-page plain-English PDF and its locally drawn figures."""
    output.parent.mkdir(parents=True, exist_ok=True)
    figures = _make_figures(figure_dir)
    styles = _styles()
    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.52 * inch,
        bottomMargin=0.48 * inch,
    )
    story: list[Any] = [
        Spacer(1, 0.35 * inch),
        Paragraph("A friendly guide to a paper about growing shapes", styles["Cover"]),
        Paragraph(
            "No maths needed. Just a wandering dot, a growing pile, and a useful idea about where to put a safety fence.",
            styles["Subtitle"],
        ),
        Paragraph("A tiny story about a growing pile", styles["H1"]),
        Paragraph(
            "Imagine dropping a tiny dot into open space. It wanders in a completely unpredictable way. The first time it touches an existing pile, it sticks. Then another dot begins. Over time, this simple rule creates a shape that looks a little like coral, a snowflake, or a branching root.",
            styles["Body"],
        ),
        Image(str(figures["growing_pile.png"]), width=7.05 * inch, height=3.15 * inch),
        Paragraph(
            "This is a computer model of how a branching pile can grow from many tiny, random journeys.",
            styles["Callout"],
        ),
        PageBreak(),
        Paragraph("Why computers add a safety fence", styles["H1"]),
        Paragraph(
            "A wandering dot can travel very far away before it finds the pile. Following every distant part of every journey would take a long time. So a computer draws a big circular safety fence around the pile. When a dot reaches that fence, the computer uses a rule for what happens next.",
            styles["Body"],
        ),
        Image(str(figures["safety_fence.png"]), width=7.0 * inch, height=3.3 * inch),
        Paragraph(
            "One common shortcut throws the far-away dot away and starts a fresh one from the inner ring. It is easy to do, but it can lose a useful clue.",
            styles["Callout"],
        ),
        PageBreak(),
        Paragraph("The small mistake in a random restart", styles["H1"]),
        Paragraph(
            "Suppose the pile has grown more on the right than on the left. A dot that escaped on the right is not just a blank slate. If it were allowed to continue, that right-hand side would still matter. A random restart forgets where the dot escaped, so it can gently lean the answer in the wrong direction.",
            styles["Body"],
        ),
        Image(str(figures["lost_direction.png"]), width=7.05 * inch, height=2.95 * inch),
        Paragraph(
            "There is a gold-standard return rule that keeps this direction clue. It is the best choice when it is available. This paper is not trying to replace that rule in a system that can already use it.",
            styles["Callout"],
        ),
        PageBreak(),
        Paragraph("The paper's key idea: move the centre", styles["H1Long"]),
        Paragraph(
            "The shortcut does not have to use a fence centred on an arbitrary old starting point. Instead, the paper asks where the shortcut itself tends to add new dots. That location is its balance point. Move the fence so its centre sits at that balance point, and the biggest left-versus-right lean cancels out.",
            styles["Body"],
        ),
        Image(str(figures["moved_centre.png"]), width=7.05 * inch, height=3.1 * inch),
        Paragraph(
            "In everyday terms: if a lopsided pile keeps pulling attention to one side, do not keep measuring from the wrong middle.",
            styles["Callout"],
        ),
        PageBreak(),
        Paragraph("Checking in batches instead of every second", styles["H1"]),
        Paragraph(
            "Finding the balance point takes extra checking. Doing that full check after every single new dot would be wasteful. The paper proves something reassuring: when only one small dot is added, the best balance point cannot suddenly fly across the page.",
            styles["Body"],
        ),
        Image(str(figures["batch_checkpoints.png"]), width=7.05 * inch, height=2.65 * inch),
        Paragraph(
            "So the computer can check the balance point, reuse it for a while, then check again. This makes the extra checking grow much more gently as the pile gets bigger, while still keeping a written allowance for possible mistakes.",
            styles["Callout"],
        ),
        PageBreak(),
        Paragraph("The paper in one minute", styles["H1"]),
        Paragraph(
            "The paper gives a careful way to improve a useful shortcut for a computer simulation of a growing branching shape. It says: keep the safety fence centred on the shortcut's own balance point, and update that point in sensible batches rather than constantly.",
            styles["Body"],
        ),
        Paragraph(
            "What it does give: a trustworthy error allowance for this particular shortcut, and a way to avoid an explosion of extra checking work.",
            styles["Callout"],
        ),
        Paragraph(
            "What it does not promise: that every final picture will be identical, a cure for every possible computer error, or a reason to avoid the gold-standard return rule when that rule is easy to use.",
            styles["Body"],
        ),
        Paragraph(
            "The idea in one line: keep the simple shortcut honest by moving its centre to the right place, then checking that centre only as often as the growing pile really needs.",
            styles["Callout"],
        ),
        Spacer(1, 0.2 * inch),
        Paragraph(
            "This guide is an intuitive explanation of the paper, not a replacement for its technical proof. It deliberately leaves out the mathematical machinery so that the main idea is easy to see.",
            styles["Small"],
        ),
    ]
    document.build(story, onFirstPage=_page_decor, onLaterPages=_page_decor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure-dir", type=Path, default=Path("output/plain_english_paper/figures")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output/pdf/harmonic_dla_paper_plain_english.pdf")
    )
    args = parser.parse_args()
    build_report(args.output, args.figure_dir)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
