"""Generate the focused benchmark report for the paper's amortization result."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Flowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from scripts.benchmark_four_approaches import APPROACHES, _summaries

from harmonic_dla.schedules import AmortizedSchedule

NAVY = HexColor("#102A43")
BLUE = HexColor("#2F6BFF")
CYAN = HexColor("#20B8CD")
TEAL = HexColor("#0E8F78")
AMBER = HexColor("#F59E0B")
PURPLE = HexColor("#7357D9")
INK = HexColor("#243B53")
MUTED = HexColor("#627D98")
PALE = HexColor("#F3F7FB")
GRID = HexColor("#D9E2EC")
WHITE = colors.white

LABELS = {
    "exact": "Exact Poisson",
    "uniform": "Uniform restart",
    "controlled-fixed": "Per-step calibrated",
    "paper-amortized": "Paper-amortized",
}
COLORS = {
    "exact": BLUE,
    "uniform": AMBER,
    "controlled-fixed": TEAL,
    "paper-amortized": PURPLE,
}


def load_benchmark_inputs(paths: list[Path]) -> dict[str, Any]:
    """Merge process-isolated benchmark payloads and recompute their summaries."""
    if not paths:
        raise ValueError("at least one benchmark input is required")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    particles = {int(payload["methodology"]["particles"]) for payload in payloads}
    threads = {int(payload["environment"]["threads"]) for payload in payloads}
    if len(particles) != 1 or len(threads) != 1:
        raise ValueError("benchmark inputs must use the same particle count and thread count")
    records = [record for payload in payloads for record in payload["records"]]
    identities = [(record["approach"], int(record["seed"])) for record in records]
    if len(identities) != len(set(identities)):
        raise ValueError("benchmark inputs contain duplicate approach/seed records")
    seeds = sorted({int(record["seed"]) for record in records})
    expected = {(approach, seed) for approach in APPROACHES for seed in seeds}
    if set(identities) != expected:
        raise ValueError("benchmark inputs do not contain every approach for every seed")
    methodology = dict(payloads[0]["methodology"])
    methodology.update({"seeds": seeds, "repeats": len(seeds)})
    return {
        "methodology": methodology,
        "environment": payloads[0]["environment"],
        "records": records,
        "summaries": _summaries(records),
    }


def schedule_scaling(horizons: list[int]) -> list[dict[str, int | float]]:
    """Compare per-attachment sample splitting with the shipped paper schedule."""
    schedule = AmortizedSchedule(
        alpha=0.48,
        gamma=0.06,
        scale=32.0,
        calibration_failure_budget=1.0e-3,
        start=1024,
    )
    rows: list[dict[str, int | float]] = []
    for horizon in horizons:
        blocks = list(schedule.blocks(horizon))
        paper_probes = sum(block.probes for block in blocks)
        fixed_probes = max(0, horizon - 1) * 8192
        rows.append(
            {
                "horizon": horizon,
                "paper_events": len(blocks),
                "paper_probes": paper_probes,
                "fixed_probes": fixed_probes,
                "probe_reduction": fixed_probes / paper_probes if paper_probes else math.inf,
            }
        )
    return rows


def _styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "h1": ParagraphStyle(
            "H1",
            parent=sample["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=NAVY,
            spaceAfter=5 * mm,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=sample["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12.5,
            leading=15,
            textColor=BLUE,
            spaceBefore=3 * mm,
            spaceAfter=2.5 * mm,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=13.4,
            textColor=INK,
            spaceAfter=2.5 * mm,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=7.3,
            leading=9.5,
            textColor=MUTED,
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=sample["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=15,
            textColor=NAVY,
        ),
        "table": ParagraphStyle(
            "Table",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=7.3,
            leading=9.2,
            textColor=INK,
        ),
        "table_head": ParagraphStyle(
            "TableHead",
            parent=sample["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.3,
            leading=9.2,
            textColor=WHITE,
        ),
        "metric": ParagraphStyle(
            "Metric",
            parent=sample["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=21,
            alignment=TA_CENTER,
            textColor=NAVY,
        ),
        "metric_label": ParagraphStyle(
            "MetricLabel",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=7.1,
            leading=8.7,
            alignment=TA_CENTER,
            textColor=MUTED,
        ),
    }


def _p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def _table(
    rows: list[list[Any]],
    widths: list[float],
    style: dict[str, ParagraphStyle],
) -> Table:
    rendered = []
    for row_index, row in enumerate(rows):
        cell_style = style["table_head"] if row_index == 0 else style["table"]
        rendered.append(
            [value if isinstance(value, Flowable) else _p(str(value), cell_style) for value in row]
        )
    table = Table(rendered, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE]),
                ("GRID", (0, 0), (-1, -1), 0.35, GRID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 1.7 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.7 * mm),
            ]
        )
    )
    return table


def _metric_card(value: str, label: str, style: dict[str, ParagraphStyle]) -> Table:
    card = Table(
        [[_p(value, style["metric"])], [_p(label, style["metric_label"])]],
        colWidths=[42.5 * mm],
        rowHeights=[12 * mm, 10 * mm],
    )
    card.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE),
                ("BOX", (0, 0), (-1, -1), 0.5, GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return card


class HorizontalBars(Flowable):
    def __init__(
        self,
        values: list[tuple[str, float, colors.Color]],
        *,
        width: float,
        height: float,
        formatter: str,
        suffix: str = "",
        log_scale: bool = False,
    ) -> None:
        super().__init__()
        self.values = values
        self.width = width
        self.height = height
        self.formatter = formatter
        self.suffix = suffix
        self.log_scale = log_scale

    def draw(self) -> None:
        left = 42 * mm
        right = 27 * mm
        chart_width = self.width - left - right
        row_height = self.height / len(self.values)
        transformed = [
            math.log10(value + 1.0) if self.log_scale else value
            for _label, value, _color in self.values
        ]
        maximum = max(transformed) or 1.0
        for index, ((label, value, color), scaled) in enumerate(
            zip(self.values, transformed, strict=True)
        ):
            y = self.height - (index + 0.68) * row_height
            self.canv.setFont("Helvetica", 8.5)
            self.canv.setFillColor(INK)
            self.canv.drawRightString(left - 3 * mm, y + 1.8 * mm, label)
            self.canv.setFillColor(PALE)
            self.canv.roundRect(left, y, chart_width, 5.5 * mm, 2.5 * mm, fill=1, stroke=0)
            self.canv.setFillColor(color)
            self.canv.roundRect(
                left,
                y,
                chart_width * scaled / maximum,
                5.5 * mm,
                2.5 * mm,
                fill=1,
                stroke=0,
            )
            self.canv.setFont("Helvetica-Bold", 8.5)
            self.canv.setFillColor(INK)
            self.canv.drawString(
                left + chart_width + 2 * mm,
                y + 1.8 * mm,
                self.formatter.format(value) + self.suffix,
            )


class ScalingChart(Flowable):
    def __init__(self, rows: list[dict[str, int | float]], width: float, height: float) -> None:
        super().__init__()
        self.rows = rows
        self.width = width
        self.height = height

    def draw(self) -> None:
        left, right, bottom, top = 18 * mm, 8 * mm, 13 * mm, 8 * mm
        width = self.width - left - right
        height = self.height - bottom - top
        x_values = [math.log10(float(row["horizon"])) for row in self.rows]
        all_probes = [
            float(row[key]) for row in self.rows for key in ("fixed_probes", "paper_probes")
        ]
        y_values = [math.log10(max(1.0, value)) for value in all_probes]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = math.floor(min(y_values)), math.ceil(max(y_values))

        def px(value: float) -> float:
            return left + width * (value - x_min) / (x_max - x_min)

        def py(value: float) -> float:
            return bottom + height * (value - y_min) / (y_max - y_min)

        self.canv.setFont("Helvetica", 7)
        for exponent in range(y_min, y_max + 1):
            y = py(float(exponent))
            self.canv.setStrokeColor(GRID)
            self.canv.line(left, y, left + width, y)
            self.canv.setFillColor(MUTED)
            self.canv.drawRightString(left - 2 * mm, y - 1, f"10^{exponent}")
        for row, x in zip(self.rows, x_values, strict=True):
            x_pos = px(x)
            self.canv.setFillColor(MUTED)
            self.canv.drawCentredString(x_pos, bottom - 4 * mm, f"{int(row['horizon']):,}")
        for key, color, label in (
            ("fixed_probes", TEAL, "Per-step calibrated"),
            ("paper_probes", PURPLE, "Paper-amortized"),
        ):
            points = [
                (px(x), py(math.log10(max(1.0, float(row[key])))))
                for row, x in zip(self.rows, x_values, strict=True)
            ]
            self.canv.setStrokeColor(color)
            self.canv.setLineWidth(2)
            path = self.canv.beginPath()
            path.moveTo(*points[0])
            for point in points[1:]:
                path.lineTo(*point)
            self.canv.drawPath(path)
            self.canv.setFillColor(color)
            for point in points:
                self.canv.circle(*point, 2, fill=1, stroke=0)
            legend_x = left + (0 if key == "fixed_probes" else 62 * mm)
            self.canv.rect(legend_x, self.height - 4 * mm, 4 * mm, 2 * mm, fill=1, stroke=0)
            self.canv.setFillColor(INK)
            self.canv.drawString(legend_x + 6 * mm, self.height - 4.5 * mm, label)


def _header_footer(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    if doc.page > 1:
        canvas.setStrokeColor(GRID)
        canvas.line(18 * mm, A4[1] - 15 * mm, A4[0] - 18 * mm, A4[1] - 15 * mm)
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.setFillColor(NAVY)
        canvas.drawString(18 * mm, A4[1] - 11 * mm, "HARMONIC-DLA PAPER SHOWCASE")
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, str(doc.page))
    canvas.restoreState()


def _cover(canvas: Any, _doc: Any, particles: int, seeds: list[int]) -> None:
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
    canvas.setFillColor(PURPLE)
    canvas.circle(A4[0] - 19 * mm, A4[1] - 28 * mm, 46 * mm, fill=1, stroke=0)
    canvas.setFillColor(CYAN)
    canvas.circle(A4[0] - 32 * mm, 20 * mm, 28 * mm, fill=1, stroke=0)
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 28)
    canvas.drawString(22 * mm, A4[1] - 70 * mm, "harmonic-dla")
    canvas.setFont("Helvetica-Bold", 21)
    canvas.drawString(22 * mm, A4[1] - 86 * mm, "Why the theorem matters")
    canvas.setFont("Helvetica", 11)
    canvas.setFillColor(HexColor("#D9EAF7"))
    canvas.drawString(22 * mm, A4[1] - 102 * mm, "A post-fix benchmark of four restart policies")
    canvas.setFont("Helvetica-Bold", 10)
    canvas.setFillColor(WHITE)
    canvas.drawString(22 * mm, 33 * mm, "Measured engineering showcase")
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(HexColor("#D9EAF7"))
    canvas.drawString(
        22 * mm,
        26 * mm,
        f"{particles:,} particles x {len(seeds)} seeds | 4 Numba threads",
    )
    canvas.drawString(22 * mm, 20 * mm, "26 July 2026")
    canvas.restoreState()


def build_report(data: dict[str, Any], output: Path) -> None:
    """Render a focused, evidence-backed paper showcase PDF."""
    style = _styles()
    summaries = data["summaries"]
    methodology = data["methodology"]
    environment = data["environment"]
    fixed = summaries["controlled-fixed"]
    paper = summaries["paper-amortized"]
    exact = summaries["exact"]
    fixed_to_paper_time = fixed["runtime_median_seconds"] / paper["runtime_median_seconds"]
    probe_reduction = fixed["calibration_probes_median"] / paper["calibration_probes_median"]
    paper_to_exact_time = paper["runtime_median_seconds"] / exact["runtime_median_seconds"]
    infinite_bounds = [
        float(record["infinite_history_bound_upper"])
        for record in data["records"]
        if record["approach"] == "paper-amortized"
    ]
    infinite_bound = statistics.median(infinite_bounds)
    scaling = schedule_scaling([1200, 3000, 10_000, 30_000, 100_000, 1_000_000])

    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=22 * mm,
        bottomMargin=17 * mm,
        title="harmonic-dla: Why the theorem matters",
        author="OpenAI Codex",
        subject="Post-fix benchmark of the paper-amortized restart schedule",
    )
    story: list[Flowable] = [Spacer(1, 230 * mm), PageBreak()]

    story.extend(
        [
            _p("The point, in plain English", style["h1"]),
            _p(
                "<b>The package</b> simulates two-dimensional off-lattice diffusion-limited "
                "aggregation. Walkers repeatedly approach a growing cluster. When they escape to "
                "an outer boundary, the simulator must decide how to return them without distorting "
                "where the next particle attaches.",
                style["body"],
            ),
            _p(
                "<b>The paper</b> shows that centering the restart boundary near the cluster's "
                "harmonic-measure center cancels the leading restart bias. Its new operational "
                "insight is that this center drifts slowly enough to be recalibrated over growing "
                "blocks, with an explicit error budget, instead of paying for a fresh statistical "
                "calibration after every attachment.",
                style["body"],
            ),
            _p(
                "<b>The practical win is certified amortization.</b> It is not a claim that an "
                "approximate restart beats the exact circular Poisson return in raw speed. On this "
                "implementation, exact return remains the best operational default.",
                style["callout"],
            ),
            Spacer(1, 6 * mm),
        ]
    )
    cards = Table(
        [
            [
                _metric_card(f"{probe_reduction:,.0f}x", "Fewer calibration probes", style),
                _metric_card(
                    f"{fixed_to_paper_time:,.0f}x", "Faster than per-step calibration", style
                ),
                _metric_card(
                    f"{paper['conditional_path_bound_median']:.4f}",
                    "Measured finite path bound",
                    style,
                ),
                _metric_card(f"{infinite_bound:.3f}", "Infinite-history upper bound", style),
            ]
        ],
        colWidths=[43.5 * mm] * 4,
    )
    cards.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.extend(
        [
            cards,
            Spacer(1, 7 * mm),
            _p("Four approaches", style["h2"]),
            _table(
                [
                    ["Approach", "Restart rule", "What it is for"],
                    [
                        "Exact Poisson",
                        "Exact exterior return angle",
                        "Fast, unbiased reference and recommended default.",
                    ],
                    [
                        "Uniform restart",
                        "Uniform relaunch on the boundary",
                        "Legacy comparison; directional bias is not certified.",
                    ],
                    [
                        "Per-step calibrated",
                        "Fresh sample-split center for every attachment",
                        "Correct but deliberately expensive control baseline.",
                    ],
                    [
                        "Paper-amortized",
                        "Exact prefix, then drift-budgeted growing blocks",
                        "Demonstrates the theorem's reusable-center schedule.",
                    ],
                ],
                [38 * mm, 57 * mm, 80 * mm],
                style,
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            _p("Measured benchmark", style["h1"]),
            _p(
                f"Each mode grew {methodology['particles']:,} particles for seeds "
                f"{', '.join(map(str, methodology['seeds']))}. Every approach/seed ran in a fresh "
                "process; its Numba path was warmed first, and compilation time was excluded. "
                "The paper schedule begins with an exact prefix through particle 1,024, so this "
                "horizon exercises both the prefix and calibrated blocks.",
                style["body"],
            ),
            _p("Median wall time (log-scaled bars; lower is better)", style["h2"]),
            HorizontalBars(
                [
                    (LABELS[name], summaries[name]["runtime_median_seconds"], COLORS[name])
                    for name in APPROACHES
                ],
                width=174 * mm,
                height=52 * mm,
                formatter="{:.3f}",
                suffix=" s",
                log_scale=True,
            ),
            Spacer(1, 4 * mm),
            _p("Median calibration probes (log-scaled bars; lower is better)", style["h2"]),
            HorizontalBars(
                [
                    (
                        LABELS[name],
                        float(summaries[name]["calibration_probes_median"]),
                        COLORS[name],
                    )
                    for name in APPROACHES
                ],
                width=174 * mm,
                height=52 * mm,
                formatter="{:,.0f}",
                log_scale=True,
            ),
            Spacer(1, 5 * mm),
            _table(
                [
                    ["Approach", "Median time", "Throughput", "Probes", "Events", "Path bound"],
                    *[
                        [
                            LABELS[name],
                            f"{summaries[name]['runtime_median_seconds']:.4f} s",
                            f"{summaries[name]['throughput_median_particles_per_second']:,.0f}/s",
                            f"{summaries[name]['calibration_probes_median']:,.0f}",
                            f"{summaries[name]['calibration_events_median']:,.0f}",
                            (
                                "none"
                                if summaries[name]["conditional_path_bound_median"] is None
                                else f"{summaries[name]['conditional_path_bound_median']:.4f}"
                            ),
                        ]
                        for name in APPROACHES
                    ],
                ],
                [36 * mm, 28 * mm, 31 * mm, 30 * mm, 22 * mm, 28 * mm],
                style,
            ),
            Spacer(1, 5 * mm),
            _p(
                f"The paper-amortized run used {probe_reduction:,.0f}x fewer probes and was "
                f"{fixed_to_paper_time:,.0f}x faster than fresh per-attachment calibration. It "
                f"still cost {paper_to_exact_time:.1f}x the exact baseline, which is why the "
                "finding should be described as making controlled certification practical - not "
                "as replacing exact return for this circular geometry.",
                style["callout"],
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            _p("How the advantage scales", style["h1"]),
            _p(
                "The plot applies the shipped theorem parameters (alpha 0.48, gamma 0.06, scale 32, "
                "start 1,024, failure budget 0.001). The per-step baseline uses 4,096 center-search "
                "and 4,096 validation probes after each attachment. Counts below are deterministic "
                "schedule arithmetic, not extrapolated wall-clock timings.",
                style["body"],
            ),
            ScalingChart(scaling, width=174 * mm, height=88 * mm),
            Spacer(1, 5 * mm),
            _table(
                [
                    ["Horizon", "Paper events", "Paper probes", "Per-step probes", "Reduction"],
                    *[
                        [
                            f"{int(row['horizon']):,}",
                            f"{int(row['paper_events']):,}",
                            f"{int(row['paper_probes']):,}",
                            f"{int(row['fixed_probes']):,}",
                            f"{float(row['probe_reduction']):,.0f}x",
                        ]
                        for row in scaling
                    ],
                ],
                [30 * mm, 32 * mm, 37 * mm, 43 * mm, 33 * mm],
                style,
            ),
            Spacer(1, 5 * mm),
            _p(
                "The theorem changes the calibration schedule qualitatively: block lengths grow "
                "with cluster size, failure probabilities are summable, and the death radius grows "
                "fast enough for the accumulated restart error to remain bounded. The resulting "
                "extra probe count is sublinear in the growth horizon under the paper's parameter "
                "conditions.",
                style["callout"],
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            _p("What the result proves - and what it does not", style["h1"]),
            _p("Supported by this run", style["h2"]),
            _table(
                [
                    ["Claim", "Evidence"],
                    [
                        "Amortization removes the calibration bottleneck.",
                        f"Median probes fell from {fixed['calibration_probes_median']:,.0f} to "
                        f"{paper['calibration_probes_median']:,.0f}; median runtime fell from "
                        f"{fixed['runtime_median_seconds']:.2f} s to "
                        f"{paper['runtime_median_seconds']:.3f} s.",
                    ],
                    [
                        "The run retains nontrivial certificate accounting.",
                        f"Median conditional finite-path bound "
                        f"{paper['conditional_path_bound_median']:.6f}; analytic infinite-history "
                        f"upper bound {infinite_bound:.6f}.",
                    ],
                    [
                        "Exact return is still the operational winner.",
                        f"Exact median runtime {exact['runtime_median_seconds']:.4f} s versus "
                        f"{paper['runtime_median_seconds']:.4f} s for paper-amortized.",
                    ],
                ],
                [67 * mm, 108 * mm],
                style,
            ),
            Spacer(1, 6 * mm),
            _p("Not established by this run", style["h2"]),
            _p(
                "Three stochastic seeds are enough to expose an order-of-magnitude engineering "
                "gap, but not to prove aggregate-law or morphology parity. A scientific comparison "
                "still needs a preregistered multi-horizon ensemble, common morphology statistics, "
                "confidence intervals, and tolerance-convergence checks. The certificate concerns "
                "restart-law approximation; finite walk-on-spheres numerical tolerance is a "
                "separate source of error.",
                style["body"],
            ),
            _p(
                "<b>Bottom line:</b> the package turns the paper into executable machinery. The "
                "showcase demonstrates the new finding exactly where it should: it makes rigorous "
                "controlled restart hundreds of times cheaper than naive recalibration, while "
                "preserving explicit path-law accounting.",
                style["callout"],
            ),
            Spacer(1, 8 * mm),
            _p("Reproducibility", style["h2"]),
            _table(
                [
                    ["Field", "Value"],
                    ["Platform", environment["platform"]],
                    [
                        "Python / NumPy / Numba",
                        f"{environment['python']} / {environment['numpy']} / {environment['numba']}",
                    ],
                    ["Threads", environment["threads"]],
                    ["Benchmark driver", "scripts/benchmark_four_approaches.py"],
                    ["Report generator", "scripts/generate_paper_showcase_report.py"],
                    ["Merged raw results", "output/benchmark/post_fix_showcase_1200.json"],
                ],
                [50 * mm, 125 * mm],
                style,
            ),
            Spacer(1, 5 * mm),
            _p(
                "Run command: .venv/bin/python scripts/benchmark_four_approaches.py "
                "--particles 1200 --seeds 42 137 911 --threads 4",
                style["small"],
            ),
        ]
    )

    doc.build(
        story,
        onFirstPage=lambda canvas, document: _cover(
            canvas,
            document,
            int(methodology["particles"]),
            list(methodology["seeds"]),
        ),
        onLaterPages=_header_footer,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--merged-output",
        type=Path,
        default=Path("output/benchmark/post_fix_showcase_1200.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/pdf/harmonic_dla_paper_finding_showcase.pdf"),
    )
    args = parser.parse_args()
    data = load_benchmark_inputs(args.input)
    args.merged_output.parent.mkdir(parents=True, exist_ok=True)
    args.merged_output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    build_report(data, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
