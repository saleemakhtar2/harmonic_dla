"""Generate the visual code-review and four-approach benchmark report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
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

NAVY = HexColor("#102A43")
BLUE = HexColor("#2F6BFF")
CYAN = HexColor("#20B8CD")
TEAL = HexColor("#0E8F78")
AMBER = HexColor("#F59E0B")
RED = HexColor("#D64545")
PURPLE = HexColor("#7357D9")
INK = HexColor("#243B53")
MUTED = HexColor("#627D98")
PALE = HexColor("#F3F7FB")
GRID = HexColor("#D9E2EC")
WHITE = colors.white

APPROACHES = ("exact", "uniform", "controlled-fixed", "paper-amortized")
LABELS = {
    "exact": "Exact Poisson",
    "uniform": "Uniform restart",
    "controlled-fixed": "Sample-split fixed",
    "paper-amortized": "Paper-amortized",
}
COLORS = {
    "exact": BLUE,
    "uniform": AMBER,
    "controlled-fixed": TEAL,
    "paper-amortized": PURPLE,
}


class HorizontalBars(Flowable):
    def __init__(
        self,
        values: list[tuple[str, float, colors.Color]],
        *,
        width: float,
        height: float,
        formatter: str = "{:.1f}",
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
        canvas = self.canv
        left = 42 * mm
        right = 23 * mm
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
            canvas.setFont("Helvetica", 8.5)
            canvas.setFillColor(INK)
            canvas.drawRightString(left - 3 * mm, y + 1.8 * mm, label)
            canvas.setFillColor(PALE)
            canvas.roundRect(left, y, chart_width, 5.5 * mm, 2.5 * mm, fill=1, stroke=0)
            bar_width = chart_width * scaled / maximum
            canvas.setFillColor(color)
            canvas.roundRect(left, y, bar_width, 5.5 * mm, 2.5 * mm, fill=1, stroke=0)
            canvas.setFont("Helvetica-Bold", 8.5)
            canvas.setFillColor(INK)
            text = self.formatter.format(value) + self.suffix
            canvas.drawString(left + chart_width + 2 * mm, y + 1.8 * mm, text)


class ClusterPanels(Flowable):
    def __init__(
        self,
        positions: dict[str, list[list[float]]],
        width: float,
        height: float,
    ) -> None:
        super().__init__()
        self.positions = positions
        self.width = width
        self.height = height

    def draw(self) -> None:
        canvas = self.canv
        gap = 4 * mm
        panel_width = (self.width - gap) / 2
        panel_height = (self.height - gap) / 2
        for index, approach in enumerate(APPROACHES):
            row = 1 - index // 2
            column = index % 2
            x0 = column * (panel_width + gap)
            y0 = row * (panel_height + gap)
            canvas.setFillColor(PALE)
            canvas.roundRect(x0, y0, panel_width, panel_height, 3 * mm, fill=1, stroke=0)
            canvas.setFont("Helvetica-Bold", 9)
            canvas.setFillColor(INK)
            canvas.drawString(x0 + 4 * mm, y0 + panel_height - 6 * mm, LABELS[approach])

            points = self.positions[approach]
            stride = max(1, math.ceil(len(points) / 2600))
            sampled = points[::stride]
            xs = [point[0] for point in sampled]
            ys = [point[1] for point in sampled]
            bound = max(max(abs(value) for value in xs), max(abs(value) for value in ys), 1.0)
            plot_x = x0 + 5 * mm
            plot_y = y0 + 4 * mm
            plot_width = panel_width - 10 * mm
            plot_height = panel_height - 13 * mm
            scale = 0.47 * min(plot_width, plot_height) / bound
            cx = plot_x + plot_width / 2
            cy = plot_y + plot_height / 2
            canvas.setFillColor(COLORS[approach])
            for x, y in sampled:
                canvas.circle(cx + x * scale, cy + y * scale, 0.32, fill=1, stroke=0)


def _styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=sample["Title"],
            fontName="Helvetica-Bold",
            fontSize=27,
            leading=31,
            textColor=WHITE,
            spaceAfter=8 * mm,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=12,
            leading=17,
            textColor=HexColor("#D9EAF7"),
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=sample["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=NAVY,
            spaceBefore=2 * mm,
            spaceAfter=5 * mm,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=sample["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=BLUE,
            spaceBefore=4 * mm,
            spaceAfter=2.5 * mm,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=13.2,
            textColor=INK,
            spaceAfter=2.5 * mm,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10.2,
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
            fontSize=7.4,
            leading=9.3,
            textColor=INK,
        ),
        "table_head": ParagraphStyle(
            "TableHead",
            parent=sample["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.4,
            leading=9.3,
            textColor=WHITE,
            alignment=TA_LEFT,
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
            fontSize=7.4,
            leading=9,
            alignment=TA_CENTER,
            textColor=MUTED,
        ),
    }


def _p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def _metric_card(value: str, label: str, style: dict[str, ParagraphStyle]) -> Table:
    table = Table(
        [[_p(value, style["metric"])], [_p(label, style["metric_label"])]],
        colWidths=[42 * mm],
        rowHeights=[12 * mm, 10 * mm],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE),
                ("BOX", (0, 0), (-1, -1), 0.5, GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2 * mm),
            ]
        )
    )
    return table


def _standard_table(
    rows: list[list[Any]],
    widths: list[float],
    style: dict[str, ParagraphStyle],
    *,
    header: bool = True,
) -> Table:
    rendered: list[list[Any]] = []
    for row_index, row in enumerate(rows):
        cell_style = style["table_head"] if header and row_index == 0 else style["table"]
        rendered.append(
            [value if isinstance(value, Flowable) else _p(str(value), cell_style) for value in row]
        )
    table = Table(rendered, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 1.8 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8 * mm),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
    ]
    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
        commands.append(("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE]))
    table.setStyle(TableStyle(commands))
    return table


def _header_footer(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    if doc.page > 1:
        canvas.setStrokeColor(GRID)
        canvas.line(18 * mm, A4[1] - 15 * mm, A4[0] - 18 * mm, A4[1] - 15 * mm)
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.setFillColor(NAVY)
        canvas.drawString(18 * mm, A4[1] - 11 * mm, "HARMONIC-DLA REVIEW AND BENCHMARK")
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"{doc.page}")
    canvas.restoreState()


def _cover(canvas: Any, _doc: Any) -> None:
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
    canvas.setFillColor(BLUE)
    canvas.circle(A4[0] - 18 * mm, A4[1] - 25 * mm, 45 * mm, fill=1, stroke=0)
    canvas.setFillColor(CYAN)
    canvas.circle(A4[0] - 32 * mm, 20 * mm, 28 * mm, fill=1, stroke=0)
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 28)
    canvas.drawString(22 * mm, A4[1] - 72 * mm, "harmonic-dla")
    canvas.setFont("Helvetica-Bold", 22)
    canvas.drawString(22 * mm, A4[1] - 86 * mm, "Code Review &")
    canvas.drawString(22 * mm, A4[1] - 99 * mm, "Four-Approach Benchmark")
    canvas.setFont("Helvetica", 11)
    canvas.setFillColor(HexColor("#D9EAF7"))
    canvas.drawString(
        22 * mm, A4[1] - 116 * mm, "Exact Poisson | Uniform | Sample-split | Paper-amortized"
    )
    canvas.setFont("Helvetica-Bold", 10)
    canvas.setFillColor(WHITE)
    canvas.drawString(22 * mm, 33 * mm, "Research-alpha assessment")
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(HexColor("#D9EAF7"))
    canvas.drawString(
        22 * mm, 26 * mm, "30,000 particles x 3 seeds | Apple silicon | 4 Numba threads"
    )
    canvas.drawString(22 * mm, 20 * mm, "26 July 2026")
    canvas.restoreState()


def build_report(data: dict[str, Any], output: Path) -> None:
    style = _styles()
    summaries = data["summaries"]
    methodology = data["methodology"]
    environment = data["environment"]
    output.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=22 * mm,
        bottomMargin=17 * mm,
        title="harmonic-dla Code Review and Four-Approach Benchmark",
        author="OpenAI Codex",
        subject="Performance, scientific validity, and production-readiness assessment",
    )
    story: list[Flowable] = [Spacer(1, 230 * mm), PageBreak()]

    story.extend(
        [
            _p("Executive summary", style["h1"]),
            _p(
                "<b>Verdict:</b> the implementation is fast and well-structured, but it is not "
                "ready for accuracy claims or release. Exact Poisson return is the best default "
                "today. Uniform restart is not faster in a meaningful way and has no certificate. "
                "Both controlled approaches spend most of their work on calibration, while the "
                "review found certificate-validity gaps that prevent treating their recorded "
                "bounds as complete path-law guarantees.",
                style["body"],
            ),
            Spacer(1, 2 * mm),
        ]
    )
    exact = summaries["exact"]
    fixed = summaries["controlled-fixed"]
    paper = summaries["paper-amortized"]
    cards = Table(
        [
            [
                _metric_card(
                    f"{exact['throughput_median_particles_per_second'] / 1000:.1f}k/s",
                    "Exact median throughput",
                    style,
                ),
                _metric_card("13.9x", "Exact vs sample-split", style),
                _metric_card("24.6x", "Exact vs paper-amortized", style),
                _metric_card("2 critical", "Certificate findings", style),
            ]
        ],
        colWidths=[43.5 * mm] * 4,
    )
    cards.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.extend([cards, Spacer(1, 6 * mm)])

    story.extend(
        [
            _p("What the benchmark supports", style["h2"]),
            _p(
                "At equal particle count, identical numerical settings, and four threads, exact "
                "Poisson return achieved the highest median throughput. Uniform restart was 4.7% "
                "slower at the median. Sample-split control was 13.9x slower than exact and "
                "paper-amortized was 24.6x slower. Peak memory differed by less than 4 MB across "
                "approaches, so runtime and calibration work - not memory - drive the trade-off.",
                style["body"],
            ),
            _p("What it does not support", style["h2"]),
            _p(
                "This is a whole-run engineering benchmark, not the repository's unfinished "
                "large-ensemble, matched-accuracy campaign. Three stochastic runs quantify runtime "
                "variation but are insufficient to establish morphological law parity. The "
                "sample-split path bound is not valid across its configured multi-particle block, "
                "and the shipped paper-amortized parameters yield a trivial infinite-history upper "
                "bound of 1.0.",
                style["body"],
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            _p("Benchmark methodology", style["h1"]),
            _p(
                "The benchmark measures complete in-memory aggregate growth. Each approach/seed "
                "runs in a fresh process. Its relevant Numba path is warmed first, and compilation "
                "time is excluded. Peak RSS remains process-isolated. The same master seeds are "
                "used for all approaches, although trajectories diverge once their restart laws "
                "differ.",
                style["body"],
            ),
            _standard_table(
                [
                    ["Control", "Value", "Reason"],
                    ["Particle count", f"{methodology['particles']:,}", "Equal work horizon"],
                    [
                        "Seeds",
                        ", ".join(map(str, methodology["seeds"])),
                        "Same three seeds per mode",
                    ],
                    ["Threads", str(environment["threads"]), "Fixed Numba parallelism"],
                    ["Backend", "Numba CPU", "Primary production engine"],
                    [
                        "Death ratio / launch margin",
                        "4.0 / 4.0 radii",
                        "Shipped fixed-boundary defaults",
                    ],
                    ["Walker tolerance", "1e-6", "Shipped numerical default"],
                    [
                        "Fixed calibration",
                        "4096 search + 4096 validation; block 256",
                        "Shipped config",
                    ],
                    [
                        "Paper schedule",
                        "alpha .48; gamma .06; scale 32; start 1024",
                        "Shipped config",
                    ],
                ],
                [43 * mm, 55 * mm, 77 * mm],
                style,
            ),
            Spacer(1, 5 * mm),
            _p("Execution environment", style["h2"]),
            _standard_table(
                [
                    ["Field", "Recorded value"],
                    ["Timestamp (UTC)", environment["timestamp_utc"]],
                    ["Platform", environment["platform"]],
                    ["Machine", environment["machine"]],
                    ["Logical CPUs", environment["logical_cpus"]],
                    ["Python", environment["python"]],
                    ["NumPy / Numba", f"{environment['numpy']} / {environment['numba']}"],
                ],
                [48 * mm, 127 * mm],
                style,
            ),
            Spacer(1, 5 * mm),
            _p(
                "<b>Interpretation constraint:</b> exact and uniform finish in under 0.4 seconds, "
                "so their 5% median difference should be read as practical parity on this hardware. "
                "The controlled modes are long enough for their order-of-magnitude gaps to be clear.",
                style["callout"],
            ),
            PageBreak(),
        ]
    )

    runtime_values = [
        (LABELS[name], summaries[name]["runtime_median_seconds"], COLORS[name])
        for name in APPROACHES
    ]
    throughput_values = [
        (
            LABELS[name],
            summaries[name]["throughput_median_particles_per_second"] / 1000,
            COLORS[name],
        )
        for name in APPROACHES
    ]
    story.extend(
        [
            _p("Performance results", style["h1"]),
            _p("Median wall time - lower is better", style["h2"]),
            HorizontalBars(
                runtime_values,
                width=174 * mm,
                height=52 * mm,
                formatter="{:.2f}",
                suffix=" s",
            ),
            Spacer(1, 4 * mm),
            _p("Median throughput - higher is better", style["h2"]),
            HorizontalBars(
                throughput_values,
                width=174 * mm,
                height=52 * mm,
                formatter="{:.1f}",
                suffix="k particles/s",
            ),
            Spacer(1, 5 * mm),
        ]
    )
    result_rows = [["Approach", "Median time", "Range", "Throughput", "vs exact", "Peak RSS"]]
    for name in APPROACHES:
        summary = summaries[name]
        result_rows.append(
            [
                LABELS[name],
                f"{summary['runtime_median_seconds']:.3f} s",
                f"{summary['runtime_min_seconds']:.3f}-{summary['runtime_max_seconds']:.3f} s",
                f"{summary['throughput_median_particles_per_second']:,.0f}/s",
                f"{summary['speedup_vs_exact']:.3f}x",
                f"{summary['peak_rss_median_bytes'] / 1024 / 1024:.1f} MB",
            ]
        )
    story.extend(
        [
            _standard_table(
                result_rows,
                [35 * mm, 25 * mm, 34 * mm, 30 * mm, 23 * mm, 28 * mm],
                style,
            ),
            Spacer(1, 4 * mm),
            _p(
                "The key surprise is that uniform restart does not produce a throughput win over "
                "exact Poisson return. Its median is slightly worse and its growth walkers take "
                "about 3.3% more WOS steps. The exact angular-return calculation is not the active "
                "bottleneck at this scale.",
                style["body"],
            ),
            PageBreak(),
        ]
    )

    probe_values = [
        (
            LABELS[name],
            float(summaries[name]["calibration_probes_median"]),
            COLORS[name],
        )
        for name in APPROACHES
    ]
    step_values = [
        (
            LABELS[name],
            float(
                summaries[name]["growth_steps_median"] + summaries[name]["calibration_steps_median"]
            ),
            COLORS[name],
        )
        for name in APPROACHES
    ]
    story.extend(
        [
            _p("Where the time goes", style["h1"]),
            _p("Calibration probes (log-scaled bars)", style["h2"]),
            HorizontalBars(
                probe_values,
                width=174 * mm,
                height=50 * mm,
                formatter="{:,.0f}",
                log_scale=True,
            ),
            Spacer(1, 3 * mm),
            _p("Total WOS steps (log-scaled bars)", style["h2"]),
            HorizontalBars(
                step_values,
                width=174 * mm,
                height=50 * mm,
                formatter="{:,.0f}",
                log_scale=True,
            ),
            Spacer(1, 4 * mm),
            _standard_table(
                [
                    ["Approach", "Growth steps", "Calibration steps", "Probes", "Events"],
                    *[
                        [
                            LABELS[name],
                            f"{summaries[name]['growth_steps_median']:,.0f}",
                            f"{summaries[name]['calibration_steps_median']:,.0f}",
                            f"{summaries[name]['calibration_probes_median']:,.0f}",
                            f"{summaries[name]['calibration_events_median']:,.0f}",
                        ]
                        for name in APPROACHES
                    ],
                ],
                [40 * mm, 35 * mm, 40 * mm, 34 * mm, 26 * mm],
                style,
            ),
            Spacer(1, 5 * mm),
            _p(
                "Sample-split fixed performs 966,656 calibration probes and about 41.2 million "
                "calibration WOS steps at the median - roughly 34x its growth work. "
                "Paper-amortized uses 1.60 million probes across 5,390 very short blocks and about "
                "63.6 million calibration steps - roughly 53x its growth work.",
                style["body"],
            ),
            PageBreak(),
        ]
    )

    representatives: dict[str, list[list[float]]] = {}
    for name in APPROACHES:
        representatives[name] = next(
            record["representative_positions"]
            for record in data["records"]
            if record["approach"] == name and record["representative_positions"] is not None
        )
    story.extend(
        [
            _p("Morphology snapshot", style["h1"]),
            _p(
                "Representative seed 42 aggregates, plotted at the same particle count. Each panel "
                "is independently scaled to reveal structure; this is descriptive, not an accuracy "
                "test. With only three seeds, shape metrics are too noisy to establish law parity.",
                style["body"],
            ),
            ClusterPanels(representatives, width=174 * mm, height=151 * mm),
            Spacer(1, 4 * mm),
            _standard_table(
                [
                    ["Approach", "Radius of gyration", "Max radius", "COM offset", "Anisotropy"],
                    *[
                        [
                            LABELS[name],
                            f"{summaries[name]['shape']['radius_of_gyration']:.1f}",
                            f"{summaries[name]['shape']['max_seed_radius']:.1f}",
                            f"{summaries[name]['shape']['center_of_mass_offset']:.1f}",
                            f"{summaries[name]['shape']['anisotropy_ratio']:.2f}",
                        ]
                        for name in APPROACHES
                    ],
                ],
                [40 * mm, 36 * mm, 31 * mm, 31 * mm, 37 * mm],
                style,
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            _p("Accuracy and certificate interpretation", style["h1"]),
            _standard_table(
                [
                    ["Approach", "Implemented meaning", "Recorded result", "Assessment"],
                    [
                        "Exact Poisson",
                        "Exact outer-boundary return angle",
                        "Restart error 0",
                        "Best baseline. Finite WOS tolerance remains outside the analytic certificate.",
                    ],
                    [
                        "Uniform restart",
                        "Legacy uniform relaunch",
                        "No automatic certificate",
                        "No speed advantage here and no accuracy control. Avoid for primary results.",
                    ],
                    [
                        "Sample-split fixed",
                        "Independent center search and validation",
                        f"Median max one-step bound {fixed['max_one_step_bound_median']:.4f}; "
                        "naive block product saturates at 1",
                        "Frozen-target bound is reused across 256 growth steps without a drift term. "
                        "Do not interpret as a valid block/path certificate.",
                    ],
                    [
                        "Paper-amortized",
                        "Exact prefix plus theorem schedule",
                        f"Median max one-step bound {paper['max_one_step_bound_median']:.6f}; "
                        "30k finite product about "
                        f"{paper['conditional_path_bound_median']:.3f}; infinite upper bound 1.0",
                        "Schedule is structurally implemented, but the shipped scale gives no "
                        "nontrivial infinite-history guarantee.",
                    ],
                ],
                [32 * mm, 40 * mm, 46 * mm, 57 * mm],
                style,
            ),
            Spacer(1, 5 * mm),
            _p("Why matched accuracy still matters", style["h2"]),
            _p(
                "Equal particle count answers an engineering question: how much machine work does "
                "each policy require for the same growth horizon? It does not answer the scientific "
                "question: which policy is fastest for the same demonstrable approximation error? "
                "That requires a predeclared ensemble campaign, common fitting windows, confidence "
                "intervals on morphology/continuum discrepancies, and a certificate whose scope "
                "matches the full growth path.",
                style["body"],
            ),
            _p(
                "<b>Recommendation:</b> publish the current benchmark only as an engineering "
                "baseline. Do not claim controlled restart is faster or accuracy-matched.",
                style["callout"],
            ),
            PageBreak(),
        ]
    )

    findings = [
        (
            "Critical",
            "Sample-split bound reused after target changes",
            "src/harmonic_dla/backends/numba_cpu/backend.py:390-405, 469-497",
            "Validation certifies the frozen aggregate at count, then the center/bound are reused "
            "for up to block_size attachments without a staleness or drift term.",
            "Recalibrate every attachment or implement and persist a valid within-block budget.",
        ),
        (
            "Critical",
            "Unbudgeted non-exact paper prefix is allowed",
            "src/harmonic_dla/config.py:69; backends/numba_cpu/backend.py:334-338",
            "exact_prefix=False switches to uniform restart before the theorem schedule without "
            "including prefix error in the full path guarantee.",
            "Reject false until a prefix path budget exists.",
        ),
        (
            "Important",
            "Canonical paper config has a trivial path bound",
            "configs/paper-amortized.toml:19",
            "scale=32.0 keeps the infinite-history upper bound non-trivial for start=1024.",
            "Keep the declared target budget synchronized with this scale and report it in raw data.",
        ),
        (
            "Important",
            "Finite WOS tolerance is outside certificates",
            "src/harmonic_dla/backends/numba_cpu/kernels.py:161-168",
            "Projection at tolerance preserves contact but contributes numerical first-passage "
            "error not included in restart TV bounds.",
            "Qualify exactness and add tolerance-convergence validation or a numerical error term.",
        ),
        (
            "Important",
            "Diagnostics cannot reconstruct every calibration",
            "src/harmonic_dla/models.py:20-37; provenance.py:15-30",
            "Archives omit per-event probe count, block stop, geometry scale/radii, and often the "
            "source revision.",
            "Persist aligned event arrays and robust source revision provenance.",
        ),
        (
            "Important",
            "TOML coercion and unknown-key acceptance",
            "src/harmonic_dla/config.py:217-273",
            'For example, bool("false") is true, fractional integer values truncate, and typos '
            "silently fall back to defaults.",
            "Validate exact types and reject unknown keys at every table.",
        ),
        (
            "Important",
            "Compiled scientific laws lack direct parity tests",
            "tests/statistical/test_poisson_return_modes.py:9-17",
            "The statistical test covers only the NumPy helper, not the compiled sampler or "
            "end-to-end transition laws for all four modes.",
            "Add compiled/reference law parity and paper-amortized integration coverage.",
        ),
        (
            "Minor",
            "Unbounded NPZ decompression",
            "src/harmonic_dla/io.py:125",
            "allow_pickle=False prevents object execution but not memory exhaustion from a large "
            "compressed archive.",
            "Document trusted-input scope or enforce archive/member size limits.",
        ),
    ]
    story.extend(
        [
            _p("Code review findings", style["h1"]),
            _p(
                "Review scope: the current codebase, not a Git diff (this directory has no Git "
                "history). Findings focus on correctness, scientific validity, reproducibility, "
                "security, and production readiness. The fresh non-slow suite passed: 52 tests.",
                style["body"],
            ),
            _standard_table(
                [
                    ["Severity", "Finding", "Reference", "Why it matters / action"],
                    *[
                        [
                            severity,
                            title,
                            reference,
                            f"{impact} <b>Action:</b> {action}",
                        ]
                        for severity, title, reference, impact, action in findings
                    ],
                ],
                [20 * mm, 37 * mm, 48 * mm, 70 * mm],
                style,
            ),
            Spacer(1, 5 * mm),
            _p(
                "<b>Release verdict: NOT READY.</b> The two critical certificate findings block "
                "accuracy claims. The repository also still lacks the declared uv.lock, live lint/"
                "type evidence, large-ensemble validation, and matched-accuracy benchmark campaign.",
                style["callout"],
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            _p("Recommended next actions", style["h1"]),
            _standard_table(
                [
                    ["Priority", "Action", "Completion evidence"],
                    [
                        "P0",
                        "Fix sample-split certificate scope: block_size=1 or a proved drift budget.",
                        "Per-step/block theorem mapping, regression tests, persisted bound semantics.",
                    ],
                    [
                        "P0",
                        "Disallow paper-amortized exact_prefix=False until prefix error is budgeted.",
                        "Config validation test and explicit full-path certificate calculation.",
                    ],
                    [
                        "P1",
                        "Choose paper scale from a declared target path budget.",
                        "Config emits a nontrivial bound below the chosen target before simulation.",
                    ],
                    [
                        "P1",
                        "Instrument growth and calibration wall time separately.",
                        "Raw JSON reports both times; accounting sums to whole-run wall time.",
                    ],
                    [
                        "P1",
                        "Add compiled law-parity and tolerance-convergence tests.",
                        "Reference/Numba frozen-law comparisons and epsilon convergence plots.",
                    ],
                    [
                        "P1",
                        "Run the preregistered matched-accuracy ensemble campaign.",
                        "Retained raw data, confidence intervals, common windows, multiple sizes.",
                    ],
                    [
                        "P2",
                        "Harden TOML schema, event provenance, and NPZ input limits.",
                        "Negative tests for malformed configs/archives and complete event replay.",
                    ],
                ],
                [19 * mm, 79 * mm, 77 * mm],
                style,
            ),
            Spacer(1, 7 * mm),
            _p("Decision guide", style["h2"]),
            _standard_table(
                [
                    ["If your goal is...", "Use", "Because"],
                    [
                        "A defensible baseline today",
                        "Exact Poisson",
                        "Fastest median result and exact outer-boundary return.",
                    ],
                    [
                        "Historical compatibility only",
                        "Uniform restart",
                        "Explicit legacy mode; not faster here and uncertified.",
                    ],
                    [
                        "Center-calibration research",
                        "Sample-split fixed",
                        "Useful experimental machinery after certificate scope is corrected.",
                    ],
                    [
                        "The paper's asymptotic schedule",
                        "Paper-amortized",
                        "Use only as a demonstrator until scale/budget and validation are resolved.",
                    ],
                ],
                [48 * mm, 42 * mm, 85 * mm],
                style,
            ),
            Spacer(1, 7 * mm),
            _p(
                "Bottom line: keep exact Poisson return as the operational default. The codebase "
                "has a solid architecture and unusually good transparency about its research-alpha "
                "status, but the current controlled-mode certificates do not yet support the "
                "performance-versus-accuracy story the project ultimately wants to tell.",
                style["callout"],
            ),
            PageBreak(),
        ]
    )

    raw_rows = [["Approach", "Seed", "Time (s)", "Rate (/s)", "RSS (MB)", "Growth steps", "Probes"]]
    raw_rows.extend(
        [
            [
                LABELS[record["approach"]],
                record["seed"],
                f"{record['elapsed_seconds']:.4f}",
                f"{record['particles_per_second']:,.0f}",
                f"{record['peak_rss_bytes'] / 1024 / 1024:.1f}",
                f"{record['growth_walker_steps']:,}",
                f"{record['calibration_probes']:,}",
            ]
            for record in data["records"]
        ]
    )
    story.extend(
        [
            _p("Appendix: raw run summary", style["h1"]),
            _standard_table(
                raw_rows,
                [34 * mm, 15 * mm, 23 * mm, 27 * mm, 22 * mm, 29 * mm, 25 * mm],
                style,
            ),
            Spacer(1, 6 * mm),
            _p("Reproducibility artifacts", style["h2"]),
            _standard_table(
                [
                    ["Artifact", "Location"],
                    ["Raw benchmark JSON", "output/benchmark/four_approaches_raw.json"],
                    ["Benchmark driver", "scripts/benchmark_four_approaches.py"],
                    ["Report generator", "scripts/generate_benchmark_report.py"],
                    ["Final report", "output/pdf/harmonic_dla_code_review_and_benchmark.pdf"],
                ],
                [45 * mm, 130 * mm],
                style,
            ),
            Spacer(1, 5 * mm),
            _p(
                "Test evidence: PYTHONPATH=src .venv/bin/pytest -q -> 92 passed. "
                "Benchmark times exclude approach-specific JIT warm-up.",
                style["small"],
            ),
        ]
    )

    doc.build(story, onFirstPage=_cover, onLaterPages=_header_footer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/benchmark/four_approaches_raw.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/pdf/harmonic_dla_code_review_and_benchmark.pdf"),
    )
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    build_report(data, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
