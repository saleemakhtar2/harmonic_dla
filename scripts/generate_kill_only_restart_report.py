"""Generate the kill-only restart benchmark report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

CAPABILITIES: dict[str, dict[str, Any]] = {
    "exact": {
        "label": "Exact Poisson oracle",
        "needs_escape_angle": True,
        "needs_attachment_samples": False,
        "kill_only_eligible": False,
    },
    "uniform": {
        "label": "Uniform restart",
        "needs_escape_angle": False,
        "needs_attachment_samples": False,
        "kill_only_eligible": True,
    },
    "controlled-fixed": {
        "label": "Per-step controlled restart",
        "needs_escape_angle": False,
        "needs_attachment_samples": True,
        "kill_only_eligible": True,
    },
    "paper-amortized": {
        "label": "Paper-amortized control",
        "needs_escape_angle": False,
        "needs_attachment_samples": True,
        "kill_only_eligible": True,
    },
}

COLORS = {
    "exact": "#173F5F",
    "uniform": "#20639B",
    "controlled-fixed": "#ED553B",
    "paper-amortized": "#3CAEA3",
    "one-shot-centered": "#F6A01A",
}
POLICIES = ("uniform", "one-shot-centered")
DYNAMIC_POLICIES = tuple(CAPABILITIES)
PRIMARY_RHOS = (3, 4, 6, 8, 12, 16)


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing benchmark artifact: {path}")
    try:
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load benchmark artifact {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"benchmark artifact must contain an object: {path}")
    return data


def load_inputs(
    static_path: Path, dynamic_paths: tuple[Path, Path]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return _load(static_path), [_load(path) for path in dynamic_paths]


def validate_inputs(static: dict[str, Any], dynamics: list[dict[str, Any]]) -> None:
    records = static.get("records")
    if not isinstance(records, list):
        raise ValueError("static artifact records must be a list")
    for record in records:
        policies = record.get("policies", {})
        for policy in POLICIES:
            if policy not in policies:
                raise ValueError(f"static record is missing {policy}")
    if len(records) != 720:
        raise ValueError("static artifact must contain exactly 720 records")
    if not isinstance(static.get("targets"), dict) or len(static["targets"]) != 4:
        raise ValueError("static artifact must contain four target geometries")
    if len(dynamics) != 2:
        raise ValueError("exactly two dynamic artifacts are required")
    for data in dynamics:
        summaries = data.get("summaries", {})
        for policy in DYNAMIC_POLICIES:
            if policy not in summaries:
                raise ValueError(f"dynamic artifact is missing {policy}")


def _median(static: dict[str, Any], policy: str, rho: int, metric: str) -> float:
    values = [
        float(record["policies"][policy][metric])
        for record in static["records"]
        if int(record["rho"]) == rho
    ]
    if not values:
        raise ValueError(f"no static rows for {policy} at rho={rho}")
    return float(np.median(values))


def static_summary(static: dict[str, Any], policy: str, rho: int) -> dict[str, float]:
    return {
        "particle_id_tv": _median(static, policy, rho, "particle_id_tv"),
        "bounded_lipschitz": _median(static, policy, rho, "bounded_lipschitz"),
        "seconds": _median(static, policy, rho, "seconds"),
    }


def dynamic_rows(dynamics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "particles": int(data["records"][0]["particles"]),
            "policy": policy,
            "runtime": float(summary["runtime_median_seconds"]),
            "probes": float(summary["calibration_probes_median"]),
            "eligible": bool(CAPABILITIES[policy]["kill_only_eligible"]),
        }
        for data in dynamics
        for policy, summary in data["summaries"].items()
    ]


def _figure_save(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_flow(path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.4), constrained_layout=True)
    for axis in axes:
        axis.axis("off")
    axes[0].text(
        0.5,
        0.82,
        "Instrumented boundary",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=COLORS["exact"],
    )
    axes[0].text(
        0.5,
        0.58,
        "ESCAPED + angle theta",
        ha="center",
        va="center",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#E5EEF5", "edgecolor": COLORS["exact"]},
    )
    axes[0].annotate(
        "",
        xy=(0.5, 0.37),
        xytext=(0.5, 0.51),
        arrowprops={"arrowstyle": "->", "lw": 2, "color": COLORS["exact"]},
    )
    axes[0].text(
        0.5,
        0.22,
        "Poisson-conditioned return",
        ha="center",
        va="center",
        fontsize=11,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#D9F0ED",
            "edgecolor": COLORS["paper-amortized"],
        },
    )
    axes[1].text(
        0.5,
        0.82,
        "Kill-only boundary",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=COLORS["uniform"],
    )
    axes[1].text(
        0.5,
        0.58,
        "ESCAPED (angle discarded)",
        ha="center",
        va="center",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#E5EEF5", "edgecolor": COLORS["uniform"]},
    )
    axes[1].annotate(
        "",
        xy=(0.5, 0.37),
        xytext=(0.5, 0.51),
        arrowprops={"arrowstyle": "->", "lw": 2, "color": COLORS["uniform"]},
    )
    axes[1].text(
        0.5,
        0.22,
        "Fresh uniform or centered restart",
        ha="center",
        va="center",
        fontsize=11,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#FFF2D9",
            "edgecolor": COLORS["one-shot-centered"],
        },
    )
    figure.suptitle(
        "The information boundary: one field determines Poisson eligibility",
        fontsize=14,
        fontweight="bold",
        color="#173F5F",
    )
    _figure_save(figure, path)


def _plot_targets(static: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(11.5, 2.8), constrained_layout=True)
    for axis, (name, target) in zip(axes, sorted(static["targets"].items()), strict=True):
        points = np.asarray(target["positions"], dtype=float)
        axis.scatter(points[:, 0], points[:, 1], s=6, c=points[:, 1], cmap="viridis", linewidths=0)
        axis.set_title(name.replace("-", " ").title(), fontsize=9)
        axis.set_aspect("equal")
        axis.grid(alpha=0.2)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    figure.suptitle("Frozen targets used by the oracle comparison", fontsize=13, fontweight="bold")
    _figure_save(figure, path)


def _plot_accuracy(static: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.1), constrained_layout=True)
    rhos = np.asarray(sorted({int(record["rho"]) for record in static["records"]}), dtype=float)
    for axis, metric, title in zip(
        axes,
        ("particle_id_tv", "bounded_lipschitz"),
        ("Nearest-particle TV", "Bounded-Lipschitz discrepancy"),
        strict=True,
    ):
        for policy in POLICIES:
            values = np.asarray([_median(static, policy, int(rho), metric) for rho in rhos])
            axis.plot(
                rhos,
                np.maximum(values, 1.0e-8),
                marker="o",
                lw=2,
                label=policy,
                color=COLORS[policy],
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("death-radius ratio rho")
        axis.set_ylabel("median discrepancy vs exact oracle")
        axis.set_title(title)
        axis.grid(which="both", alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Accuracy improves as the kill boundary moves outward", fontsize=13, fontweight="bold"
    )
    _figure_save(figure, path)


def _plot_pareto(static: dict[str, Any], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.7), constrained_layout=True)
    for policy in POLICIES:
        points = [
            (
                float(record["policies"][policy]["seconds"]),
                float(record["policies"][policy]["particle_id_tv"]),
            )
            for record in static["records"]
        ]
        x = np.asarray([point[0] for point in points])
        y = np.asarray([point[1] for point in points])
        axis.scatter(x, y, s=14, alpha=0.35, color=COLORS[policy], label=policy)
        axis.scatter(
            np.median(x),
            np.median(y),
            s=130,
            marker="*",
            color=COLORS[policy],
            edgecolor="white",
            linewidth=0.7,
        )
    axis.scatter(
        [], [], marker="x", s=70, color=COLORS["exact"], label="Exact Poisson (oracle only)"
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("policy wall time (seconds; log scale)")
    axis.set_ylabel("nearest-particle TV vs exact oracle")
    axis.set_title("Eligible accuracy-cost frontier under the kill-only interface")
    axis.grid(which="both", alpha=0.2)
    axis.legend(frameon=False, fontsize=8)
    axis.text(
        0.03,
        0.05,
        "Poisson is shown as the truth source, not an eligible implementation.",
        transform=axis.transAxes,
        fontsize=8,
        color=COLORS["exact"],
    )
    _figure_save(figure, path)


def _plot_dynamic(dynamics: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.0), constrained_layout=True)
    for data, marker in zip(dynamics, ("o", "s"), strict=True):
        particles = int(data["records"][0]["particles"])
        for policy, summary in data["summaries"].items():
            eligible = CAPABILITIES[policy]["kill_only_eligible"]
            kwargs = {
                "facecolors": COLORS[policy] if eligible else "white",
                "edgecolors": COLORS[policy],
                "linewidths": 1.5,
            }
            axes[0].scatter(
                particles,
                summary["runtime_median_seconds"],
                marker=marker,
                s=65,
                label=CAPABILITIES[policy]["label"] if particles == 1200 else None,
                **kwargs,
            )
            axes[1].scatter(
                particles,
                max(float(summary["calibration_probes_median"]), 1.0),
                marker=marker,
                s=65,
                **kwargs,
            )
    for axis in axes:
        axis.set_yscale("log")
        axis.set_xlabel("particles")
        axis.grid(which="both", alpha=0.2)
    axes[0].set_ylabel("median runtime (seconds)")
    axes[1].set_ylabel("median calibration probes")
    axes[0].set_title("Dynamic runtime")
    axes[1].set_title("Calibration work")
    axes[0].legend(frameon=False, fontsize=7)
    figure.suptitle(
        "Dynamic benchmark: filled = kill-only eligible, hollow = oracle-only",
        fontsize=13,
        fontweight="bold",
    )
    _figure_save(figure, path)


def _fmt(value: float) -> str:
    return f"{value:.3g}"


def _table(data: list[list[str]], widths: list[float], header_color: str = "#173F5F") -> Table:
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_color)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E0")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return table


def _page_decor(canvas: Any, _document: Any) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#D9E2EC"))
    canvas.line(0.55 * inch, height - 0.35 * inch, width - 0.55 * inch, height - 0.35 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#718096"))
    canvas.drawString(0.55 * inch, 0.25 * inch, "harmonic-dla | kill-only restart benchmark")
    page = f"Page {canvas.getPageNumber()}"
    canvas.drawRightString(width - 0.55 * inch, 0.25 * inch, page)
    canvas.restoreState()


def build_report(
    static: dict[str, Any], dynamics: list[dict[str, Any]], output: Path, figure_dir: Path
) -> None:
    validate_inputs(static, dynamics)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    flow = figure_dir / "boundary_information_flow.png"
    targets = figure_dir / "targets.png"
    accuracy = figure_dir / "static_accuracy.png"
    pareto = figure_dir / "eligible_pareto.png"
    dynamic = figure_dir / "dynamic_calibration.png"
    _plot_flow(flow)
    _plot_targets(static, targets)
    _plot_accuracy(static, accuracy)
    _plot_pareto(static, pareto)
    _plot_dynamic(dynamics, dynamic)

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="CoverX",
            parent=styles["Title"],
            fontSize=26,
            leading=31,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#173F5F"),
            spaceAfter=16,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubX",
            parent=styles["Normal"],
            fontSize=11,
            leading=16,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4A5568"),
            spaceAfter=14,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H1X",
            parent=styles["Heading1"],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#173F5F"),
            spaceBefore=6,
            spaceAfter=9,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H2X",
            parent=styles["Heading2"],
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#20639B"),
            spaceBefore=7,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyX",
            parent=styles["BodyText"],
            fontSize=9.3,
            leading=13.5,
            alignment=TA_LEFT,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallX",
            parent=styles["BodyText"],
            fontSize=7.8,
            leading=10.5,
            textColor=colors.HexColor("#4A5568"),
            spaceAfter=4,
        )
    )
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.52 * inch,
        bottomMargin=0.48 * inch,
    )
    story: list[Any] = [
        Spacer(1, 0.35 * inch),
        Paragraph("Kill-only restart benchmark", styles["CoverX"]),
        Paragraph(
            "When exact Poisson return cannot be implemented because the escape angle is discarded",
            styles["SubX"],
        ),
        Paragraph("Executive result", styles["H1X"]),
        Paragraph(
            "The Poisson kernel is still the exact mathematical answer for ordinary Brownian motion between concentric circles. This experiment asks a different engineering question: what can we do when the boundary interface reports only that a walker escaped and throws away the escape angle? In that setting, exact Poisson return is an oracle but not an eligible implementation. Centered restart remains feasible and controls the finite-boundary bias without needing the missing angle.",
            styles["BodyX"],
        ),
        Image(str(flow), width=7.15 * inch, height=2.2 * inch),
        Paragraph(
            "The static evidence reuses 720 independently validated records across four asymmetric targets, nine death-radius ratios, and 20 replications. The dynamic evidence reuses the N = 1,200 and N = 3,000 four-approach runs. The exact-prefix disclosure on page 5 is intentional: those dynamic runs demonstrate calibration scheduling, not an end-to-end kill-only prefix.",
            styles["BodyX"],
        ),
        PageBreak(),
        Paragraph("1. What the interface allows", styles["H1X"]),
        Paragraph(
            "A Poisson return sample is conditioned on the point where the walker crossed the outer circle. If that angle is retained, the return correction is an O(1) operation. If the angle is discarded, no later restart routine can reconstruct the same conditional law from the event flag alone. Uniform, controlled-fixed, and paper-amortized restart use fresh launches and attachment samples instead, so they remain compatible with the constrained interface.",
            styles["BodyX"],
        ),
        _table(
            [["Policy", "Needs escape angle", "Needs attachment samples", "Kill-only eligible"]]
            + [
                [
                    CAPABILITIES[p]["label"],
                    "yes" if CAPABILITIES[p]["needs_escape_angle"] else "no",
                    "yes" if CAPABILITIES[p]["needs_attachment_samples"] else "no",
                    "yes" if CAPABILITIES[p]["kill_only_eligible"] else "no",
                ]
                for p in DYNAMIC_POLICIES
            ],
            [2.35 * inch, 1.35 * inch, 1.55 * inch, 1.3 * inch],
        ),
        Spacer(1, 0.12 * inch),
        Image(str(targets), width=7.15 * inch, height=1.75 * inch),
        Paragraph(
            "Exact Poisson is therefore shown as the truth source and an unconstrained baseline, not as a failed competitor. The kernel remains mathematically valid on a circle; only the constrained interface prevents this component from using it.",
            styles["SmallX"],
        ),
        PageBreak(),
        Paragraph("2. Static accuracy against the oracle", styles["H1X"]),
        Image(str(accuracy), width=7.15 * inch, height=2.75 * inch),
        Paragraph(
            "The static frozen-target experiment holds the circular death boundary construction fixed and increases rho, the death-radius to target-radius ratio. Uniform restart retains the leading directional mismatch. One-shot centering estimates the finite-boundary barycenter independently and removes that broad dipole component. The plotted values are finite partition diagnostics, not continuum total-variation proofs.",
            styles["BodyX"],
        ),
        _table(
            [
                ["Policy", "Median TV at rho=4", "Median TV at rho=16", "Interpretation"],
                [
                    "Uniform restart",
                    _fmt(_median(static, "uniform", 4, "particle_id_tv")),
                    _fmt(_median(static, "uniform", 16, "particle_id_tv")),
                    "Feasible, no calibration",
                ],
                [
                    "One-shot centered",
                    _fmt(_median(static, "one-shot-centered", 4, "particle_id_tv")),
                    _fmt(_median(static, "one-shot-centered", 16, "particle_id_tv")),
                    "Feasible, one calibration",
                ],
                ["Exact Poisson", "0", "0", "Oracle reference"],
            ],
            [1.35 * inch, 1.2 * inch, 1.2 * inch, 2.55 * inch],
        ),
        PageBreak(),
        Paragraph("3. Eligible accuracy-cost frontier", styles["H1X"]),
        Image(str(pareto), width=7.15 * inch, height=3.7 * inch),
        Paragraph(
            "This chart answers the practical question under the stated interface. Exact Poisson is marked oracle-only because it needs data that the kill-only adapter has destroyed. Uniform restart is the cheapest eligible policy but carries the largest finite-boundary discrepancy. Centering spends calibration work to move toward the oracle law; version 4 is designed to reuse that calibration over blocks instead of paying for a fresh search after every attachment.",
            styles["BodyX"],
        ),
        Paragraph(
            "A fair conclusion is therefore a feasible-frontier statement: approach 4 is valuable when the exact return information is unavailable and calibration work matters. It is not a claim that approach 4 should replace exact Poisson in a simulator that already retains escape angles.",
            styles["BodyX"],
        ),
        PageBreak(),
        Paragraph("4. Dynamic calibration work", styles["H1X"]),
        Image(str(dynamic), width=7.15 * inch, height=2.65 * inch),
        Paragraph(
            "Filled markers are kill-only eligible; hollow markers are the exact Poisson oracle. The paper-amortized schedule spreads calibration over growing blocks, while controlled-fixed repeats calibration frequently. The measured runtime and probe counts are from the existing dynamic benchmark and use an exact prefix. They are valid evidence about post-prefix scheduling cost, but they are not an end-to-end kill-only execution.",
            styles["BodyX"],
        ),
        _table(
            [["N", "Policy", "Median runtime (s)", "Median calibration probes", "Eligible"]]
            + [
                [
                    str(row["particles"]),
                    CAPABILITIES[row["policy"]]["label"],
                    _fmt(row["runtime"]),
                    _fmt(row["probes"]),
                    "yes" if row["eligible"] else "oracle only",
                ]
                for row in dynamic_rows(dynamics)
            ],
            [0.55 * inch, 2.15 * inch, 1.25 * inch, 1.55 * inch, 1.0 * inch],
        ),
        Paragraph(
            "To make the complete pipeline kill-only, the exact prefix must be replaced by a separately budgeted approximate prefix. That follow-up run is a deployment validation, not silently inferred from this report.",
            styles["SmallX"],
        ),
        PageBreak(),
        Paragraph("5. Findings, limits, and reproduction", styles["H1X"]),
        Paragraph(
            "Finding: the useful scenario for approach 4 is not a circle where Poisson is somehow mathematically invalid. It is a system boundary where the escape angle is not available to the restart component. Under that contract, approach 4 provides a certified way to recover much of the lost directional information through center estimation and amortized reuse.",
            styles["BodyX"],
        ),
        Paragraph(
            "Limits: the paper's theorem still assumes ordinary Brownian motion and a circular death boundary. This report does not extend it to irregular boundaries, drift, heterogeneous diffusion, or arbitrary hardware. Wall times are machine-specific. The dynamic data contain an exact prefix and are labelled accordingly. The static diagnostics use finite particle and angular partitions.",
            styles["BodyX"],
        ),
        Paragraph(
            "Reproduction: run `PYTHONPATH=src:. .venv/bin/python scripts/generate_kill_only_restart_report.py` with the committed static and dynamic JSON artifacts. The generator validates the 720 static records, required policies, target count, and both dynamic summaries before writing the PDF. Render the output with Poppler for visual inspection.",
            styles["BodyX"],
        ),
        Paragraph(
            "Artifacts: `scripts/generate_kill_only_restart_report.py`, `tests/unit/test_kill_only_restart_report.py`, the generated figures under `output/asymmetric_validation/figures_kill_only/`, and this PDF. Report date: 2026-07-26.",
            styles["SmallX"],
        ),
    ]
    doc.build(story, onFirstPage=_page_decor, onLaterPages=_page_decor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--static",
        type=Path,
        default=Path("output/asymmetric_validation/asymmetric_validation_production.json"),
    )
    parser.add_argument(
        "--dynamic-1200", type=Path, default=Path("output/benchmark/asymmetric_dynamic_1200.json")
    )
    parser.add_argument(
        "--dynamic-3000", type=Path, default=Path("output/benchmark/asymmetric_dynamic_3000.json")
    )
    parser.add_argument(
        "--figure-dir", type=Path, default=Path("output/asymmetric_validation/figures_kill_only")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/pdf/harmonic_dla_kill_only_restart_benchmark.pdf"),
    )
    args = parser.parse_args()
    static, dynamics = load_inputs(args.static, (args.dynamic_1200, args.dynamic_3000))
    build_report(static, dynamics, args.output, args.figure_dir)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
