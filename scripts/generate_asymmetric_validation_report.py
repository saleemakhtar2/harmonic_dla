"""Generate the asymmetric-target finite-boundary validation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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

APPROACH_LABELS = {
    "exact": "Exact Poisson",
    "uniform": "Uniform restart",
    "controlled-fixed": "Controlled fixed",
    "paper-amortized": "Paper-amortized",
}
COLORS = {
    "exact": "#173F5F",
    "uniform": "#20639B",
    "controlled-fixed": "#ED553B",
    "paper-amortized": "#3CAEA3",
    "one-shot-centered": "#F6A01A",
}


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _records_by_policy(
    data: dict[str, Any], policy: str, metric: str, *, target_name: str | None = None
) -> dict[int, list[float]]:
    result: dict[int, list[float]] = {}
    for record in data["records"]:
        if target_name is not None and record["target_name"] != target_name:
            continue
        rho = int(record["rho"])
        if policy == "reference":
            value = float(record["reference"].get(metric, 0.0))
        else:
            value = float(record["policies"][policy][metric])
        result.setdefault(rho, []).append(value)
    return result


def _median_iqr(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    return (
        float(np.median(array)),
        float(np.percentile(array, 25.0)),
        float(np.percentile(array, 75.0)),
    )


def _plot_targets(data: dict[str, Any], path: Path) -> None:
    targets = data["targets"]
    figure, axes = plt.subplots(1, 4, figsize=(12, 3.0), constrained_layout=True)
    for axis, (name, target) in zip(axes, sorted(targets.items()), strict=True):
        points = np.asarray(target["positions"], dtype=float)
        axis.scatter(points[:, 0], points[:, 1], s=7, c=points[:, 1], cmap="viridis", linewidths=0)
        axis.set_title(name.replace("-", " ").title(), fontsize=9)
        axis.set_aspect("equal")
        axis.grid(alpha=0.2)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    figure.suptitle("Frozen asymmetric targets", fontsize=13, fontweight="bold")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_static_errors(data: dict[str, Any], path: Path) -> dict[str, dict[str, float]]:
    policies = ("uniform", "one-shot-centered")
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.0), constrained_layout=True)
    slope_table: dict[str, dict[str, float]] = {}
    panels = (
        (axes[0, 0], "particle_id_tv", "Nearest-particle TV · all targets", None),
        (axes[0, 1], "bounded_lipschitz", "Bounded-Lipschitz discrepancy · all targets", None),
        (axes[1, 0], "particle_id_tv", "Nearest-particle TV · lopsided comb", "lopsided-comb"),
        (
            axes[1, 1],
            "bounded_lipschitz",
            "Bounded-Lipschitz discrepancy · lopsided comb",
            "lopsided-comb",
        ),
    )
    for axis, metric, title, target_name in panels:
        for policy in policies:
            grouped = _records_by_policy(data, policy, metric, target_name=target_name)
            rhos = np.asarray(sorted(grouped), dtype=float)
            medians = np.asarray([_median_iqr(grouped[int(rho)])[0] for rho in rhos])
            lows = np.asarray([_median_iqr(grouped[int(rho)])[1] for rho in rhos])
            highs = np.asarray([_median_iqr(grouped[int(rho)])[2] for rho in rhos])
            medians = np.maximum(medians, 1.0e-8)
            lows = np.maximum(lows, 1.0e-8)
            highs = np.maximum(highs, 1.0e-8)
            axis.plot(rhos, medians, marker="o", label=policy, color=COLORS[policy])
            axis.fill_between(rhos, lows, highs, color=COLORS[policy], alpha=0.13)
            if target_name is None:
                primary = np.isin(rhos, [3, 4, 6, 8, 12, 16])
                slope = float(np.polyfit(np.log(rhos[primary]), np.log(medians[primary]), 1)[0])
                slope_table[policy] = slope_table.get(policy, {})
                slope_table[policy][metric] = slope
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("death-radius ratio rho")
        axis.set_ylabel("median discrepancy")
        axis.set_title(title, fontsize=10)
        axis.grid(which="both", alpha=0.2)
        axis.legend(frameon=False, fontsize=7)
    figure.suptitle(
        "Static frozen-target error as the killing radius grows", fontsize=13, fontweight="bold"
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return slope_table


def _plot_angles(data: dict[str, Any], path: Path) -> None:
    bins = [16, 32, 64, 128]
    figure, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
    for axis, bin_count in zip(axes.flat, bins, strict=True):
        values: dict[str, list[float]] = {}
        for policy in ("uniform", "one-shot-centered"):
            values[policy] = [
                float(item["policies"][policy]["angle_tv"][str(bin_count)])
                for item in data["records"]
                if item["target_name"] == "lopsided-comb" and item["rho"] == 4
            ]
        axis.boxplot(
            values.values(),
            tick_labels=["uniform", "centered"],
            patch_artist=True,
            boxprops={"facecolor": "#D9EEF2"},
            medianprops={"color": "#173F5F"},
        )
        axis.set_title(f"{bin_count} angular bins")
        axis.set_ylabel("TV vs exact reference")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "Representative angular diagnostics: lopsided comb, rho = 4", fontsize=13, fontweight="bold"
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_pareto(data: dict[str, Any], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    grouped: dict[str, list[tuple[float, float]]] = {
        policy: [] for policy in ("uniform", "one-shot-centered")
    }
    for record in data["records"]:
        for policy in grouped:
            item = record["policies"][policy]
            grouped[policy].append((float(item["seconds"]), float(item["particle_id_tv"])))
    for policy, points in grouped.items():
        x = np.asarray([point[0] for point in points])
        y = np.asarray([point[1] for point in points])
        axis.scatter(x, y, s=14, alpha=0.45, label=policy, color=COLORS[policy])
        axis.scatter(
            np.median(x),
            np.median(y),
            s=100,
            marker="*",
            color=COLORS[policy],
            edgecolor="white",
            linewidth=0.7,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("policy wall time (s; log scale)")
    axis.set_ylabel("nearest-particle TV vs exact (log scale)")
    axis.set_title("Accuracy-cost trade-off across targets, radii, and replications")
    axis.grid(which="both", alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_dynamic(dynamic: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for data, marker in zip(dynamic, ("o", "s"), strict=True):
        particles = int(data["records"][0]["particles"])
        summaries = data["summaries"]
        for approach, summary in summaries.items():
            axes[0].scatter(
                particles,
                summary["runtime_median_seconds"],
                marker=marker,
                s=55,
                color=COLORS[approach],
                label=APPROACH_LABELS[approach] if particles == 1200 else None,
            )
            axes[1].scatter(
                particles,
                summary["calibration_probes_median"],
                marker=marker,
                s=55,
                color=COLORS[approach],
            )
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_xlabel("particles")
    axes[1].set_xlabel("particles")
    axes[0].set_ylabel("median runtime (s)")
    axes[1].set_ylabel("median calibration probes")
    axes[0].set_title("Dynamic growth runtime")
    axes[1].set_title("Calibration work")
    axes[0].grid(which="both", alpha=0.2)
    axes[1].grid(which="both", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Four-approach dynamic benchmark (circles: N=1,200; squares: N=3,000)",
        fontsize=13,
        fontweight="bold",
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _fmt(value: float) -> str:
    return f"{value:.3g}"


def build_report(
    static: dict[str, Any],
    pilot: dict[str, Any],
    dynamics: list[dict[str, Any]],
    output: Path,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    target_plot = figure_dir / "targets.png"
    error_plot = figure_dir / "errors.png"
    angle_plot = figure_dir / "angles.png"
    pareto_plot = figure_dir / "pareto.png"
    dynamic_plot = figure_dir / "dynamic.png"
    _plot_targets(static, target_plot)
    slopes = _plot_static_errors(static, error_plot)
    _plot_angles(static, angle_plot)
    _plot_pareto(static, pareto_plot)
    _plot_dynamic(dynamics, dynamic_plot)

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="Cover",
            parent=styles["Title"],
            fontSize=27,
            leading=32,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#173F5F"),
            spaceAfter=18,
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
            spaceAfter=20,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H1x",
            parent=styles["Heading1"],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#173F5F"),
            spaceBefore=8,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H2x",
            parent=styles["Heading2"],
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#20639B"),
            spaceBefore=8,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Bodyx",
            parent=styles["BodyText"],
            fontSize=9.5,
            leading=14,
            alignment=TA_LEFT,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Smallx",
            parent=styles["BodyText"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#4A5568"),
            spaceAfter=4,
        )
    )

    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.5 * inch,
    )
    story: list[Any] = []
    story += [
        Spacer(1, 0.55 * inch),
        Paragraph("Asymmetric-target finite-boundary validation", styles["Cover"]),
        Paragraph(
            "A preregistered benchmark of exact Poisson, uniform restart, one-shot centering, and the four dynamic approaches",
            styles["Subtitle"],
        ),
    ]
    story.append(Paragraph("Executive result", styles["H1x"]))
    story.append(
        Paragraph(
            "The experiment creates a non-circular target geometry while keeping the death boundary circular, so the comparison is non-circular in morphology without changing the theorem's boundary assumptions. Exact Poisson return remains the reference law. As the killing radius grows, uniform restart and one-shot centering approach that reference; centering is the certified approximation that can remove the dominant finite-boundary bias while preserving a practical restart implementation.",
            styles["Bodyx"],
        )
    )
    pilot_decision = pilot["pilot_decision"]
    story.append(
        Paragraph(
            f"The pilot completed {len(pilot['records'])} validated records and triggered doubled evaluation counts for {sum(item['double_evaluation_counts'] for item in pilot_decision['by_rho'].values())} of {len(pilot_decision['by_rho'])} radii. Production therefore used 50,000 evaluation attachments per policy and 8,192 search probes per centered replication.",
            styles["Bodyx"],
        )
    )
    story.append(Image(str(target_plot), width=7.15 * inch, height=1.78 * inch))
    story.append(Paragraph("Scope and interpretation", styles["H2x"]))
    story.append(
        Paragraph(
            "The reported TV values are empirical diagnostics on fixed finite partitions (nearest-particle IDs and angular histograms), not continuum total-variation proofs. The target suite is one deterministic lopsided comb plus three exact-seed DLA targets (256 particles; seeds 42, 137, 911). All static comparisons use the same target, circular boundary construction, and independent deterministic streams.",
            styles["Bodyx"],
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("1. Geometry and experimental contract", styles["H1x"]))
    story.append(Image(str(error_plot), width=7.15 * inch, height=4.5 * inch))
    story.append(
        Paragraph(
            "The top row pools all targets; the bottom row isolates the deliberately lopsided comb, where the centering advantage is clearest at small killing radii. Lines are medians across replications and shaded bands are interquartile ranges. The preregistered slope window is rho in {3, 4, 6, 8, 12, 16}; rho = 2, 24, 32 are diagnostics outside the fit window.",
            styles["Bodyx"],
        )
    )
    slope_rows = [["Policy", "TV slope", "BL slope"]]
    for policy, values in slopes.items():
        slope_rows.append(
            [policy, _fmt(values["particle_id_tv"]), _fmt(values["bounded_lipschitz"])]
        )
    slope_table = Table(slope_rows, colWidths=[2.5 * inch, 1.7 * inch, 1.7 * inch])
    slope_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173F5F")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E0")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
            ]
        )
    )
    story.append(slope_table)
    story.append(
        Paragraph(
            "These slopes are descriptive fits to the production medians, not theorem estimates. A slope closer to zero indicates slower decay of the measured finite-partition discrepancy over this radius range.",
            styles["Smallx"],
        )
    )
    story.append(Image(str(angle_plot), width=6.9 * inch, height=4.6 * inch))
    story.append(PageBreak())

    story.append(Paragraph("2. Accuracy versus cost", styles["H1x"]))
    story.append(Image(str(pareto_plot), width=6.9 * inch, height=4.15 * inch))
    story.append(
        Paragraph(
            "Each point is one target x radius x replication. Stars mark policy medians. The centered policy pays an independent search cost but changes the boundary center once, making it the relevant frozen-target approximation when the morphology is lopsided. Exact Poisson is deliberately not plotted as a competing policy here: it defines the reference attachment law against which the approximations are scored.",
            styles["Bodyx"],
        )
    )
    story.append(Paragraph("Production accounting", styles["H2x"]))
    cost_rows = [["Policy", "Median seconds", "Median probes", "Median TV"]]
    for policy in ("uniform", "one-shot-centered"):
        seconds = []
        probes = []
        tv = []
        for record in static["records"]:
            item = record["policies"][policy]
            seconds.append(float(item["seconds"]))
            probes.append(float(item["samples"]) + float(item.get("calibration_probes", 0)))
            tv.append(float(item["particle_id_tv"]))
        cost_rows.append(
            [
                policy,
                _fmt(float(np.median(seconds))),
                _fmt(float(np.median(probes))),
                _fmt(float(np.median(tv))),
            ]
        )
    cost_table = Table(cost_rows, colWidths=[2.3 * inch, 1.5 * inch, 1.5 * inch, 1.4 * inch])
    cost_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#20639B")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E0")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
            ]
        )
    )
    story.append(cost_table)
    story.append(PageBreak())

    story.append(Paragraph("3. Dynamic four-approach benchmark", styles["H1x"]))
    story.append(Image(str(dynamic_plot), width=7.1 * inch, height=2.65 * inch))
    dynamic_rows = [
        ["N", "Approach", "Median runtime (s)", "Median calibration probes", "Speedup vs exact"]
    ]
    for data in dynamics:
        particles = data["records"][0]["particles"]
        for approach, summary in data["summaries"].items():
            dynamic_rows.append(
                [
                    str(particles),
                    APPROACH_LABELS[approach],
                    _fmt(summary["runtime_median_seconds"]),
                    _fmt(summary["calibration_probes_median"]),
                    _fmt(summary["speedup_vs_exact"]),
                ]
            )
    dynamic_table = Table(
        dynamic_rows,
        colWidths=[0.55 * inch, 1.55 * inch, 1.35 * inch, 1.55 * inch, 1.1 * inch],
        repeatRows=1,
    )
    dynamic_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173F5F")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E0")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
            ]
        )
    )
    story.append(dynamic_table)
    story.append(
        Paragraph(
            "The dynamic run uses the package's four post-fix configurations at N = 1,200 and 3,000, seeds 42/137/911, and four Numba threads. Runtime differences reflect the intended scheduling trade-off: controlled-fixed spends substantial repeated calibration work, while paper-amortized spreads a certified schedule over growth. These runs compare work and certificates; three seeds are not evidence of morphology parity.",
            styles["Bodyx"],
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("4. Reproducibility and limitations", styles["H1x"]))
    story.append(Paragraph("Artifacts", styles["H2x"]))
    story.append(
        Paragraph(
            "The production manifest contains 720 shard records (4 targets x 9 radii x 20 replications), target digests, per-shard SHA-256 inventory, runtime provenance, deterministic batch seeds, geometry metadata, and the pilot decision. The pilot contains 180 records (4 x 9 x 5) and is revalidated before production. Dynamic outputs are the two benchmark JSON files consumed by this report.",
            styles["Bodyx"],
        )
    )
    story.append(Paragraph("What this demonstrates", styles["H2x"]))
    story.append(
        Paragraph(
            "The benchmark is a non-circular morphology stress test: it makes the target lopsided and increases the killing radius, exposing how a fixed center can retain finite-boundary bias while one-shot centering removes the leading geometric mismatch. It does not claim a theorem for non-circular death boundaries. The exact Poisson kernel remains optimal when exact return is available; the practical value of approach 4 is as a certified, lower-overhead approximation when exact return is unavailable or too costly.",
            styles["Bodyx"],
        )
    )
    story.append(Paragraph("Limitations", styles["H2x"]))
    story.append(
        Paragraph(
            "The static experiment uses circular death boundaries by design, finite nearest-particle and angular partitions, a fixed bounded-Lipschitz feature family, and only three stochastic DLA seeds. Wall times are machine-specific. The pilot's noise rule is mechanical and deliberately conservative; it triggered doubled evaluations for every tested radius in this run. Results should be read as validation of the predicted centering/error trend, not as a universal ranking for every geometry or workload.",
            styles["Bodyx"],
        )
    )
    story.append(Spacer(1, 0.15 * inch))
    story.append(
        Paragraph(
            "Generated from the canonical pilot, production, and dynamic benchmark JSON artifacts. Report date: 2026-07-26.",
            styles["Smallx"],
        )
    )
    document.build(story)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--static",
        type=Path,
        default=Path("output/asymmetric_validation/asymmetric_validation_production.json"),
    )
    parser.add_argument(
        "--pilot",
        type=Path,
        default=Path("output/asymmetric_validation/asymmetric_validation_pilot.json"),
    )
    parser.add_argument(
        "--dynamic-1200", type=Path, default=Path("output/benchmark/asymmetric_dynamic_1200.json")
    )
    parser.add_argument(
        "--dynamic-3000", type=Path, default=Path("output/benchmark/asymmetric_dynamic_3000.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/pdf/harmonic_dla_asymmetric_target_validation.pdf"),
    )
    parser.add_argument("--figure-dir", type=Path, default=Path("tmp/pdfs/asymmetric_validation"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_report(
        _load(args.static),
        _load(args.pilot),
        [_load(args.dynamic_1200), _load(args.dynamic_3000)],
        args.output,
        args.figure_dir,
    )
    print(args.output)


if __name__ == "__main__":
    main()
