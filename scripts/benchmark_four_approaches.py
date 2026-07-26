"""Reproducible whole-run benchmark for the four boundary approaches.

The controller launches every measurement in a fresh process. Each worker warms
the relevant Numba path before timing, so reported runtime excludes compilation
while peak RSS remains isolated to one approach/run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from harmonic_dla.api import simulate
from harmonic_dla.certificates import amortized_path_budget_upper, path_budget
from harmonic_dla.config import (
    BoundaryConfig,
    CalibrationConfig,
    PerformanceConfig,
    RunConfig,
)
from harmonic_dla.enums import CalibrationStrategy, RestartMode

APPROACHES = ("exact", "uniform", "controlled-fixed", "paper-amortized")


def _config(approach: str, particles: int, seed: int, threads: int, *, warm: bool) -> RunConfig:
    performance = PerformanceConfig(threads=threads)
    if approach == "exact":
        return RunConfig(particles=particles, seed=seed, performance=performance)
    if approach == "uniform":
        return RunConfig(
            particles=particles,
            seed=seed,
            boundary=BoundaryConfig(restart_mode=RestartMode.UNIFORM_RESTART),
            performance=performance,
        )
    if approach == "controlled-fixed":
        probes = 16 if warm else 4096
        block_size = 8 if warm else 256
        return RunConfig(
            particles=particles,
            seed=seed,
            boundary=BoundaryConfig(restart_mode=RestartMode.CONTROLLED_RESTART),
            calibration=CalibrationConfig(
                enabled=True,
                strategy=CalibrationStrategy.SAMPLE_SPLIT_FIXED,
                search_probes=probes,
                validation_probes=probes,
                block_size=block_size,
                confidence_failure=1.0e-4,
            ),
            performance=performance,
        )
    if approach == "paper-amortized":
        return RunConfig(
            particles=particles,
            seed=seed,
            boundary=BoundaryConfig(restart_mode=RestartMode.CONTROLLED_RESTART),
            calibration=CalibrationConfig(
                enabled=True,
                strategy=CalibrationStrategy.PAPER_AMORTIZED,
                alpha=0.48,
                gamma=0.06,
                scale=32.0,
                start=64 if warm else 1024,
                failure_budget=1.0e-3,
                exact_prefix=True,
            ),
            performance=performance,
        )
    raise ValueError(f"unknown approach: {approach}")


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _shape_metrics(positions: np.ndarray) -> dict[str, float]:
    center = np.mean(positions, axis=0)
    centered = positions - center
    covariance = centered.T @ centered / positions.shape[0]
    eigenvalues = np.linalg.eigvalsh(covariance)
    radii = np.linalg.norm(positions, axis=1)
    anisotropy = float(eigenvalues[-1] / max(eigenvalues[0], np.finfo(float).tiny))
    return {
        "radius_of_gyration": float(math.sqrt(float(np.trace(covariance)))),
        "max_seed_radius": float(np.max(radii)),
        "center_of_mass_offset": float(np.linalg.norm(center)),
        "anisotropy_ratio": anisotropy,
    }


def _conditional_path_bound(result: Any) -> float | None:
    diagnostics = result.diagnostics
    if diagnostics.calibration_bounds.size == 0:
        return 0.0 if result.restart_mode == "exact_return" else None
    bounds: list[float] = []
    sizes = diagnostics.calibration_sizes.tolist()
    for index, (start, bound) in enumerate(
        zip(sizes, diagnostics.calibration_bounds.tolist(), strict=True)
    ):
        stop = sizes[index + 1] if index + 1 < len(sizes) else result.particle_count
        bounds.extend([float(bound)] * max(0, int(stop) - int(start)))
    conditional = path_budget(bounds) if bounds else 0.0
    # Calibration confidence failures are block-level events and must be
    # charged once per calibration, not repeated for every attachment.
    calibration_failures = float(
        np.sum(diagnostics.calibration_failure_probabilities, dtype=np.float64)
    )
    return min(1.0, conditional + calibration_failures)


def _worker(args: argparse.Namespace) -> None:
    warm_particles = 72
    simulate(_config(args.approach, warm_particles, args.seed, args.threads, warm=True))

    config = _config(args.approach, args.particles, args.seed, args.threads, warm=False)
    started = time.perf_counter()
    result = simulate(config)
    elapsed = time.perf_counter() - started
    diagnostics = result.diagnostics
    record: dict[str, Any] = {
        "approach": args.approach,
        "particles": args.particles,
        "seed": args.seed,
        "threads": args.threads,
        "elapsed_seconds": elapsed,
        "particles_per_second": args.particles / elapsed,
        "peak_rss_bytes": _peak_rss_bytes(),
        "growth_walker_steps": diagnostics.growth_walker_steps,
        "growth_restarts": diagnostics.growth_restarts,
        "calibration_walker_steps": diagnostics.calibration_walker_steps,
        "calibration_restarts": diagnostics.calibration_restarts,
        "calibration_probes": diagnostics.calibration_probes,
        "calibration_events": int(diagnostics.calibration_sizes.size),
        "max_one_step_bound": (
            float(np.max(diagnostics.calibration_bounds))
            if diagnostics.calibration_bounds.size
            else (0.0 if args.approach == "exact" else None)
        ),
        "conditional_path_bound": _conditional_path_bound(result),
        "shape": _shape_metrics(result.positions),
        "representative_positions": (result.positions.tolist() if args.seed == 42 else None),
    }
    if args.approach == "paper-amortized":
        record["infinite_history_bound_upper"] = amortized_path_budget_upper(
            alpha=config.calibration.alpha,
            gamma=config.calibration.gamma,
            scale=config.calibration.scale,
            start=config.calibration.start,
            calibration_failure_budget=config.calibration.failure_budget,
        )
    print(json.dumps(record, separators=(",", ":")))


def _environment(threads: int) -> dict[str, Any]:
    import numba

    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "threads": threads,
        "numpy": np.__version__,
        "numba": numba.__version__,
        "benchmark_script": str(Path(__file__).resolve()),
    }


def _summaries(records: list[dict[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for approach in APPROACHES:
        group = [record for record in records if record["approach"] == approach]
        runtime = [float(record["elapsed_seconds"]) for record in group]
        throughput = [float(record["particles_per_second"]) for record in group]
        rss = [int(record["peak_rss_bytes"]) for record in group]
        summaries[approach] = {
            "runs": len(group),
            "runtime_median_seconds": statistics.median(runtime),
            "runtime_min_seconds": min(runtime),
            "runtime_max_seconds": max(runtime),
            "throughput_median_particles_per_second": statistics.median(throughput),
            "peak_rss_median_bytes": statistics.median(rss),
            "growth_steps_median": statistics.median(
                int(record["growth_walker_steps"]) for record in group
            ),
            "growth_restarts_median": statistics.median(
                int(record["growth_restarts"]) for record in group
            ),
            "calibration_steps_median": statistics.median(
                int(record["calibration_walker_steps"]) for record in group
            ),
            "calibration_probes_median": statistics.median(
                int(record["calibration_probes"]) for record in group
            ),
            "calibration_events_median": statistics.median(
                int(record["calibration_events"]) for record in group
            ),
            "max_one_step_bound_median": (
                statistics.median(
                    float(record["max_one_step_bound"])
                    for record in group
                    if record["max_one_step_bound"] is not None
                )
                if any(record["max_one_step_bound"] is not None for record in group)
                else None
            ),
            "conditional_path_bound_median": (
                statistics.median(
                    float(record["conditional_path_bound"])
                    for record in group
                    if record["conditional_path_bound"] is not None
                )
                if any(record["conditional_path_bound"] is not None for record in group)
                else None
            ),
            "shape": {
                key: statistics.median(float(record["shape"][key]) for record in group)
                for key in group[0]["shape"]
            },
        }
    exact_runtime = summaries["exact"]["runtime_median_seconds"]
    for summary in summaries.values():
        summary["speedup_vs_exact"] = exact_runtime / summary["runtime_median_seconds"]
    return summaries


def _controller(args: argparse.Namespace) -> None:
    script = Path(__file__).resolve()
    records: list[dict[str, Any]] = []
    for approach in APPROACHES:
        for seed in args.seeds:
            command = [
                sys.executable,
                str(script),
                "--worker",
                "--approach",
                approach,
                "--particles",
                str(args.particles),
                "--seed",
                str(seed),
                "--threads",
                str(args.threads),
            ]
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(script.parents[1] / "src")},
            )
            record = json.loads(completed.stdout.strip().splitlines()[-1])
            records.append(record)
            print(
                f"{approach:18s} seed={seed:4d} "
                f"time={record['elapsed_seconds']:.3f}s "
                f"rate={record['particles_per_second']:.1f}/s",
                flush=True,
            )

    payload = {
        "methodology": {
            "particles": args.particles,
            "seeds": args.seeds,
            "repeats": len(args.seeds),
            "warmup": "approach-specific Numba path, excluded from elapsed time",
            "process_isolation": True,
            "comparison_scope": (
                "equal particle count and numerical settings; not a completed "
                "matched-accuracy ensemble campaign"
            ),
        },
        "environment": _environment(args.threads),
        "records": records,
        "summaries": _summaries(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--approach", choices=APPROACHES)
    parser.add_argument("--particles", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 911])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/benchmark/four_approaches_raw.json"),
    )
    args = parser.parse_args()
    if args.worker:
        if args.approach is None:
            parser.error("--approach is required with --worker")
        _worker(args)
    else:
        _controller(args)


if __name__ == "__main__":
    main()
