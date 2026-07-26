"""Command-line interface for reproducible simulation and certificates."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from harmonic_dla import __version__
from harmonic_dla.analysis import bounding_box_diameter_upper, radius_about
from harmonic_dla.api import simulate
from harmonic_dla.certificates import (
    constant_ratio_for_path_budget,
    monte_carlo_tv_bound,
    residual_tv_bound,
    self_centered_tv_bound,
)
from harmonic_dla.config import (
    BoundaryConfig,
    OutputConfig,
    RunConfig,
    WalkerConfig,
    load_config,
)
from harmonic_dla.enums import BackendKind, RestartMode
from harmonic_dla.exceptions import HarmonicDLAError
from harmonic_dla.io import load_result


def _summary(result_path: Path) -> dict[str, Any]:
    result = load_result(result_path)
    center_raw = result.metadata.get("final_center", [0.0, 0.0])
    center = (float(center_raw[0]), float(center_raw[1]))
    return {
        "path": str(result_path),
        "particles": result.particle_count,
        "particle_radius": result.particle_radius,
        "backend": result.backend,
        "restart_mode": result.restart_mode,
        "seed": result.seed,
        "target_radius_about_final_center": radius_about(result.positions, center)
        + 2.0 * result.particle_radius,
        "bounding_box_diameter_upper": bounding_box_diameter_upper(
            result.positions,
            result.particle_radius,
        ),
        "growth_walker_steps": result.diagnostics.growth_walker_steps,
        "growth_restarts": result.diagnostics.growth_restarts,
        "calibration_probes": result.diagnostics.calibration_probes,
        "calibration_walker_steps": result.diagnostics.calibration_walker_steps,
        "metadata": result.metadata,
    }


def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    output = Path(args.output) if args.output else config.output.path
    overwrite = bool(args.overwrite or config.output.overwrite)
    config = replace(config, output=replace(config.output, path=output, overwrite=overwrite))
    result = simulate(config)
    saved = result.save(output, overwrite=overwrite)
    print(json.dumps(_summary(saved), indent=2, sort_keys=True))
    return 0


def _inspect(args: argparse.Namespace) -> int:
    summary = _summary(Path(args.result))
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        for key, value in summary.items():
            if key != "metadata":
                print(f"{key}: {value}")
        print("metadata:")
        print(json.dumps(summary["metadata"], indent=2, sort_keys=True))
    return 0


def _check_config(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    print(repr(config))
    return 0


def _certificate(args: argparse.Namespace) -> int:
    if args.kind == "self-centered":
        value = self_centered_tv_bound(args.rho)
    elif args.kind == "residual":
        value = residual_tv_bound(args.rho, args.residual_over_radius)
    elif args.kind == "monte-carlo":
        value = monte_carlo_tv_bound(
            args.rho,
            args.residual_over_radius,
            args.probes,
            args.failure_probability,
        )
    else:
        value = constant_ratio_for_path_budget(args.steps, args.budget)
    print(json.dumps({"value": value}, sort_keys=True))
    return 0


def _warmup(args: argparse.Namespace) -> int:
    config = RunConfig(
        particles=args.particles,
        seed=args.seed,
        backend=BackendKind.NUMBA_CPU,
        walker=WalkerConfig(tolerance=1.0e-5, max_steps=10_000, max_restarts=10_000),
        boundary=BoundaryConfig(
            restart_mode=RestartMode.EXACT_RETURN,
            death_ratio=3.0,
            launch_margin=3.0,
        ),
        output=OutputConfig(path=Path("warmup.npz"), overwrite=True),
    )
    result = simulate(config)
    print(
        json.dumps(
            {
                "compiled": True,
                "particles": result.particle_count,
                "elapsed_seconds": result.metadata.get("elapsed_seconds"),
            },
            sort_keys=True,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hdla",
        description="Accuracy-controlled planar off-lattice DLA",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run a TOML-configured simulation")
    run.add_argument("config", type=Path)
    run.add_argument("--output", type=Path)
    run.add_argument("--overwrite", action="store_true")
    run.set_defaults(handler=_run)

    inspect = subparsers.add_parser("inspect", help="inspect a saved result")
    inspect.add_argument("result", type=Path)
    inspect.add_argument("--json", action="store_true")
    inspect.set_defaults(handler=_inspect)

    check = subparsers.add_parser("check-config", help="parse and validate a TOML config")
    check.add_argument("config", type=Path)
    check.set_defaults(handler=_check_config)

    certificate = subparsers.add_parser("certificate", help="evaluate a paper certificate")
    certificate_sub = certificate.add_subparsers(dest="kind", required=True)

    centered = certificate_sub.add_parser("self-centered")
    centered.add_argument("--rho", type=float, required=True)
    centered.set_defaults(handler=_certificate)

    residual = certificate_sub.add_parser("residual")
    residual.add_argument("--rho", type=float, required=True)
    residual.add_argument("--residual-over-radius", type=float, required=True)
    residual.set_defaults(handler=_certificate)

    monte_carlo = certificate_sub.add_parser("monte-carlo")
    monte_carlo.add_argument("--rho", type=float, required=True)
    monte_carlo.add_argument("--residual-over-radius", type=float, required=True)
    monte_carlo.add_argument("--probes", type=int, required=True)
    monte_carlo.add_argument("--failure-probability", type=float, required=True)
    monte_carlo.set_defaults(handler=_certificate)

    path_ratio = certificate_sub.add_parser("path-ratio")
    path_ratio.add_argument("--steps", type=int, required=True)
    path_ratio.add_argument("--budget", type=float, required=True)
    path_ratio.set_defaults(handler=_certificate)

    warmup = subparsers.add_parser("warmup", help="compile Numba kernels with a tiny run")
    warmup.add_argument("--particles", type=int, default=8)
    warmup.add_argument("--seed", type=int, default=0)
    warmup.set_defaults(handler=_warmup)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except (HarmonicDLAError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
