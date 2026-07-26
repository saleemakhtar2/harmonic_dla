"""Validated configuration objects and TOML loading."""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harmonic_dla.enums import BackendKind, CalibrationStrategy, RestartMode
from harmonic_dla.exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class WalkerConfig:
    """Numerical controls for walk-on-spheres trajectories."""

    tolerance: float = 1.0e-6
    max_steps: int = 100_000
    max_restarts: int = 100_000

    def __post_init__(self) -> None:
        if not 0.0 < self.tolerance < 0.1:
            raise ConfigurationError("walker.tolerance must lie in (0, 0.1)")
        if self.max_steps < 10:
            raise ConfigurationError("walker.max_steps must be at least 10")
        if self.max_restarts < 1:
            raise ConfigurationError("walker.max_restarts must be positive")


@dataclass(frozen=True, slots=True)
class BoundaryConfig:
    """Launch/death boundary policy for fixed-ratio runs."""

    restart_mode: RestartMode = RestartMode.EXACT_RETURN
    death_ratio: float = 4.0
    launch_margin: float = 4.0
    center: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if not math.isfinite(self.death_ratio) or self.death_ratio <= 1.0:
            raise ConfigurationError("boundary.death_ratio must be finite and exceed 1")
        if not math.isfinite(self.launch_margin) or self.launch_margin <= 0.0:
            raise ConfigurationError("boundary.launch_margin must be finite and positive")
        if len(self.center) != 2 or not all(math.isfinite(value) for value in self.center):
            raise ConfigurationError("boundary.center must contain two finite coordinates")


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """Frozen-probe controls for controlled restart."""

    enabled: bool = False
    strategy: CalibrationStrategy = CalibrationStrategy.SAMPLE_SPLIT_FIXED
    search_probes: int = 2048
    validation_probes: int = 2048
    block_size: int = 128
    confidence_failure: float = 1.0e-3
    pilot: str = "previous-center"

    # Parameters of the paper's diameter-scaled amortized schedule.
    alpha: float = 0.48
    gamma: float = 0.06
    scale: float = 2.0
    start: int = 1024
    failure_budget: float = 1.0e-3
    exact_prefix: bool = True

    def __post_init__(self) -> None:
        if self.search_probes < 2:
            raise ConfigurationError("calibration.search_probes must be at least 2")
        if self.validation_probes < 2:
            raise ConfigurationError("calibration.validation_probes must be at least 2")
        if self.block_size < 1:
            raise ConfigurationError("calibration.block_size must be positive")
        if not 0.0 < self.confidence_failure < 1.0:
            raise ConfigurationError("calibration.confidence_failure must lie in (0, 1)")
        if self.pilot not in {"previous-center", "seed", "mass-centroid"}:
            raise ConfigurationError(
                "calibration.pilot must be previous-center, seed, or mass-centroid"
            )
        if self.strategy is CalibrationStrategy.PAPER_AMORTIZED:
            if not all(
                math.isfinite(value)
                for value in (
                    self.alpha,
                    self.gamma,
                    self.scale,
                    self.failure_budget,
                )
            ):
                raise ConfigurationError("paper-amortized parameters must be finite")
            if self.alpha <= 11.0 / 24.0:
                raise ConfigurationError("paper-amortized alpha must exceed 11/24")
            lower = max(0.0, 1.0 - 2.0 * self.alpha)
            if not lower < self.gamma < 1.0 / 12.0:
                raise ConfigurationError(
                    "paper-amortized gamma must lie in (max(0, 1-2alpha), 1/12)"
                )
            if self.scale < math.sqrt(2.0):
                raise ConfigurationError("paper-amortized scale must be at least sqrt(2)")
            kappa = 0.25 - self.gamma
            minimum_start = max(16, math.ceil(2.0 ** (1.0 / kappa)))
            if self.start < minimum_start:
                raise ConfigurationError(
                    "paper-amortized start must be at least "
                    f"{minimum_start} for the equal-disk theorem constants"
                )
            if not 0.0 < self.failure_budget < 1.0:
                raise ConfigurationError("calibration.failure_budget must lie in (0, 1)")


@dataclass(frozen=True, slots=True)
class PerformanceConfig:
    """Runtime and memory controls."""

    threads: int = 0
    quadtree_bucket_size: int = 12
    quadtree_node_factor: int = 10
    growth_chunk_size: int = 0

    def __post_init__(self) -> None:
        if self.threads < 0:
            raise ConfigurationError("performance.threads cannot be negative")
        if not 4 <= self.quadtree_bucket_size <= 64:
            raise ConfigurationError("performance.quadtree_bucket_size must lie in [4, 64]")
        if not 6 <= self.quadtree_node_factor <= 32:
            raise ConfigurationError("performance.quadtree_node_factor must lie in [6, 32]")
        if self.growth_chunk_size < 0:
            raise ConfigurationError("performance.growth_chunk_size cannot be negative")


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Result and checkpoint persistence."""

    path: Path = Path("cluster.npz")
    checkpoint_every: int = 0
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.checkpoint_every < 0:
            raise ConfigurationError("output.checkpoint_every cannot be negative")


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Complete simulator configuration."""

    particles: int = 10_000
    particle_radius: float = 0.5
    seed: int = 0
    backend: BackendKind = BackendKind.NUMBA_CPU
    walker: WalkerConfig = field(default_factory=WalkerConfig)
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self) -> None:
        if self.particles < 2:
            raise ConfigurationError("particles must be at least 2")
        if not math.isfinite(self.particle_radius) or self.particle_radius <= 0.0:
            raise ConfigurationError("particle_radius must be finite and positive")
        if self.seed < 0:
            raise ConfigurationError("seed must be non-negative")
        controlled = self.boundary.restart_mode is RestartMode.CONTROLLED_RESTART
        if controlled and not self.calibration.enabled:
            raise ConfigurationError("controlled restart requires calibration.enabled = true")
        if not controlled and self.calibration.enabled:
            raise ConfigurationError(
                "calibration.enabled is only valid with boundary.restart_mode = controlled-restart"
            )


def _table(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{key!r} must be a TOML table")
    return value


def _backend_kind(value: Any) -> BackendKind:
    try:
        return BackendKind(str(value))
    except ValueError as exc:
        allowed = ", ".join(item.value for item in BackendKind)
        raise ConfigurationError(
            f"unknown backend {value!r}; choose one of: {allowed}"
        ) from exc


def _calibration_strategy(value: Any) -> CalibrationStrategy:
    try:
        return CalibrationStrategy(str(value))
    except ValueError as exc:
        allowed = ", ".join(item.value for item in CalibrationStrategy)
        raise ConfigurationError(
            f"unknown calibration strategy {value!r}; choose one of: {allowed}"
        ) from exc


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a run configuration from TOML."""
    config_path = Path(path)
    with config_path.open("rb") as stream:
        raw = tomllib.load(stream)

    walker_raw = _table(raw, "walker")
    boundary_raw = _table(raw, "boundary")
    calibration_raw = _table(raw, "calibration")
    performance_raw = _table(raw, "performance")
    output_raw = _table(raw, "output")

    backend = _backend_kind(raw.get("backend", BackendKind.NUMBA_CPU.value))
    strategy = _calibration_strategy(
        calibration_raw.get("strategy", CalibrationStrategy.SAMPLE_SPLIT_FIXED.value)
    )

    restart_text = str(boundary_raw.get("restart_mode", "exact-return"))
    try:
        restart_mode = RestartMode.from_text(restart_text)
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc

    center_raw = boundary_raw.get("center", [0.0, 0.0])
    if not isinstance(center_raw, list) or len(center_raw) != 2:
        raise ConfigurationError("boundary.center must be a two-element TOML array")

    return RunConfig(
        particles=int(raw.get("particles", 10_000)),
        particle_radius=float(raw.get("particle_radius", 0.5)),
        seed=int(raw.get("seed", 0)),
        backend=backend,
        walker=WalkerConfig(
            tolerance=float(walker_raw.get("tolerance", 1.0e-6)),
            max_steps=int(walker_raw.get("max_steps", 100_000)),
            max_restarts=int(walker_raw.get("max_restarts", 100_000)),
        ),
        boundary=BoundaryConfig(
            restart_mode=restart_mode,
            death_ratio=float(boundary_raw.get("death_ratio", 4.0)),
            launch_margin=float(boundary_raw.get("launch_margin", 4.0)),
            center=(float(center_raw[0]), float(center_raw[1])),
        ),
        calibration=CalibrationConfig(
            enabled=bool(calibration_raw.get("enabled", False)),
            strategy=strategy,
            search_probes=int(calibration_raw.get("search_probes", 2048)),
            validation_probes=int(calibration_raw.get("validation_probes", 2048)),
            block_size=int(calibration_raw.get("block_size", 128)),
            confidence_failure=float(calibration_raw.get("confidence_failure", 1.0e-3)),
            pilot=str(calibration_raw.get("pilot", "previous-center")),
            alpha=float(calibration_raw.get("alpha", 0.48)),
            gamma=float(calibration_raw.get("gamma", 0.06)),
            scale=float(calibration_raw.get("scale", 2.0)),
            start=int(calibration_raw.get("start", 1024)),
            failure_budget=float(calibration_raw.get("failure_budget", 1.0e-3)),
            exact_prefix=bool(calibration_raw.get("exact_prefix", True)),
        ),
        performance=PerformanceConfig(
            threads=int(performance_raw.get("threads", 0)),
            quadtree_bucket_size=int(performance_raw.get("quadtree_bucket_size", 12)),
            quadtree_node_factor=int(performance_raw.get("quadtree_node_factor", 10)),
            growth_chunk_size=int(performance_raw.get("growth_chunk_size", 0)),
        ),
        output=OutputConfig(
            path=Path(str(output_raw.get("path", "cluster.npz"))),
            checkpoint_every=int(output_raw.get("checkpoint_every", 0)),
            overwrite=bool(output_raw.get("overwrite", False)),
        ),
    )


def config_to_dict(config: RunConfig) -> dict[str, Any]:
    """Convert a validated configuration to JSON-compatible primitives."""
    return {
        "particles": config.particles,
        "particle_radius": config.particle_radius,
        "seed": config.seed,
        "backend": config.backend.value,
        "walker": {
            "tolerance": config.walker.tolerance,
            "max_steps": config.walker.max_steps,
            "max_restarts": config.walker.max_restarts,
        },
        "boundary": {
            "restart_mode": config.boundary.restart_mode.name.lower(),
            "death_ratio": config.boundary.death_ratio,
            "launch_margin": config.boundary.launch_margin,
            "center": list(config.boundary.center),
        },
        "calibration": {
            "enabled": config.calibration.enabled,
            "strategy": config.calibration.strategy.value,
            "search_probes": config.calibration.search_probes,
            "validation_probes": config.calibration.validation_probes,
            "block_size": config.calibration.block_size,
            "confidence_failure": config.calibration.confidence_failure,
            "pilot": config.calibration.pilot,
            "alpha": config.calibration.alpha,
            "gamma": config.calibration.gamma,
            "scale": config.calibration.scale,
            "start": config.calibration.start,
            "failure_budget": config.calibration.failure_budget,
            "exact_prefix": config.calibration.exact_prefix,
        },
        "performance": {
            "threads": config.performance.threads,
            "quadtree_bucket_size": config.performance.quadtree_bucket_size,
            "quadtree_node_factor": config.performance.quadtree_node_factor,
            "growth_chunk_size": config.performance.growth_chunk_size,
        },
        "output": {
            "path": str(config.output.path),
            "checkpoint_every": config.output.checkpoint_every,
            "overwrite": config.output.overwrite,
        },
    }
