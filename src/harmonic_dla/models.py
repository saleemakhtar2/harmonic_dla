"""Public result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(slots=True)
class SimulationDiagnostics:
    """Counters and calibration history collected during one run."""

    growth_walker_steps: int = 0
    growth_restarts: int = 0
    calibration_walker_steps: int = 0
    calibration_restarts: int = 0
    calibration_probes: int = 0
    calibration_centers: FloatArray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float64)
    )
    calibration_sizes: IntArray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    calibration_bounds: FloatArray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    calibration_failure_probabilities: FloatArray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    calibration_death_ratios: FloatArray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )

    def __post_init__(self) -> None:
        counters = (
            self.growth_walker_steps,
            self.growth_restarts,
            self.calibration_walker_steps,
            self.calibration_restarts,
            self.calibration_probes,
        )
        if any(value < 0 for value in counters):
            raise ValueError("diagnostic counters must be non-negative")

        centers = np.asarray(self.calibration_centers, dtype=np.float64)
        sizes = np.asarray(self.calibration_sizes, dtype=np.int64)
        bounds = np.asarray(self.calibration_bounds, dtype=np.float64)
        failures = np.asarray(self.calibration_failure_probabilities, dtype=np.float64)
        ratios = np.asarray(self.calibration_death_ratios, dtype=np.float64)
        if centers.ndim != 2 or centers.shape[1] != 2:
            raise ValueError("calibration_centers must have shape (m, 2)")
        history_length = centers.shape[0]
        for name, array in (
            ("calibration_sizes", sizes),
            ("calibration_bounds", bounds),
            ("calibration_failure_probabilities", failures),
            ("calibration_death_ratios", ratios),
        ):
            if array.ndim != 1 or array.shape[0] != history_length:
                raise ValueError(f"{name} must have shape ({history_length},)")
        if not np.all(np.isfinite(centers)):
            raise ValueError("calibration_centers must be finite")
        if np.any(sizes < 1):
            raise ValueError("calibration_sizes must be positive")
        if not np.all(np.isfinite(bounds)) or np.any((bounds < 0.0) | (bounds > 1.0)):
            raise ValueError("calibration_bounds must lie in [0, 1]")
        if not np.all(np.isfinite(failures)) or np.any((failures < 0.0) | (failures >= 1.0)):
            raise ValueError("calibration failure probabilities must lie in [0, 1)")
        if not np.all(np.isfinite(ratios)) or np.any(ratios <= 1.0):
            raise ValueError("calibration death ratios must exceed one")

        self.calibration_centers = np.ascontiguousarray(centers)
        self.calibration_sizes = np.ascontiguousarray(sizes)
        self.calibration_bounds = np.ascontiguousarray(bounds)
        self.calibration_failure_probabilities = np.ascontiguousarray(failures)
        self.calibration_death_ratios = np.ascontiguousarray(ratios)

    @property
    def total_walker_steps(self) -> int:
        """Growth and calibration walk-on-spheres steps combined."""
        return self.growth_walker_steps + self.calibration_walker_steps

    @property
    def total_restarts(self) -> int:
        """Growth and calibration outer-boundary events combined."""
        return self.growth_restarts + self.calibration_restarts


@dataclass(slots=True)
class SimulationResult:
    """An in-memory aggregate and its reproducibility metadata."""

    positions: FloatArray
    particle_radius: float
    seed: int
    backend: str
    restart_mode: str
    diagnostics: SimulationDiagnostics
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        array = np.asarray(self.positions, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < 1:
            raise ValueError("positions must have shape (n, 2) with n >= 1")
        if not np.all(np.isfinite(array)):
            raise ValueError("positions must contain only finite values")
        if not np.isfinite(self.particle_radius) or self.particle_radius <= 0.0:
            raise ValueError("particle_radius must be finite and positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.backend:
            raise ValueError("backend must be non-empty")
        if not self.restart_mode:
            raise ValueError("restart_mode must be non-empty")
        self.positions = np.ascontiguousarray(array)

    @property
    def particle_count(self) -> int:
        """Number of particles in the returned aggregate."""
        return int(self.positions.shape[0])

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Save the result as a compressed, pickle-free NumPy archive."""
        from harmonic_dla.io import save_result

        return save_result(self, path, overwrite=overwrite)
