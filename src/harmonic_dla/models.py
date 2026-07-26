"""Public result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
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
    calibration_bounds: FloatArray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    calibration_failure_probabilities: FloatArray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    calibration_death_ratios: FloatArray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    # Per-calibration audit fields.  They are optional at construction time so
    # archives written by older backends remain loadable; omitted values are
    # represented by a deterministic block index and zero probe split.
    calibration_block_indices: IntArray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    calibration_probe_counts: IntArray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    calibration_search_probes: IntArray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    calibration_validation_probes: IntArray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
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
        block_indices = np.asarray(self.calibration_block_indices, dtype=np.int64)
        probe_counts = np.asarray(self.calibration_probe_counts, dtype=np.int64)
        search_probes = np.asarray(self.calibration_search_probes, dtype=np.int64)
        validation_probes = np.asarray(self.calibration_validation_probes, dtype=np.int64)
        if centers.ndim != 2 or centers.shape[1] != 2:
            raise ValueError("calibration_centers must have shape (m, 2)")
        history_length = centers.shape[0]
        audit_names = {
            "calibration_block_indices",
            "calibration_probe_counts",
            "calibration_search_probes",
            "calibration_validation_probes",
        }
        for name, array in (
            ("calibration_sizes", sizes),
            ("calibration_bounds", bounds),
            ("calibration_failure_probabilities", failures),
            ("calibration_death_ratios", ratios),
            ("calibration_block_indices", block_indices),
            ("calibration_probe_counts", probe_counts),
            ("calibration_search_probes", search_probes),
            ("calibration_validation_probes", validation_probes),
        ):
            # The four audit arrays were added after schema version 1.  An
            # empty value means the caller is using the legacy constructor;
            # fill it below rather than making old backends fail validation.
            if name in audit_names and array.size == 0 and history_length:
                continue
            if array.ndim != 1 or array.shape[0] != history_length:
                raise ValueError(f"{name} must have shape ({history_length},)")
        if not np.all(np.isfinite(centers)):
            raise ValueError("calibration_centers must be finite")
        if np.any(sizes < 1):
            raise ValueError("calibration_sizes must be positive")
        if sizes.size > 1 and np.any(np.diff(sizes) <= 0):
            raise ValueError("calibration_sizes must be strictly increasing")
        if not np.all(np.isfinite(bounds)) or np.any((bounds < 0.0) | (bounds > 1.0)):
            raise ValueError("calibration_bounds must lie in [0, 1]")
        if not np.all(np.isfinite(failures)) or np.any((failures < 0.0) | (failures >= 1.0)):
            raise ValueError("calibration failure probabilities must lie in [0, 1)")
        if not np.all(np.isfinite(ratios)) or np.any(ratios <= 1.0):
            raise ValueError("calibration death ratios must exceed one")
        for name, array in (
            ("calibration_block_indices", block_indices),
            ("calibration_probe_counts", probe_counts),
            ("calibration_search_probes", search_probes),
            ("calibration_validation_probes", validation_probes),
        ):
            if array.size and np.any(array < 0):
                raise ValueError(f"{name} must be non-negative")

        if block_indices.size == 0:
            block_indices = np.arange(history_length, dtype=np.int64)
        if probe_counts.size == 0:
            probe_counts = np.zeros(history_length, dtype=np.int64)
        if search_probes.size == 0:
            search_probes = np.zeros(history_length, dtype=np.int64)
        if validation_probes.size == 0:
            validation_probes = np.zeros(history_length, dtype=np.int64)

        self.calibration_centers = np.ascontiguousarray(centers)
        self.calibration_sizes = np.ascontiguousarray(sizes)
        self.calibration_bounds = np.ascontiguousarray(bounds)
        self.calibration_failure_probabilities = np.ascontiguousarray(failures)
        self.calibration_death_ratios = np.ascontiguousarray(ratios)
        self.calibration_block_indices = np.ascontiguousarray(block_indices)
        self.calibration_probe_counts = np.ascontiguousarray(probe_counts)
        self.calibration_search_probes = np.ascontiguousarray(search_probes)
        self.calibration_validation_probes = np.ascontiguousarray(validation_probes)

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
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, (int, np.integer))
            or self.seed < 0
            or self.seed > np.iinfo(np.int64).max
        ):
            raise ValueError("seed must be a non-negative signed int64")
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


@dataclass(slots=True)
class ProbeResult:
    """Attachment samples and diagnostics from a frozen-cluster probe batch."""

    attachments: FloatArray
    walker_steps: int
    restarts: int

    def __post_init__(self) -> None:
        array = np.asarray(self.attachments, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < 1:
            raise ValueError("attachments must have shape (n, 2) with n >= 1")
        if not np.all(np.isfinite(array)):
            raise ValueError("attachments must contain only finite values")
        for name, value in (("walker_steps", self.walker_steps), ("restarts", self.restarts)):
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        self.attachments = np.ascontiguousarray(array)
        self.walker_steps = int(self.walker_steps)
        self.restarts = int(self.restarts)
