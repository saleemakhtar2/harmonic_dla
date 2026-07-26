"""Stable public API for simulation and frozen-cluster probes."""

from __future__ import annotations

import math
from os import PathLike
from pathlib import Path

import numpy as np
import numpy.typing as npt

from harmonic_dla.backends.base import Backend
from harmonic_dla.config import RunConfig, load_config
from harmonic_dla.enums import BackendKind
from harmonic_dla.models import SimulationResult

FloatArray = npt.NDArray[np.float64]
ConfigLike = RunConfig | str | PathLike[str]


def create_backend(kind: BackendKind | str) -> Backend:
    """Construct a backend lazily so importing the package remains lightweight."""
    backend_kind = kind if isinstance(kind, BackendKind) else BackendKind(str(kind))
    if backend_kind is BackendKind.NUMBA_CPU:
        from harmonic_dla.backends.numba_cpu import NumbaCPUBackend

        return NumbaCPUBackend()
    from harmonic_dla.backends.reference import ReferenceBackend

    return ReferenceBackend()


def resolve_config(config: ConfigLike) -> RunConfig:
    """Return a validated ``RunConfig`` from an object or TOML path."""
    if isinstance(config, RunConfig):
        return config
    return load_config(Path(config))


def simulate(config: ConfigLike) -> SimulationResult:
    """Grow one aggregate and return it in memory.

    Persistence is explicit: call ``result.save(...)`` or use the ``hdla run`` CLI.
    """
    resolved = resolve_config(config)
    return create_backend(resolved.backend).simulate(resolved)


def probe(
    positions: FloatArray,
    *,
    particle_radius: float,
    center: tuple[float, float] = (0.0, 0.0),
    death_ratio: float = 4.0,
    launch_margin: float = 4.0,
    probes: int = 1024,
    seed: int = 0,
    backend: BackendKind | str = BackendKind.NUMBA_CPU,
) -> FloatArray:
    """Draw target-radius-scaled uniform-restart attachment probes."""
    array = np.ascontiguousarray(positions, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < 1:
        raise ValueError("positions must have shape (n, 2) with n >= 1")
    if not np.all(np.isfinite(array)):
        raise ValueError("positions must contain only finite values")
    if not math.isfinite(particle_radius) or particle_radius <= 0.0:
        raise ValueError("particle_radius must be finite and positive")
    if len(center) != 2 or not all(math.isfinite(value) for value in center):
        raise ValueError("center must contain two finite coordinates")
    if not math.isfinite(death_ratio) or death_ratio <= 1.0:
        raise ValueError("death_ratio must be finite and greater than one")
    if not math.isfinite(launch_margin) or launch_margin <= 0.0:
        raise ValueError("launch_margin must be finite and positive")
    if probes < 1:
        raise ValueError("probes must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return create_backend(backend).probe(
        array,
        particle_radius,
        center,
        death_ratio,
        launch_margin,
        probes,
        seed,
    )
