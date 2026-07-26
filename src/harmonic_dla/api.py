"""Stable public API for simulation and frozen-cluster probes."""

from __future__ import annotations

import math
from os import PathLike
from pathlib import Path

import numpy as np
import numpy.typing as npt

from harmonic_dla.backends.base import Backend
from harmonic_dla.config import RunConfig, load_config
from harmonic_dla.enums import BackendKind, RestartMode
from harmonic_dla.models import ProbeResult, SimulationResult

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


def _validate_probe_arguments(
    positions: FloatArray,
    particle_radius: float,
    center: tuple[float, float],
    death_ratio: float,
    launch_margin: float,
    probes: int,
    seed: int,
) -> FloatArray:
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
    if isinstance(probes, bool) or not isinstance(probes, (int, np.integer)) or probes < 1:
        raise ValueError("probes must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if seed < 0 or seed > np.iinfo(np.int64).max:
        raise ValueError("seed must be a non-negative signed int64")
    return array


def _validate_probe_restart_mode(restart_mode: RestartMode) -> None:
    if restart_mode not in (RestartMode.UNIFORM_RESTART, RestartMode.EXACT_RETURN):
        raise ValueError("restart_mode must be exact-return or uniform-restart")


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
    array = _validate_probe_arguments(
        positions,
        particle_radius,
        center,
        death_ratio,
        launch_margin,
        probes,
        seed,
    )
    return create_backend(backend).probe(
        array,
        particle_radius,
        center,
        death_ratio,
        launch_margin,
        probes,
        seed,
    )


def probe_detailed(
    positions: FloatArray,
    *,
    particle_radius: float,
    center: tuple[float, float] = (0.0, 0.0),
    death_ratio: float = 4.0,
    launch_margin: float = 4.0,
    probes: int = 1024,
    seed: int = 0,
    backend: BackendKind | str = BackendKind.NUMBA_CPU,
    restart_mode: RestartMode = RestartMode.UNIFORM_RESTART,
) -> ProbeResult:
    """Draw frozen-cluster probes with explicit return-law diagnostics."""
    array = _validate_probe_arguments(
        positions,
        particle_radius,
        center,
        death_ratio,
        launch_margin,
        probes,
        seed,
    )
    if not isinstance(restart_mode, RestartMode):
        raise ValueError("restart_mode must be exact-return or uniform-restart")
    _validate_probe_restart_mode(restart_mode)
    return create_backend(backend).probe_detailed(
        array,
        particle_radius,
        center,
        death_ratio,
        launch_margin,
        probes,
        seed,
        restart_mode,
    )
