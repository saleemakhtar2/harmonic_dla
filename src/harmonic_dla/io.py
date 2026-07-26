"""Safe, versioned persistence for simulation results."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from harmonic_dla.exceptions import CheckpointError
from harmonic_dla.models import SimulationDiagnostics, SimulationResult

_SCHEMA_VERSION = 1


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _normalise_path(path: str | Path) -> Path:
    output = Path(path)
    if output.suffix.lower() != ".npz":
        output = output.with_suffix(".npz")
    return output


def save_result(
    result: SimulationResult,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically save a result without Python pickle payloads."""
    output = _normalise_path(path)
    if output.exists() and not overwrite:
        raise CheckpointError(f"refusing to overwrite existing file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    metadata = dict(result.metadata)
    metadata.setdefault("schema_version", _SCHEMA_VERSION)
    metadata_text = json.dumps(metadata, sort_keys=True, default=_json_default)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            diagnostics = result.diagnostics
            np.savez_compressed(
                stream,
                schema_version=np.asarray(_SCHEMA_VERSION, dtype=np.int64),
                positions=np.ascontiguousarray(result.positions, dtype=np.float64),
                particle_radius=np.asarray(result.particle_radius, dtype=np.float64),
                seed=np.asarray(result.seed, dtype=np.int64),
                backend=np.asarray(result.backend),
                restart_mode=np.asarray(result.restart_mode),
                growth_walker_steps=np.asarray(
                    diagnostics.growth_walker_steps,
                    dtype=np.int64,
                ),
                growth_restarts=np.asarray(diagnostics.growth_restarts, dtype=np.int64),
                calibration_walker_steps=np.asarray(
                    diagnostics.calibration_walker_steps,
                    dtype=np.int64,
                ),
                calibration_restarts=np.asarray(
                    diagnostics.calibration_restarts,
                    dtype=np.int64,
                ),
                calibration_probes=np.asarray(diagnostics.calibration_probes, dtype=np.int64),
                calibration_centers=np.ascontiguousarray(
                    diagnostics.calibration_centers,
                    dtype=np.float64,
                ),
                calibration_sizes=np.ascontiguousarray(
                    diagnostics.calibration_sizes,
                    dtype=np.int64,
                ),
                calibration_bounds=np.ascontiguousarray(
                    diagnostics.calibration_bounds,
                    dtype=np.float64,
                ),
                calibration_failure_probabilities=np.ascontiguousarray(
                    diagnostics.calibration_failure_probabilities,
                    dtype=np.float64,
                ),
                calibration_death_ratios=np.ascontiguousarray(
                    diagnostics.calibration_death_ratios,
                    dtype=np.float64,
                ),
                metadata_json=np.asarray(metadata_text),
            )
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        if isinstance(exc, CheckpointError):
            raise
        raise CheckpointError(f"failed to save result to {output}: {exc}") from exc
    return output


def _scalar_text(archive: np.lib.npyio.NpzFile, key: str) -> str:
    value = archive[key]
    return str(value.item())


def load_result(path: str | Path) -> SimulationResult:
    """Load a result archive and validate its schema and core array shapes."""
    source = _normalise_path(path)
    try:
        with np.load(source, allow_pickle=False) as archive:
            schema = int(archive["schema_version"].item())
            if schema != _SCHEMA_VERSION:
                raise CheckpointError(
                    f"unsupported result schema {schema}; expected {_SCHEMA_VERSION}"
                )
            metadata_raw = _scalar_text(archive, "metadata_json")
            metadata_value: Any = json.loads(metadata_raw)
            if not isinstance(metadata_value, dict):
                raise CheckpointError("metadata_json must decode to an object")
            diagnostics = SimulationDiagnostics(
                growth_walker_steps=int(archive["growth_walker_steps"].item()),
                growth_restarts=int(archive["growth_restarts"].item()),
                calibration_walker_steps=int(archive["calibration_walker_steps"].item()),
                calibration_restarts=int(archive["calibration_restarts"].item()),
                calibration_probes=int(archive["calibration_probes"].item()),
                calibration_centers=np.asarray(
                    archive["calibration_centers"],
                    dtype=np.float64,
                ).copy(),
                calibration_sizes=np.asarray(
                    archive["calibration_sizes"],
                    dtype=np.int64,
                ).copy(),
                calibration_bounds=np.asarray(
                    archive["calibration_bounds"],
                    dtype=np.float64,
                ).copy(),
                calibration_failure_probabilities=np.asarray(
                    archive["calibration_failure_probabilities"],
                    dtype=np.float64,
                ).copy(),
                calibration_death_ratios=np.asarray(
                    archive["calibration_death_ratios"],
                    dtype=np.float64,
                ).copy(),
            )
            return SimulationResult(
                positions=np.asarray(archive["positions"], dtype=np.float64).copy(),
                particle_radius=float(archive["particle_radius"].item()),
                seed=int(archive["seed"].item()),
                backend=_scalar_text(archive, "backend"),
                restart_mode=_scalar_text(archive, "restart_mode"),
                diagnostics=diagnostics,
                metadata=metadata_value,
            )
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError(f"failed to load result from {source}: {exc}") from exc
