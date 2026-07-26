"""Safe, versioned persistence for simulation results."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np

from harmonic_dla.exceptions import CheckpointError
from harmonic_dla.models import SimulationDiagnostics, SimulationResult

_SCHEMA_VERSION = 1
_MAX_MEMBER_BYTES = 1 << 30
_MAX_METADATA_BYTES = 16 << 20


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return cast(Any, value).tolist()
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _normalise_path(path: str | Path) -> Path:
    output = Path(path)
    if output.suffix.lower() != ".npz":
        output = output.with_suffix(".npz")
    return output


def _calibration_blocks(
    diagnostics: SimulationDiagnostics,
    particle_count: int,
) -> list[dict[str, Any]]:
    """Build a self-contained, human-auditable calibration history."""
    return [
        {
            "block_index": int(diagnostics.calibration_block_indices[index]),
            "particle_count": int(diagnostics.calibration_sizes[index]),
            "block_stop": int(
                diagnostics.calibration_sizes[index + 1]
                if index + 1 < diagnostics.calibration_sizes.shape[0]
                else particle_count
            ),
            "block_length": int(
                (
                    diagnostics.calibration_sizes[index + 1]
                    if index + 1 < diagnostics.calibration_sizes.shape[0]
                    else particle_count
                )
                - diagnostics.calibration_sizes[index]
            ),
            "center": diagnostics.calibration_centers[index].tolist(),
            "probe_count": int(diagnostics.calibration_probe_counts[index]),
            "search_probes": int(diagnostics.calibration_search_probes[index]),
            "validation_probes": int(diagnostics.calibration_validation_probes[index]),
            "tv_bound": float(diagnostics.calibration_bounds[index]),
            "failure_probability": float(diagnostics.calibration_failure_probabilities[index]),
            "death_ratio": float(diagnostics.calibration_death_ratios[index]),
        }
        for index in range(diagnostics.calibration_centers.shape[0])
    ]


def _member_header(
    archive: np.lib.npyio.NpzFile,
    key: str,
) -> tuple[tuple[int, ...], np.dtype[Any], int, int]:
    """Read one NPY header without allocating the member's data array."""
    try:
        # NpzFile strips the conventional ``.npy`` suffix from ``files``;
        # archives emitted by ``save_result`` always use that member name.
        filename = f"{key}.npy"
        zip_file = archive.zip
        if zip_file is None:
            raise CheckpointError("NPZ archive is closed")
        info = zip_file.getinfo(filename)
        with zip_file.open(info, "r") as stream:
            major, minor = np.lib.format.read_magic(stream)
            if (major, minor) == (1, 0):
                shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
            elif (major, minor) == (2, 0):
                shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise CheckpointError(f"{key} uses unsupported NPY format {major}.{minor}")
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError(f"invalid NPZ member {key}: {exc}") from exc

    if not isinstance(shape, tuple) or any(not isinstance(dim, int) or dim < 0 for dim in shape):
        raise CheckpointError(f"{key} has an invalid logical shape {shape!r}")
    if dtype.hasobject:
        raise CheckpointError(f"{key} uses an object dtype")
    try:
        logical_bytes = int(np.prod(shape, dtype=object)) * int(dtype.itemsize)
    except (OverflowError, TypeError, ValueError) as exc:
        raise CheckpointError(f"{key} has an invalid logical size") from exc
    if logical_bytes < 0 or logical_bytes > _MAX_MEMBER_BYTES:
        raise CheckpointError(
            f"{key} logical size {logical_bytes} exceeds {_MAX_MEMBER_BYTES} bytes"
        )
    # Compressed NPZ members may be much smaller than their logical arrays. The
    # logical-size cap above bounds decompression; the ZIP size cap bounds
    # already-compressed input without assuming an uncompressed payload.
    if int(info.file_size) > _MAX_MEMBER_BYTES:
        raise CheckpointError(
            f"{key} compressed size {int(info.file_size)} exceeds {_MAX_MEMBER_BYTES} bytes"
        )
    return tuple(shape), np.dtype(dtype), logical_bytes, int(info.file_size)


def _validate_archive_layout(archive: np.lib.npyio.NpzFile) -> None:
    """Validate member shapes and logical sizes before materialising arrays."""
    required = {
        "schema_version",
        "positions",
        "particle_radius",
        "seed",
        "backend",
        "restart_mode",
        "growth_walker_steps",
        "growth_restarts",
        "calibration_walker_steps",
        "calibration_restarts",
        "calibration_probes",
        "calibration_centers",
        "calibration_sizes",
        "calibration_bounds",
        "calibration_failure_probabilities",
        "calibration_death_ratios",
        "metadata_json",
    }
    missing = sorted(required.difference(archive.files))
    if missing:
        raise CheckpointError(f"result archive is missing members: {', '.join(missing)}")

    # Additive audit members are optional for schema-1 archives written before
    # calibration provenance was expanded.
    optional = (
        "calibration_block_indices",
        "calibration_probe_counts",
        "calibration_search_probes",
        "calibration_validation_probes",
    )
    headers = {key: _member_header(archive, key) for key in required}
    headers.update({key: _member_header(archive, key) for key in optional if key in archive.files})

    scalar_ints = {
        "schema_version",
        "seed",
        "growth_walker_steps",
        "growth_restarts",
        "calibration_walker_steps",
        "calibration_restarts",
        "calibration_probes",
    }
    for key in scalar_ints:
        shape, dtype, _logical, _file_size = headers[key]
        if shape != () or dtype.kind not in "iu" or dtype.itemsize > 8:
            raise CheckpointError(f"{key} must be an integer scalar")
    for key in ("particle_radius",):
        shape, dtype, _logical, _file_size = headers[key]
        if shape != () or dtype.kind != "f" or dtype.itemsize != 8:
            raise CheckpointError(f"{key} must be a float64 scalar")
    for key in ("backend", "restart_mode", "metadata_json"):
        shape, dtype, logical, _file_size = headers[key]
        if shape != () or dtype.kind not in "SU" or logical > _MAX_METADATA_BYTES:
            raise CheckpointError(f"{key} must be a bounded Unicode scalar")

    positions_shape, positions_dtype, _logical, _file_size = headers["positions"]
    if (
        len(positions_shape) != 2
        or positions_shape[0] < 1
        or positions_shape[1] != 2
        or positions_dtype != np.dtype(np.float64)
    ):
        raise CheckpointError("positions must have logical shape (n, 2) and dtype float64")

    centers_shape, centers_dtype, _logical, _file_size = headers["calibration_centers"]
    if len(centers_shape) != 2 or centers_shape[1] != 2 or centers_dtype != np.dtype(np.float64):
        raise CheckpointError(
            "calibration_centers must have logical shape (m, 2) and dtype float64"
        )
    history_length = centers_shape[0]
    vector_specs: Mapping[str, np.dtype[Any]] = {
        "calibration_sizes": np.dtype(np.int64),
        "calibration_bounds": np.dtype(np.float64),
        "calibration_failure_probabilities": np.dtype(np.float64),
        "calibration_death_ratios": np.dtype(np.float64),
        "calibration_block_indices": np.dtype(np.int64),
        "calibration_probe_counts": np.dtype(np.int64),
        "calibration_search_probes": np.dtype(np.int64),
        "calibration_validation_probes": np.dtype(np.int64),
    }
    for key, expected_dtype in vector_specs.items():
        if key not in headers:
            continue
        shape, dtype, _logical, _file_size = headers[key]
        # Optional audit arrays may be absent, but if present they must align
        # with the calibration center history exactly.
        if shape != (history_length,) or dtype != expected_dtype:
            raise CheckpointError(
                f"{key} must have logical shape ({history_length},) and dtype {expected_dtype}"
            )


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
    metadata.setdefault(
        "calibration_blocks",
        _calibration_blocks(result.diagnostics, result.particle_count),
    )
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
                calibration_block_indices=np.ascontiguousarray(
                    diagnostics.calibration_block_indices,
                    dtype=np.int64,
                ),
                calibration_probe_counts=np.ascontiguousarray(
                    diagnostics.calibration_probe_counts,
                    dtype=np.int64,
                ),
                calibration_search_probes=np.ascontiguousarray(
                    diagnostics.calibration_search_probes,
                    dtype=np.int64,
                ),
                calibration_validation_probes=np.ascontiguousarray(
                    diagnostics.calibration_validation_probes,
                    dtype=np.int64,
                ),
                metadata_json=np.asarray(metadata_text),
            )
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            temporary.replace(output)
        else:
            try:
                # Linking a fully-written temporary file is an atomic
                # create-if-absent operation.  Unlike ``os.replace`` it cannot
                # clobber a destination created by a competing writer.
                os.link(temporary, output)
            except FileExistsError as exc:
                raise CheckpointError(f"refusing to overwrite existing file: {output}") from exc
            finally:
                temporary.unlink(missing_ok=True)
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
            _validate_archive_layout(archive)
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
                calibration_block_indices=np.asarray(
                    archive["calibration_block_indices"],
                    dtype=np.int64,
                ).copy()
                if "calibration_block_indices" in archive.files
                else np.empty(0, dtype=np.int64),
                calibration_probe_counts=np.asarray(
                    archive["calibration_probe_counts"],
                    dtype=np.int64,
                ).copy()
                if "calibration_probe_counts" in archive.files
                else np.empty(0, dtype=np.int64),
                calibration_search_probes=np.asarray(
                    archive["calibration_search_probes"],
                    dtype=np.int64,
                ).copy()
                if "calibration_search_probes" in archive.files
                else np.empty(0, dtype=np.int64),
                calibration_validation_probes=np.asarray(
                    archive["calibration_validation_probes"],
                    dtype=np.int64,
                ).copy()
                if "calibration_validation_probes" in archive.files
                else np.empty(0, dtype=np.int64),
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
