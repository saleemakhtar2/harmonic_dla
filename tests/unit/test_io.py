from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from harmonic_dla.exceptions import CheckpointError
from harmonic_dla.io import load_result, save_result
from harmonic_dla.models import SimulationDiagnostics, SimulationResult
from harmonic_dla.provenance import runtime_provenance


def test_result_round_trip_without_pickle(tmp_path: Path) -> None:
    result = SimulationResult(
        positions=np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        particle_radius=0.5,
        seed=3,
        backend="reference",
        restart_mode="exact_return",
        diagnostics=SimulationDiagnostics(
            growth_walker_steps=10,
            calibration_probes=12,
            calibration_centers=np.asarray([[0.1, 0.2]], dtype=np.float64),
            calibration_sizes=np.asarray([2], dtype=np.int64),
            calibration_bounds=np.asarray([0.01], dtype=np.float64),
            calibration_failure_probabilities=np.asarray([0.001], dtype=np.float64),
            calibration_death_ratios=np.asarray([4.0], dtype=np.float64),
            calibration_block_indices=np.asarray([7], dtype=np.int64),
            calibration_probe_counts=np.asarray([12], dtype=np.int64),
            calibration_search_probes=np.asarray([8], dtype=np.int64),
            calibration_validation_probes=np.asarray([4], dtype=np.int64),
        ),
        metadata={"complete": True, "final_center": [0.0, 0.0]},
    )
    path = result.save(tmp_path / "result", overwrite=False)
    loaded = load_result(path)
    np.testing.assert_allclose(loaded.positions, result.positions)
    assert loaded.diagnostics.growth_walker_steps == 10
    assert loaded.metadata["complete"] is True
    assert loaded.metadata["calibration_blocks"] == [
        {
            "block_index": 7,
            "block_length": 0,
            "block_stop": 2,
            "center": [0.1, 0.2],
            "death_ratio": 4.0,
            "failure_probability": 0.001,
            "particle_count": 2,
            "probe_count": 12,
            "search_probes": 8,
            "tv_bound": 0.01,
            "validation_probes": 4,
        }
    ]


def test_diagnostics_reject_mismatched_history_lengths() -> None:
    with pytest.raises(ValueError, match="calibration_sizes"):
        SimulationDiagnostics(
            calibration_centers=np.asarray([[0.0, 0.0]], dtype=np.float64),
            calibration_sizes=np.empty(0, dtype=np.int64),
            calibration_bounds=np.asarray([0.1], dtype=np.float64),
            calibration_failure_probabilities=np.asarray([0.01], dtype=np.float64),
            calibration_death_ratios=np.asarray([2.0], dtype=np.float64),
        )


def test_result_rejects_unpersistable_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        SimulationResult(
            positions=np.asarray([[0.0, 0.0]], dtype=np.float64),
            particle_radius=0.5,
            seed=2**63,
            backend="reference",
            restart_mode="exact_return",
            diagnostics=SimulationDiagnostics(),
        )


def _minimal_result(*, complete: bool = True) -> SimulationResult:
    return SimulationResult(
        positions=np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        particle_radius=0.5,
        seed=3,
        backend="reference",
        restart_mode="exact_return",
        diagnostics=SimulationDiagnostics(),
        metadata={"complete": complete, "final_center": [2.0, 0.0]},
    )


def test_save_result_no_clobber_is_atomic_when_destination_appears(
    monkeypatch, tmp_path: Path
) -> None:
    """A destination created after the initial check must not be replaced."""
    import os

    output = tmp_path / "race.npz"
    original_link = os.link

    def race_link(
        source: str | bytes | os.PathLike[str],
        destination: str | bytes | os.PathLike[str],
        *args,
        **kwargs,
    ):
        output.write_bytes(b"winner")
        return original_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "link", race_link)
    with pytest.raises(CheckpointError, match="refusing to overwrite"):
        save_result(_minimal_result(), output, overwrite=False)
    assert output.read_bytes() == b"winner"


def test_load_result_rejects_oversized_logical_member_before_materialization(
    tmp_path: Path,
) -> None:
    result_path = _minimal_result().save(tmp_path / "valid.npz")
    malformed = tmp_path / "malformed.npz"
    with zipfile.ZipFile(result_path, "r") as source, zipfile.ZipFile(malformed, "w") as target:
        for info in source.infolist():
            payload = source.read(info.filename)
            if info.filename == "positions.npy":
                # A valid NPY header claiming an impossible allocation, with no payload.
                header = "{'descr': '<f8', 'fortran_order': False, 'shape': (1099511627776, 2), }\n"
                encoded = header.encode("latin1")
                padding = (-((10 + len(encoded)) % 16)) % 16
                encoded = header[:-1].encode("latin1") + (" " * padding).encode("ascii") + b"\n"
                payload = b"\x93NUMPY\x01\x00" + len(encoded).to_bytes(2, "little") + encoded
            target.writestr(info, payload)
    with pytest.raises(CheckpointError, match="positions"):
        load_result(malformed)


def test_runtime_provenance_uses_source_revision_environment(monkeypatch) -> None:
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.setenv("HDLA_SOURCE_REVISION", "deadbeef")
    assert runtime_provenance()["source_revision"] == "deadbeef"
