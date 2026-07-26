from __future__ import annotations

from pathlib import Path

import numpy as np

from harmonic_dla.io import load_result
from harmonic_dla.models import SimulationDiagnostics, SimulationResult


def test_result_round_trip_without_pickle(tmp_path: Path) -> None:
    result = SimulationResult(
        positions=np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        particle_radius=0.5,
        seed=3,
        backend="reference",
        restart_mode="exact_return",
        diagnostics=SimulationDiagnostics(
            growth_walker_steps=10,
            calibration_centers=np.asarray([[0.1, 0.2]], dtype=np.float64),
            calibration_sizes=np.asarray([2], dtype=np.int64),
            calibration_bounds=np.asarray([0.01], dtype=np.float64),
            calibration_failure_probabilities=np.asarray([0.001], dtype=np.float64),
            calibration_death_ratios=np.asarray([4.0], dtype=np.float64),
        ),
        metadata={"complete": True, "final_center": [0.0, 0.0]},
    )
    path = result.save(tmp_path / "result", overwrite=False)
    loaded = load_result(path)
    np.testing.assert_allclose(loaded.positions, result.positions)
    assert loaded.diagnostics.growth_walker_steps == 10
    assert loaded.metadata["complete"] is True


def test_diagnostics_reject_mismatched_history_lengths() -> None:
    import pytest

    with pytest.raises(ValueError, match="calibration_sizes"):
        SimulationDiagnostics(
            calibration_centers=np.asarray([[0.0, 0.0]], dtype=np.float64),
            calibration_sizes=np.empty(0, dtype=np.int64),
            calibration_bounds=np.asarray([0.1], dtype=np.float64),
            calibration_failure_probabilities=np.asarray([0.01], dtype=np.float64),
            calibration_death_ratios=np.asarray([2.0], dtype=np.float64),
        )
