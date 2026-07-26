from __future__ import annotations

import numpy as np

from harmonic_dla.analysis import minimum_pair_distance
from harmonic_dla.api import simulate
from harmonic_dla.config import BoundaryConfig, CalibrationConfig, RunConfig
from harmonic_dla.enums import BackendKind, RestartMode


def test_numba_exact_smoke_and_reproducibility() -> None:
    config = RunConfig(particles=40, seed=11, backend=BackendKind.NUMBA_CPU)
    first = simulate(config)
    second = simulate(config)
    np.testing.assert_array_equal(first.positions, second.positions)
    assert minimum_pair_distance(first.positions) >= 1.0 - 1.0e-12


def test_reference_smoke() -> None:
    result = simulate(RunConfig(particles=10, seed=5, backend=BackendKind.REFERENCE))
    assert result.particle_count == 10
    assert minimum_pair_distance(result.positions) >= 1.0 - 1.0e-12


def test_sample_split_controlled_smoke() -> None:
    config = RunConfig(
        particles=16,
        seed=9,
        boundary=BoundaryConfig(
            restart_mode=RestartMode.CONTROLLED_RESTART,
            death_ratio=3.0,
            launch_margin=3.0,
        ),
        calibration=CalibrationConfig(
            enabled=True,
            search_probes=8,
            validation_probes=8,
            block_size=5,
            confidence_failure=0.1,
        ),
    )
    result = simulate(config)
    assert result.diagnostics.calibration_probes > 0
    assert result.diagnostics.calibration_bounds.size > 0
    assert np.all(
        (0.0 <= result.diagnostics.calibration_bounds)
        & (result.diagnostics.calibration_bounds <= 1.0)
    )


def test_public_probe_rejects_invalid_geometry() -> None:
    import pytest

    from harmonic_dla.api import probe

    with pytest.raises(ValueError, match="shape"):
        probe(
            np.asarray([0.0, 0.0]),
            particle_radius=0.5,
            probes=4,
        )
