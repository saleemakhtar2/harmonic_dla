from __future__ import annotations

import numpy as np
from numba import get_num_threads, set_num_threads

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
    assert np.all(result.diagnostics.calibration_bounds >= 0.0)
    assert np.all(result.diagnostics.calibration_bounds <= 1.0)
    np.testing.assert_array_equal(result.diagnostics.calibration_sizes, np.arange(1, 16))


def test_reference_sample_split_recalibrates_each_attachment() -> None:
    config = RunConfig(
        particles=8,
        seed=9,
        backend=BackendKind.REFERENCE,
        boundary=BoundaryConfig(restart_mode=RestartMode.CONTROLLED_RESTART),
        calibration=CalibrationConfig(
            enabled=True,
            search_probes=4,
            validation_probes=4,
            block_size=4,
            confidence_failure=0.1,
        ),
    )
    result = simulate(config)
    np.testing.assert_array_equal(result.diagnostics.calibration_sizes, np.arange(1, 8))


def test_numba_probe_is_translation_invariant() -> None:
    from harmonic_dla.api import probe

    positions = np.asarray([[100.0, -75.0]], dtype=np.float64)
    result = probe(
        positions,
        particle_radius=0.5,
        center=(100.0, -75.0),
        probes=4,
        seed=3,
        backend=BackendKind.NUMBA_CPU,
    )
    assert result.shape == (4, 2)


def test_numba_simulation_restores_thread_state() -> None:
    from harmonic_dla.config import PerformanceConfig

    previous = get_num_threads()
    try:
        set_num_threads(min(2, previous))
        before = get_num_threads()
        simulate(
            RunConfig(
                particles=4,
                seed=13,
                performance=PerformanceConfig(threads=1),
            )
        )
        assert get_num_threads() == before
    finally:
        set_num_threads(previous)


def test_numba_checkpoint_respects_existing_artifact(tmp_path) -> None:
    from harmonic_dla.backends.numba_cpu.backend import _checkpoint_path
    from harmonic_dla.config import OutputConfig

    output = tmp_path / "cluster.npz"
    checkpoint = _checkpoint_path(output)
    checkpoint.write_bytes(b"keep me")
    config = RunConfig(
        particles=6,
        seed=2,
        output=OutputConfig(path=output, checkpoint_every=2, overwrite=False),
    )
    import pytest

    with pytest.raises(Exception, match=r"checkpoint|overwrite|existing"):
        simulate(config)
    assert checkpoint.read_bytes() == b"keep me"


def test_public_probe_rejects_invalid_geometry() -> None:
    import pytest

    from harmonic_dla.api import probe

    with pytest.raises(ValueError, match="shape"):
        probe(
            np.asarray([0.0, 0.0]),
            particle_radius=0.5,
            probes=4,
        )
