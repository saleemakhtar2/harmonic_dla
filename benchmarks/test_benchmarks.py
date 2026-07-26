from __future__ import annotations

import pytest

from harmonic_dla.api import probe, simulate
from harmonic_dla.config import BoundaryConfig, RunConfig
from harmonic_dla.enums import RestartMode


def _warm_numba() -> None:
    simulate(RunConfig(particles=8, seed=123))


@pytest.mark.benchmark
def test_numba_exact_aggregate_500(benchmark) -> None:
    _warm_numba()
    config = RunConfig(particles=500, seed=123)
    result = benchmark(simulate, config)
    assert result.particle_count == 500


@pytest.mark.benchmark
def test_numba_uniform_aggregate_500(benchmark) -> None:
    _warm_numba()
    config = RunConfig(
        particles=500,
        seed=123,
        boundary=BoundaryConfig(
            restart_mode=RestartMode.UNIFORM_RESTART,
            death_ratio=4.0,
            launch_margin=4.0,
        ),
    )
    result = benchmark(simulate, config)
    assert result.particle_count == 500


@pytest.mark.benchmark
def test_numba_frozen_probe_batch_2000(benchmark) -> None:
    _warm_numba()
    cluster = simulate(RunConfig(particles=200, seed=321))
    attachments = benchmark(
        probe,
        cluster.positions,
        particle_radius=cluster.particle_radius,
        center=(0.0, 0.0),
        death_ratio=4.0,
        launch_margin=4.0,
        probes=2_000,
        seed=999,
    )
    assert attachments.shape == (2_000, 2)
