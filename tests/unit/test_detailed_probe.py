from __future__ import annotations

import numpy as np
import pytest

from harmonic_dla.api import probe, probe_detailed
from harmonic_dla.enums import BackendKind, RestartMode
from harmonic_dla.models import ProbeResult

POSITIONS = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
ASYMMETRIC_POSITIONS = np.asarray(
    [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0]], dtype=np.float64
)


@pytest.mark.parametrize("restart_mode", [RestartMode.UNIFORM_RESTART, RestartMode.EXACT_RETURN])
@pytest.mark.parametrize("backend", [BackendKind.REFERENCE, BackendKind.NUMBA_CPU])
def test_probe_detailed_returns_validated_result(
    restart_mode: RestartMode,
    backend: BackendKind,
) -> None:
    result = probe_detailed(
        POSITIONS,
        particle_radius=0.5,
        death_ratio=2.0,
        launch_margin=1.0,
        probes=4,
        seed=17,
        backend=backend,
        restart_mode=restart_mode,
    )
    assert isinstance(result, ProbeResult)
    assert result.attachments.shape == (4, 2)
    assert result.attachments.dtype == np.float64
    assert np.all(np.isfinite(result.attachments))
    assert result.walker_steps >= 0
    assert result.restarts >= 0


def test_legacy_probe_still_returns_attachment_array() -> None:
    result = probe(
        POSITIONS,
        particle_radius=0.5,
        death_ratio=2.0,
        launch_margin=1.0,
        probes=4,
        seed=17,
        backend=BackendKind.REFERENCE,
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == (4, 2)


def test_probe_detailed_default_matches_uniform_legacy_shape() -> None:
    result = probe_detailed(
        POSITIONS,
        particle_radius=0.5,
        death_ratio=2.0,
        launch_margin=1.0,
        probes=4,
        seed=17,
        backend=BackendKind.REFERENCE,
    )
    legacy = probe(
        POSITIONS,
        particle_radius=0.5,
        death_ratio=2.0,
        launch_margin=1.0,
        probes=4,
        seed=17,
        backend=BackendKind.REFERENCE,
    )
    np.testing.assert_array_equal(result.attachments, legacy)


@pytest.mark.parametrize(
    "restart_mode",
    ["uniform", 1, None, RestartMode.CONTROLLED_RESTART],
)
def test_probe_detailed_rejects_invalid_restart_mode(restart_mode: object) -> None:
    with pytest.raises(ValueError, match="restart_mode"):
        probe_detailed(
            POSITIONS,
            particle_radius=0.5,
            backend=BackendKind.REFERENCE,
            restart_mode=restart_mode,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("backend", [BackendKind.REFERENCE, BackendKind.NUMBA_CPU])
def test_exact_and_uniform_dispatch_to_distinct_return_laws(backend: BackendKind) -> None:
    uniform = probe_detailed(
        ASYMMETRIC_POSITIONS,
        particle_radius=0.5,
        center=(0.0, 0.0),
        death_ratio=2.0,
        launch_margin=1.0,
        probes=16,
        seed=17,
        backend=backend,
        restart_mode=RestartMode.UNIFORM_RESTART,
    )
    exact = probe_detailed(
        ASYMMETRIC_POSITIONS,
        particle_radius=0.5,
        center=(0.0, 0.0),
        death_ratio=2.0,
        launch_margin=1.0,
        probes=16,
        seed=17,
        backend=backend,
        restart_mode=RestartMode.EXACT_RETURN,
    )
    assert not np.array_equal(uniform.attachments, exact.attachments)
