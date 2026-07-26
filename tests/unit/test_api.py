from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from harmonic_dla.api import create_backend, probe, resolve_config
from harmonic_dla.config import RunConfig
from harmonic_dla.enums import BackendKind


def test_create_backend_and_resolve_object() -> None:
    assert create_backend(BackendKind.REFERENCE).name == "reference"
    assert create_backend("numba-cpu").name == "numba-cpu"
    config = RunConfig(particles=2)
    assert resolve_config(config) is config


def test_resolve_config_path(tmp_path: Path) -> None:
    path = tmp_path / "run.toml"
    path.write_text('particles = 2\nbackend = "reference"\n')
    config = resolve_config(path)
    assert config.particles == 2
    assert config.backend is BackendKind.REFERENCE


def test_reference_probe_is_reproducible() -> None:
    positions = np.asarray([[0.0, 0.0]], dtype=np.float64)
    first = probe(
        positions,
        particle_radius=0.5,
        death_ratio=2.0,
        launch_margin=1.0,
        probes=4,
        seed=17,
        backend=BackendKind.REFERENCE,
    )
    second = probe(
        positions,
        particle_radius=0.5,
        death_ratio=2.0,
        launch_margin=1.0,
        probes=4,
        seed=17,
        backend=BackendKind.REFERENCE,
    )
    assert first.shape == (4, 2)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("particle_radius", 0.0, "particle_radius"),
        ("death_ratio", 1.0, "death_ratio"),
        ("death_ratio", math.inf, "death_ratio"),
        ("launch_margin", 0.0, "launch_margin"),
        ("probes", 0, "probes"),
        ("seed", -1, "seed"),
        ("seed", 2**63, "seed"),
    ],
)
def test_probe_rejects_invalid_scalar_parameters(
    keyword: str,
    value: float | int,
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "particle_radius": 0.5,
        "death_ratio": 2.0,
        "launch_margin": 1.0,
        "probes": 4,
        "seed": 0,
        "backend": BackendKind.REFERENCE,
    }
    arguments[keyword] = value
    with pytest.raises(ValueError, match=message):
        probe(np.asarray([[0.0, 0.0]], dtype=np.float64), **arguments)  # type: ignore[arg-type]


def test_probe_rejects_invalid_center_and_nonfinite_positions() -> None:
    with pytest.raises(ValueError, match="center"):
        probe(
            np.asarray([[0.0, 0.0]], dtype=np.float64),
            particle_radius=0.5,
            center=(math.nan, 0.0),
            backend=BackendKind.REFERENCE,
        )
    with pytest.raises(ValueError, match="finite"):
        probe(
            np.asarray([[math.inf, 0.0]], dtype=np.float64),
            particle_radius=0.5,
            backend=BackendKind.REFERENCE,
        )
