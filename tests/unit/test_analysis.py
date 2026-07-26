from __future__ import annotations

import math

import numpy as np
import pytest

from harmonic_dla.analysis import (
    bounding_box_diameter_upper,
    minimum_pair_distance,
    radius_about,
)


def test_geometry_summaries() -> None:
    positions = np.asarray([[0.0, 0.0], [3.0, 4.0]], dtype=np.float64)
    assert radius_about(positions, (0.0, 0.0)) == pytest.approx(5.0)
    assert bounding_box_diameter_upper(positions, 0.5) == pytest.approx(math.sqrt(61.0))
    assert minimum_pair_distance(positions) == pytest.approx(5.0)


def test_single_particle_minimum_distance_is_infinite() -> None:
    positions = np.asarray([[1.0, -2.0]], dtype=np.float64)
    assert math.isinf(minimum_pair_distance(positions))


@pytest.mark.parametrize(
    "positions",
    [
        np.asarray([0.0, 0.0]),
        np.empty((0, 2), dtype=np.float64),
        np.asarray([[math.nan, 0.0]], dtype=np.float64),
    ],
)
def test_geometry_rejects_invalid_positions(positions: np.ndarray) -> None:
    with pytest.raises(ValueError):
        radius_about(positions, (0.0, 0.0))


@pytest.mark.parametrize("center", [(math.inf, 0.0), (0.0,), (0.0, 1.0, 2.0)])
def test_radius_rejects_invalid_center(center: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="center"):
        radius_about(np.asarray([[0.0, 0.0]], dtype=np.float64), center)  # type: ignore[arg-type]


@pytest.mark.parametrize("particle_radius", [0.0, -1.0, math.inf, math.nan])
def test_diameter_rejects_invalid_particle_radius(particle_radius: float) -> None:
    with pytest.raises(ValueError, match="particle_radius"):
        bounding_box_diameter_upper(
            np.asarray([[0.0, 0.0]], dtype=np.float64),
            particle_radius,
        )
