"""Small geometry and integrity checks for generated aggregates."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _positions_array(positions: FloatArray) -> FloatArray:
    array = np.asarray(positions, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < 1:
        raise ValueError("positions must have shape (n, 2) with n >= 1")
    if not np.all(np.isfinite(array)):
        raise ValueError("positions must contain only finite values")
    return array


def radius_about(positions: FloatArray, center: tuple[float, float]) -> float:
    """Maximum particle-center radius about ``center``."""
    array = _positions_array(positions)
    if len(center) != 2 or not all(math.isfinite(value) for value in center):
        raise ValueError("center must contain two finite coordinates")
    offsets = array - np.asarray(center, dtype=np.float64)
    return float(np.sqrt(np.max(np.sum(offsets * offsets, axis=1))))


def bounding_box_diameter_upper(positions: FloatArray, particle_radius: float) -> float:
    """Return a conservative diameter bound for the capture target."""
    array = _positions_array(positions)
    if not math.isfinite(particle_radius) or particle_radius <= 0.0:
        raise ValueError("particle_radius must be finite and positive")
    span = np.max(array, axis=0) - np.min(array, axis=0) + 4.0 * particle_radius
    return float(math.hypot(float(span[0]), float(span[1])))


def minimum_pair_distance(positions: FloatArray) -> float:
    """Return the exact minimum center distance; intended for tests and small runs."""
    array = _positions_array(positions)
    best = math.inf
    for index in range(array.shape[0] - 1):
        delta = array[index + 1 :] - array[index]
        distances = np.sqrt(np.sum(delta * delta, axis=1))
        candidate = float(np.min(distances))
        if candidate < best:
            best = candidate
    return best
