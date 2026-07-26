"""Boundary-return mathematics shared by reference and compiled backends."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def poisson_return_delta(q: float, uniforms: FloatArray) -> FloatArray:
    """Sample angular offsets from the exterior Poisson kernel."""
    if not 0.0 <= q < 1.0:
        raise ValueError("q must lie in [0, 1)")
    u = np.asarray(uniforms, dtype=np.float64)
    if not np.all(np.isfinite(u)):
        raise ValueError("uniform samples must be finite")
    if np.any((u <= 0.0) | (u >= 1.0)):
        raise ValueError("uniform samples must lie strictly inside (0, 1)")
    phi = np.pi * (u - 0.5)
    scale = (1.0 - q) / (1.0 + q)
    return 2.0 * np.arctan2(scale * np.sin(phi), np.cos(phi))


def launch_point(center: tuple[float, float], radius: float, angle: float) -> tuple[float, float]:
    """Convert polar launch coordinates to Cartesian coordinates."""
    if len(center) != 2 or not all(math.isfinite(value) for value in center):
        raise ValueError("center must contain two finite coordinates")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    if not math.isfinite(angle):
        raise ValueError("angle must be finite")
    return (
        center[0] + radius * math.cos(angle),
        center[1] + radius * math.sin(angle),
    )
