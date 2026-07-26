"""Center estimation and independent validation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from harmonic_dla.certificates import monte_carlo_tv_bound

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """One sample-split calibration outcome."""

    center: tuple[float, float]
    empirical_residual: float
    residual_over_radius: float
    tv_bound: float
    search_probes: int
    validation_probes: int


def empirical_center(attachments: FloatArray) -> tuple[float, float]:
    """Return the barycenter of attachment samples."""
    array = np.asarray(attachments, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < 1:
        raise ValueError("attachments must have shape (n, 2) with n >= 1")
    if not np.all(np.isfinite(array)):
        raise ValueError("attachments must contain only finite values")
    mean = np.mean(array, axis=0)
    return float(mean[0]), float(mean[1])


def validate_center(
    center: tuple[float, float],
    validation_attachments: FloatArray,
    target_radius: float,
    death_ratio: float,
    failure_probability: float,
    *,
    search_probes: int,
) -> CalibrationResult:
    """Validate a center using a probe batch independent of its search batch."""
    if len(center) != 2 or not all(math.isfinite(value) for value in center):
        raise ValueError("center must contain two finite coordinates")
    if not math.isfinite(target_radius) or target_radius <= 0.0:
        raise ValueError("target_radius must be finite and positive")
    if search_probes < 1:
        raise ValueError("search_probes must be positive")
    validation_center = empirical_center(validation_attachments)
    residual = math.hypot(validation_center[0] - center[0], validation_center[1] - center[1])
    normalized = residual / target_radius
    bound = monte_carlo_tv_bound(
        death_ratio,
        normalized,
        int(validation_attachments.shape[0]),
        failure_probability,
    )
    return CalibrationResult(
        center=center,
        empirical_residual=residual,
        residual_over_radius=normalized,
        tv_bound=bound,
        search_probes=search_probes,
        validation_probes=int(validation_attachments.shape[0]),
    )
