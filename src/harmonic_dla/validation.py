"""Deterministic metrics for the preregistered asymmetric-target validation.

The functions in this module intentionally operate only on NumPy arrays and
plain Python values.  Discretized TV values are empirical diagnostics for the
fixed partitions below; they are not continuum total-variation certificates.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple, cast

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

__all__ = [
    "ANGLE_BINS",
    "PRIMARY_RHOS",
    "BootstrapInterval",
    "angle_histogram",
    "assign_nearest_disk",
    "bootstrap_interval",
    "bootstrap_summary",
    "bounded_lipschitz_discrepancy",
    "bounded_lipschitz_features",
    "fit_loglog_slope",
    "fit_power_law",
    "histogram_total_variation",
    "make_lopsided_comb",
    "particle_id_total_variation",
    "schedule_seed",
]

PRIMARY_RHOS: tuple[int, ...] = (3, 4, 6, 8, 12, 16)
ANGLE_BINS: tuple[int, ...] = (16, 32, 64, 128)


class BootstrapInterval(NamedTuple):
    """Central bootstrap interval with tuple and attribute access."""

    lower: float
    upper: float


def _points(value: npt.ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < 1:
        raise ValueError(f"{name} must have shape (n, 2) with n >= 1")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array)


def _center(center: Sequence[float]) -> FloatArray:
    if len(center) != 2 or not all(math.isfinite(float(value)) for value in center):
        raise ValueError("center must contain two finite coordinates")
    return np.asarray(center, dtype=np.float64)


def make_lopsided_comb(
    particle_radius: float = 0.5,
    *,
    spine_length: int = 17,
    tall_branch_height: int = 9,
    short_branch_height: int = 4,
) -> FloatArray:
    """Return a deterministic connected comb of touching equal disks.

    The centers lie on a square lattice with spacing ``2 * particle_radius``.
    A long spine has a tall branch near its right end and a shorter branch near
    its left end, giving a fixed asymmetric target without data-dependent
    choices.
    """
    if not math.isfinite(particle_radius) or particle_radius <= 0.0:
        raise ValueError("particle_radius must be finite and positive")
    for name, value in (
        ("spine_length", spine_length),
        ("tall_branch_height", tall_branch_height),
        ("short_branch_height", short_branch_height),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if spine_length < 3:
        raise ValueError("spine_length must be at least three")

    # Integer lattice coordinates are converted only once at the end, which
    # keeps ordering and duplicate handling deterministic.
    grid: list[tuple[int, int]] = [(index, 0) for index in range(spine_length)]
    tall_x = spine_length - 1 - max(1, spine_length // 5)
    short_x = max(1, spine_length // 5)
    grid.extend((tall_x, height) for height in range(1, tall_branch_height))
    grid.extend((short_x, -height) for height in range(1, short_branch_height))
    return np.asarray(grid, dtype=np.float64) * (2.0 * float(particle_radius))


def assign_nearest_disk(attachments: npt.ArrayLike, disks: npt.ArrayLike) -> IntArray:
    """Assign each attachment to its nearest disk, preferring lowest IDs.

    ``numpy.argmin`` returns the first minimum, so exact distance ties are
    resolved by the lowest row index in ``disks``.
    """
    points = _points(attachments, name="attachments")
    target = _points(disks, name="disks")
    # Bound the temporary distance matrix instead of allocating O(n*m) memory
    # for a large probe batch and target.  ``argmin`` still runs per chunk and
    # therefore preserves its lowest-index tie rule exactly.
    chunk_size = max(1, 1_000_000 // target.shape[0])
    result = np.empty(points.shape[0], dtype=np.int64)
    for start in range(0, points.shape[0], chunk_size):
        stop = min(points.shape[0], start + chunk_size)
        deltas = points[start:stop, None, :] - target[None, :, :]
        distances_squared = np.sum(deltas * deltas, axis=2)
        result[start:stop] = np.argmin(distances_squared, axis=1)
    return result


def _validate_bins(bins: int) -> int:
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)):
        raise ValueError("bins must be an integer")
    value = int(bins)
    if value not in ANGLE_BINS:
        allowed = ", ".join(str(item) for item in ANGLE_BINS)
        raise ValueError(f"bins must be one of {allowed}")
    return value


def angle_histogram(
    attachments: npt.ArrayLike,
    center: Sequence[float] = (0.0, 0.0),
    *,
    bins: int = 16,
) -> FloatArray:
    """Return a normalized polar-angle histogram on a fixed partition."""
    points = _points(attachments, name="attachments")
    origin = _center(center)
    count = np.asarray(
        np.histogram(
            np.mod(np.arctan2(points[:, 1] - origin[1], points[:, 0] - origin[0]), 2.0 * np.pi),
            bins=_validate_bins(bins),
            range=(0.0, 2.0 * np.pi),
        )[0],
        dtype=np.float64,
    )
    return count / float(points.shape[0])


def _histogram_vector(value: npt.ArrayLike, *, bins: int, center: Sequence[float]) -> FloatArray:
    array = np.asarray(value)
    if array.ndim == 1:
        vector = np.asarray(array, dtype=np.float64)
        if vector.shape[0] != bins or not np.all(np.isfinite(vector)) or np.any(vector < 0.0):
            raise ValueError("histogram vectors must be finite and non-negative")
        total = float(np.sum(vector))
        if total <= 0.0:
            raise ValueError("histogram vector must have positive mass")
        return vector / total
    return angle_histogram(array, center, bins=bins)


def histogram_total_variation(
    first: npt.ArrayLike,
    second: npt.ArrayLike,
    *,
    bins: int = 16,
    center: Sequence[float] = (0.0, 0.0),
) -> float:
    """Return empirical TV on the fixed polar-angle partition."""
    bins_value = _validate_bins(bins)
    lhs = _histogram_vector(first, bins=bins_value, center=center)
    rhs = _histogram_vector(second, bins=bins_value, center=center)
    return float(0.5 * np.sum(np.abs(lhs - rhs)))


def _particle_ids(value: Sequence[int] | npt.ArrayLike, *, name: str) -> IntArray:
    array = np.asarray(value)
    if array.ndim != 1 or array.shape[0] < 1:
        raise ValueError(f"{name} IDs must be a non-empty one-dimensional array")
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} IDs must be numeric integers")
    if array.dtype.kind == "f" and (
        not np.all(np.isfinite(array)) or np.any(array != np.floor(array))
    ):
        raise ValueError(f"{name} IDs must be finite integers")
    maximum = np.iinfo(np.int64).max
    if np.any(array < 0) or np.any(array > maximum):
        raise ValueError(f"{name} IDs must fit non-negative int64")
    return np.asarray(array, dtype=np.int64)


def particle_id_total_variation(
    first: Sequence[int] | npt.ArrayLike,
    second: Sequence[int] | npt.ArrayLike,
    *,
    particle_count: int | None = None,
) -> float:
    """Return empirical TV over nearest-target particle IDs."""
    lhs = _particle_ids(first, name="first")
    rhs = _particle_ids(second, name="second")
    if lhs.shape[0] < 1 or rhs.shape[0] < 1:
        raise ValueError("particle IDs must be non-empty one-dimensional arrays")
    if particle_count is None:
        support = int(max(np.max(lhs), np.max(rhs))) + 1
    else:
        if (
            isinstance(particle_count, (bool, np.bool_))
            or not isinstance(particle_count, (int, np.integer))
            or particle_count < 1
        ):
            raise ValueError("particle_count must be a positive integer")
        support = int(particle_count)
        if np.any(lhs >= support) or np.any(rhs >= support):
            raise ValueError("particle ID exceeds particle_count")
    left = np.bincount(lhs, minlength=support).astype(np.float64) / float(lhs.size)
    right = np.bincount(rhs, minlength=support).astype(np.float64) / float(rhs.size)
    return float(0.5 * np.sum(np.abs(left - right)))


def bounded_lipschitz_features(
    points: npt.ArrayLike,
    center: Sequence[float] = (0.0, 0.0),
    *,
    clip_radius: float = 1.0,
) -> FloatArray:
    """Evaluate the fixed clipped-coordinate feature family."""
    values = _points(points, name="points")
    origin = _center(center)
    if not math.isfinite(clip_radius) or clip_radius <= 0.0:
        raise ValueError("clip_radius must be finite and positive")
    scaled = (values - origin) / float(clip_radius)
    directions = np.asarray(
        (
            (1.0, 0.0),
            (math.sqrt(0.5), math.sqrt(0.5)),
            (0.0, 1.0),
            (-math.sqrt(0.5), math.sqrt(0.5)),
        ),
        dtype=np.float64,
    )
    return np.clip(scaled @ directions.T, -1.0, 1.0)


def bounded_lipschitz_discrepancy(
    first: npt.ArrayLike,
    second: npt.ArrayLike,
    center: Sequence[float] = (0.0, 0.0),
    *,
    clip_radius: float = 1.0,
) -> float:
    """Return the maximum mean-feature discrepancy in the fixed family."""
    lhs = bounded_lipschitz_features(first, center, clip_radius=clip_radius)
    rhs = bounded_lipschitz_features(second, center, clip_radius=clip_radius)
    return float(np.max(np.abs(np.mean(lhs, axis=0) - np.mean(rhs, axis=0))))


def bootstrap_interval(
    values: npt.ArrayLike,
    *,
    confidence: float = 0.95,
    resamples: int = 2_000,
    seed: int = 0,
    statistic: Callable[[FloatArray], float] = np.median,
    complete: Sequence[bool] | npt.ArrayLike | None = None,
) -> BootstrapInterval:
    """Return a deterministic central bootstrap interval for a statistic."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.shape[0] < 1 or not np.all(np.isfinite(data)):
        raise ValueError("values must be a non-empty finite vector")
    if complete is not None:
        mask = np.asarray(complete)
        if mask.dtype.kind != "b" or mask.ndim != 1 or mask.shape != data.shape:
            raise ValueError("complete must be a boolean mask aligned with values")
        data = data[mask]
        if data.shape[0] < 1:
            raise ValueError("at least one complete replication is required")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if isinstance(resamples, bool) or not isinstance(resamples, (int, np.integer)) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, data.shape[0], size=(int(resamples), data.shape[0]))
    draws = data[indices]
    estimates = np.asarray([statistic(draw) for draw in draws], dtype=np.float64)
    if estimates.shape != (int(resamples),) or not np.all(np.isfinite(estimates)):
        raise ValueError("statistic must return finite scalar estimates")
    tail = 50.0 * (1.0 - float(confidence))
    return BootstrapInterval(
        float(np.percentile(estimates, tail)),
        float(np.percentile(estimates, 100.0 - tail)),
    )


def bootstrap_summary(
    values: npt.ArrayLike,
    *,
    confidence: float = 0.95,
    resamples: int = 2_000,
    seed: int = 0,
    complete: Sequence[bool] | npt.ArrayLike | None = None,
) -> dict[str, float]:
    """Return median and central bootstrap interval as JSON-ready scalars."""
    data = np.asarray(values, dtype=np.float64)
    low, high = bootstrap_interval(
        data,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
        complete=complete,
    )
    if complete is not None:
        mask = np.asarray(complete)
        if mask.dtype.kind != "b" or mask.ndim != 1 or mask.shape != data.shape or not np.any(mask):
            raise ValueError("complete must be a boolean mask with one complete replication")
        data = data[mask]
    return {"median": float(np.median(data)), "lower": low, "upper": high}


def schedule_seed(parts: Sequence[Any]) -> int:
    """Derive a reproducible non-negative seed from an experiment tuple."""
    if not isinstance(parts, (tuple, list)):
        raise TypeError("seed parts must be a tuple or list of stable values")
    values = tuple(parts)
    if not values:
        raise ValueError("seed tuple must not be empty")
    encoded = _stable_seed_value(values).encode("utf-8")
    digest = hashlib.blake2b(encoded, digest_size=8, person=b"hdla-val").digest()
    return int.from_bytes(digest, byteorder="little") & ((1 << 63) - 1)


def _stable_seed_value(value: Any, active: set[int] | None = None) -> str:
    """Serialize only deterministic primitive values for seed derivation."""
    active = set() if active is None else active
    if value is None:
        return "null"
    if isinstance(value, (bool, np.bool_)):
        return f"bool:{int(bool(value))}"
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return f"int:{int(value)}"
    if isinstance(value, (float, np.floating)) and not isinstance(value, (bool, np.bool_)):
        number = float(value)
        if not math.isfinite(number):
            raise TypeError("seed parts must use finite stable primitive values")
        return f"float:{number.hex()}"
    if isinstance(value, str):
        return f"str:{json.dumps(value, ensure_ascii=True, separators=(',', ':'))}"
    if isinstance(value, bytes):
        return f"bytes:{value.hex()}"
    if isinstance(value, (tuple, list)):
        identity = id(value)
        if identity in active:
            raise TypeError("seed parts must not contain cyclic sequences")
        active.add(identity)
        marker = "tuple" if isinstance(value, tuple) else "list"
        encoded = ",".join(_stable_seed_value(item, active) for item in value)
        active.remove(identity)
        return f"{marker}:[{encoded}]"
    raise TypeError(
        f"seed parts must use stable primitive values (unsupported {type(value).__qualname__})"
    )


def _fit_input(
    ratios: Mapping[float, float] | Sequence[float] | npt.ArrayLike,
    errors: Sequence[float] | npt.ArrayLike | None,
) -> tuple[np.ndarray, np.ndarray]:
    if errors is None:
        if not isinstance(ratios, Mapping):
            raise TypeError("errors are required when ratios are not a mapping")
        data = {float(cast(Any, key)): float(cast(Any, value)) for key, value in ratios.items()}
        x = np.asarray(sorted(data), dtype=np.float64)
        y = np.asarray([data[key] for key in sorted(data)], dtype=np.float64)
    else:
        x = np.asarray(ratios, dtype=np.float64)
        y = np.asarray(errors, dtype=np.float64)
        if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
            raise ValueError("ratios and errors must be one-dimensional vectors of equal length")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.any(y <= 0.0):
        raise ValueError("ratios and errors must be finite, with positive errors")
    return x, y


def _positive_fit_order(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    integer = int(value)
    if integer < 1:
        raise ValueError(f"{name} must be positive")
    return integer


def fit_power_law(
    ratios: Mapping[float, float] | Sequence[float] | npt.ArrayLike,
    errors: Sequence[float] | npt.ArrayLike | None = None,
    *,
    leading_order: int = 1,
    order: int | None = None,
    m: int | None = None,
) -> dict[str, Any]:
    """Fit the preregistered two-term law on the immutable primary window.

    The fitted model is ``a*rho**(-2m) + b*rho**(-2m-2)``.  Extra diagnostic
    ratios are ignored; every primary ratio must be supplied.
    """
    base_order = _positive_fit_order(leading_order, "leading_order")
    aliases = [
        _positive_fit_order(value, name)
        for name, value in (("order", order), ("m", m))
        if value is not None
    ]
    if aliases and any(value != aliases[0] for value in aliases):
        raise ValueError("order aliases disagree")
    if aliases and base_order not in (1, aliases[0]):
        raise ValueError("order aliases disagree with leading_order")
    m = aliases[0] if aliases else base_order
    observed_ratios, observed_errors = _fit_input(ratios, errors)
    if np.unique(observed_ratios).shape[0] != observed_ratios.shape[0]:
        raise ValueError("ratios must be unique")
    by_ratio = {
        float(value): float(error)
        for value, error in zip(observed_ratios, observed_errors, strict=True)
    }
    missing = [rho for rho in PRIMARY_RHOS if rho not in by_ratio]
    if missing:
        raise ValueError(f"missing primary fitting-window ratios: {missing}")
    window = np.asarray(PRIMARY_RHOS, dtype=np.float64)
    values = np.asarray([by_ratio[rho] for rho in PRIMARY_RHOS], dtype=np.float64)
    design = np.column_stack((window ** (-2 * m), window ** (-2 * m - 2)))
    coefficients, _residuals, _rank, _singular = np.linalg.lstsq(design, values, rcond=None)
    predicted = design @ coefficients
    slope, intercept = np.polyfit(np.log(window), np.log(values), 1)
    return {
        "ratios": PRIMARY_RHOS,
        "errors": tuple(float(value) for value in values),
        "leading_order": m,
        "a": float(coefficients[0]),
        "b": float(coefficients[1]),
        "predicted": tuple(float(value) for value in predicted),
        "residual_sum_squares": float(np.sum((predicted - values) ** 2)),
        "slope": float(slope),
        "exponent": float(-slope),
        "intercept": float(intercept),
    }


def fit_loglog_slope(
    ratios: Mapping[float, float] | Sequence[float] | npt.ArrayLike,
    errors: Sequence[float] | npt.ArrayLike | None = None,
) -> dict[str, float]:
    """Return the descriptive log-log fit on the primary fitting window."""
    fit = fit_power_law(ratios, errors)
    return {
        "slope": float(fit["slope"]),
        "exponent": float(fit["exponent"]),
        "intercept": float(fit["intercept"]),
    }
