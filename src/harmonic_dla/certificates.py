"""Error certificates proved in the accompanying paper."""

from __future__ import annotations

import math


def _validate_rho(rho: float) -> None:
    if not math.isfinite(rho) or rho <= 1.0:
        raise ValueError("rho must be finite and greater than one")


def self_centered_tv_bound(rho: float) -> float:
    """Return the geometry-uniform fourth-order one-step TV bound."""
    _validate_rho(rho)
    return min(1.0, 1.0 / (rho * rho * (rho * rho - 1.0)))


def residual_tv_bound(rho: float, residual_over_radius: float) -> float:
    """Bound one-step TV error for an arbitrary trial center."""
    _validate_rho(rho)
    if not math.isfinite(residual_over_radius) or residual_over_radius < 0.0:
        raise ValueError("residual_over_radius must be finite and non-negative")
    value = residual_over_radius / (rho * rho) + self_centered_tv_bound(rho)
    return min(1.0, value)


def monte_carlo_tv_bound(
    rho: float,
    empirical_residual_over_radius: float,
    probes: int,
    failure_probability: float,
) -> float:
    """Return the distribution-free Monte Carlo certificate."""
    if probes < 1:
        raise ValueError("probes must be positive")
    if not 0.0 < failure_probability < 1.0:
        raise ValueError("failure_probability must lie in (0, 1)")
    sampling = 2.0 * math.sqrt(math.log(4.0 / failure_probability) / probes)
    return residual_tv_bound(rho, empirical_residual_over_radius + sampling)


def path_budget(step_bounds: list[float] | tuple[float, ...]) -> float:
    """Combine conditional one-step TV bounds using the product coupling bound."""
    product = 1.0
    for bound in step_bounds:
        if not 0.0 <= bound <= 1.0:
            raise ValueError("every step bound must lie in [0, 1]")
        product *= 1.0 - bound
    return 1.0 - product


def constant_ratio_for_path_budget(steps: int, budget: float) -> float:
    """Smallest constant ratio certified by the self-centered product bound."""
    if steps < 1:
        raise ValueError("steps must be positive")
    if not 0.0 < budget < 1.0:
        raise ValueError("budget must lie in (0, 1)")
    per_step = 1.0 - (1.0 - budget) ** (1.0 / steps)
    x = (1.0 + math.sqrt(1.0 + 4.0 / per_step)) / 2.0
    return math.sqrt(x)


def one_shot_diameter_tv_bound(
    diameter_ratio: float,
    probes: int,
    failure_probability: float,
) -> float:
    """Paper bound after one empirical barycenter update on a diameter-scaled disk."""
    _validate_rho(diameter_ratio)
    if probes < 1:
        raise ValueError("probes must be positive")
    if not 0.0 < failure_probability < 1.0:
        raise ValueError("failure_probability must lie in (0, 1)")
    rho_sq = diameter_ratio * diameter_ratio
    sampling = 2.0 / rho_sq * math.sqrt(math.log(4.0 / failure_probability) / probes)
    remainder = 5.0 / (rho_sq * (rho_sq - 1.0))
    return min(1.0, sampling + remainder)


def _p_series_tail_upper(exponent: float, start: int) -> float:
    if exponent <= 1.0:
        raise ValueError("exponent must exceed one")
    if start < 1:
        raise ValueError("start must be positive")
    return start ** (-exponent) + start ** (1.0 - exponent) / (exponent - 1.0)


def amortized_path_budget_upper(
    *,
    alpha: float,
    gamma: float,
    scale: float,
    start: int,
    calibration_failure_budget: float,
    drift_constant: float = 3.0 * math.sqrt(2.0),
) -> float:
    """Conservative infinite-history bound using integral p-series tail estimates.

    The default drift constant is the paper's equal-disk value for capture targets built from
    physical particles of radius ``a0`` and one-step parallel enlargement ``delta0 = 2*a0``.
    """
    if not all(
        math.isfinite(value)
        for value in (alpha, gamma, scale, calibration_failure_budget, drift_constant)
    ):
        raise ValueError("all amortized-bound parameters must be finite")
    if 2.0 * alpha + gamma <= 1.0 or 4.0 * alpha <= 1.0:
        raise ValueError("the path-error exponents must be summable")
    if scale < math.sqrt(2.0):
        raise ValueError("scale must be at least sqrt(2)")
    if start < 1:
        raise ValueError("start must be positive")
    if not 0.0 <= calibration_failure_budget < 1.0:
        raise ValueError("calibration_failure_budget must lie in [0, 1)")
    if drift_constant < 0.0:
        raise ValueError("drift_constant cannot be negative")
    c1 = 2.0**gamma * (1.0 + drift_constant)
    c2 = 2.0 ** (2.0 * alpha + 2.0) + 6.0
    value = (
        calibration_failure_budget
        + c1 * scale**-2.0 * _p_series_tail_upper(2.0 * alpha + gamma, start)
        + c2 * scale**-4.0 * _p_series_tail_upper(4.0 * alpha, start)
    )
    return min(1.0, value)
