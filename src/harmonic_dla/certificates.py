"""Error certificates proved in the accompanying paper."""

from __future__ import annotations

import math
from numbers import Integral, Real


def _validate_rho(rho: float) -> None:
    if not isinstance(rho, Real) or isinstance(rho, bool) or not math.isfinite(rho) or rho <= 1.0:
        raise ValueError("rho must be finite and greater than one")


def _validate_probes(probes: int) -> None:
    if not isinstance(probes, Integral) or isinstance(probes, bool) or probes < 1:
        raise ValueError("probes must be a positive integer")


def _validate_probability(value: float, name: str = "failure_probability") -> None:
    if not isinstance(value, Real) or isinstance(value, bool) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie in (0, 1)")


def self_centered_tv_bound(rho: float) -> float:
    """Return the geometry-uniform fourth-order one-step TV bound."""
    _validate_rho(rho)
    return min(1.0, 1.0 / (rho * rho * (rho * rho - 1.0)))


def residual_tv_bound(rho: float, residual_over_radius: float) -> float:
    """Bound one-step TV error for an arbitrary trial center."""
    _validate_rho(rho)
    if (
        not isinstance(residual_over_radius, Real)
        or isinstance(residual_over_radius, bool)
        or not math.isfinite(residual_over_radius)
        or residual_over_radius < 0.0
    ):
        raise ValueError("residual_over_radius must be finite and non-negative")
    rho_value = float(rho)
    residual_value = float(residual_over_radius)
    value = residual_value / (rho_value * rho_value) + self_centered_tv_bound(rho_value)
    return min(1.0, value)


def monte_carlo_tv_bound(
    rho: float,
    empirical_residual_over_radius: float,
    probes: int,
    failure_probability: float,
) -> float:
    """Return the distribution-free Monte Carlo certificate."""
    _validate_probes(probes)
    _validate_probability(failure_probability)
    if (
        not isinstance(empirical_residual_over_radius, Real)
        or isinstance(empirical_residual_over_radius, bool)
        or not math.isfinite(empirical_residual_over_radius)
        or empirical_residual_over_radius < 0.0
    ):
        raise ValueError("empirical_residual_over_radius must be finite and non-negative")
    sampling = 2.0 * math.sqrt(
        (math.log(4.0) - math.log(float(failure_probability))) / float(probes)
    )
    empirical_value = float(empirical_residual_over_radius)
    return residual_tv_bound(float(rho), empirical_value + sampling)


def path_budget(step_bounds: list[float] | tuple[float, ...]) -> float:
    """Combine conditional one-step TV bounds using the product coupling bound."""
    log_survival = 0.0
    for bound in step_bounds:
        if not 0.0 <= bound <= 1.0:
            raise ValueError("every step bound must lie in [0, 1]")
        if bound == 1.0:
            return 1.0
        # Summing log1p terms avoids rounding each survival factor to one.
        log_survival += math.log1p(-bound)
    return -math.expm1(log_survival)


def constant_ratio_for_path_budget(steps: int, budget: float) -> float:
    """Smallest constant ratio certified by the self-centered product bound."""
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be positive")
    if not math.isfinite(budget) or not 0.0 < budget < 1.0:
        raise ValueError("budget must lie in (0, 1)")
    # Equivalent to 1 - (1 - budget)**(1/steps), without cancellation for tiny budgets.
    per_step = -math.expm1(math.log1p(-budget) / steps)
    x = (1.0 + math.sqrt(1.0 + 4.0 / per_step)) / 2.0
    return math.sqrt(x)


def one_shot_diameter_tv_bound(
    diameter_ratio: float,
    probes: int,
    failure_probability: float,
) -> float:
    """Paper bound after one empirical barycenter update on a diameter-scaled disk."""
    _validate_rho(diameter_ratio)
    _validate_probes(probes)
    _validate_probability(failure_probability)
    rho_sq = diameter_ratio * diameter_ratio
    sampling = 2.0 / rho_sq * math.sqrt((math.log(4.0) - math.log(failure_probability)) / probes)
    remainder = 5.0 / (rho_sq * (rho_sq - 1.0))
    return min(1.0, sampling + remainder)


def amortized_local_tv_bound(
    n: int,
    *,
    alpha: float,
    gamma: float,
    scale: float,
    start: int | None = None,
    drift_constant: float = 3.0 * math.sqrt(2.0),
) -> float:
    """The theorem's conditional one-step bound at particle count ``n``.

    This is the ``epsilon_n`` term in the blockwise path theorem, with the
    equal-disk drift constant exposed for geometry-specific refinements.
    """
    if not isinstance(n, Integral) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    if not all(
        isinstance(value, Real) and not isinstance(value, bool)
        for value in (alpha, gamma, scale, drift_constant)
    ) or not all(math.isfinite(value) for value in (alpha, gamma, scale, drift_constant)):
        raise ValueError("all amortized-bound parameters must be finite real numbers")
    if alpha <= 11.0 / 24.0:
        raise ValueError("alpha must exceed 11/24 for the area-only theorem")
    lower = max(0.0, 1.0 - 2.0 * alpha)
    if not lower < gamma < 1.0 / 12.0:
        raise ValueError("gamma must lie in (max(0, 1-2alpha), 1/12)")
    if scale < math.sqrt(2.0):
        raise ValueError("scale must be at least sqrt(2)")
    if start is not None:
        if not isinstance(start, Integral) or isinstance(start, bool):
            raise ValueError("start must be an integer")
        kappa = 0.25 - gamma
        minimum_start = max(16, math.ceil(2.0 ** (1.0 / kappa)))
        if start < minimum_start:
            raise ValueError(
                f"start must be at least {minimum_start} for the equal-disk theorem constants"
            )
        if n < start:
            raise ValueError("n must be at least start for the amortized theorem")
    if drift_constant < 0.0:
        raise ValueError("drift_constant cannot be negative")
    exponent_one = 2.0 * alpha + gamma
    exponent_two = 4.0 * alpha
    if not math.isfinite(exponent_one) or not math.isfinite(exponent_two):
        raise ValueError("the path-error exponents must be finite")
    c1 = 2.0**gamma * (1.0 + drift_constant)
    c2 = 2.0 ** (2.0 * alpha + 2.0) + 6.0
    return min(
        1.0,
        c1 * math.pow(scale, -2.0) * math.pow(float(n), -exponent_one)
        + c2 * math.pow(scale, -4.0) * math.pow(float(n), -exponent_two),
    )


def _p_series_tail_upper(exponent: float, start: int) -> float:
    if (
        not isinstance(exponent, Real)
        or isinstance(exponent, bool)
        or not math.isfinite(exponent)
        or exponent <= 1.0
    ):
        raise ValueError("exponent must exceed one")
    if not isinstance(start, Integral) or isinstance(start, bool) or start < 1:
        raise ValueError("start must be a positive integer")
    start_float = float(start)
    exponent_float = float(exponent)
    return math.pow(start_float, -exponent_float) + math.pow(start_float, 1.0 - exponent_float) / (
        exponent_float - 1.0
    )


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
        isinstance(value, Real) and not isinstance(value, bool)
        for value in (alpha, gamma, scale, calibration_failure_budget, drift_constant)
    ):
        raise ValueError("all amortized-bound parameters must be real numbers")
    if not all(
        math.isfinite(value)
        for value in (alpha, gamma, scale, calibration_failure_budget, drift_constant)
    ):
        raise ValueError("all amortized-bound parameters must be finite")
    if alpha <= 11.0 / 24.0:
        raise ValueError("alpha must exceed 11/24 for the area-only theorem")
    lower = max(0.0, 1.0 - 2.0 * alpha)
    if not lower < gamma < 1.0 / 12.0:
        raise ValueError("gamma must lie in (max(0, 1-2alpha), 1/12)")
    if scale < math.sqrt(2.0):
        raise ValueError("scale must be at least sqrt(2)")
    if not isinstance(start, int) or isinstance(start, bool):
        raise ValueError("start must be an integer")
    kappa = 0.25 - gamma
    minimum_start = max(16, math.ceil(2.0 ** (1.0 / kappa)))
    if start < minimum_start:
        raise ValueError(
            f"start must be at least {minimum_start} for the equal-disk theorem constants"
        )
    if not 0.0 < calibration_failure_budget < 1.0:
        raise ValueError("calibration_failure_budget must lie in (0, 1)")
    if drift_constant < 0.0:
        raise ValueError("drift_constant cannot be negative")
    c1 = 2.0**gamma * (1.0 + drift_constant)
    c2 = 2.0 ** (2.0 * alpha + 2.0) + 6.0
    exponent_one = 2.0 * alpha + gamma
    exponent_two = 4.0 * alpha
    if not math.isfinite(exponent_one) or not math.isfinite(exponent_two):
        raise ValueError("the path-error exponents must be finite")
    value = (
        calibration_failure_budget
        + c1 * scale**-2.0 * _p_series_tail_upper(2.0 * alpha + gamma, start)
        + c2 * scale**-4.0 * _p_series_tail_upper(4.0 * alpha, start)
    )
    return min(1.0, value)
