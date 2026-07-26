from __future__ import annotations

import math

import pytest

from harmonic_dla.certificates import (
    amortized_local_tv_bound,
    amortized_path_budget_upper,
    constant_ratio_for_path_budget,
    monte_carlo_tv_bound,
    one_shot_diameter_tv_bound,
    path_budget,
    residual_tv_bound,
    self_centered_tv_bound,
)


def test_self_centered_bound_matches_formula() -> None:
    rho = 2.5
    expected = 1.0 / (rho * rho * (rho * rho - 1.0))
    assert self_centered_tv_bound(rho) == pytest.approx(expected)


def test_residual_bound_adds_dipole_term() -> None:
    rho = 3.0
    residual = 0.2
    assert residual_tv_bound(rho, residual) == pytest.approx(
        residual / rho**2 + self_centered_tv_bound(rho)
    )


def test_monte_carlo_bound_is_conservative_and_decreases() -> None:
    coarse = monte_carlo_tv_bound(4.0, 0.0, 100, 0.01)
    fine = monte_carlo_tv_bound(4.0, 0.0, 10_000, 0.01)
    assert 0.0 < fine < coarse <= 1.0


def test_one_shot_diameter_bound() -> None:
    rho = 3.0
    probes = 1_000
    failure = 0.01
    expected = min(
        1.0,
        2.0 / rho**2 * math.sqrt(math.log(4.0 / failure) / probes)
        + 5.0 / (rho**2 * (rho**2 - 1.0)),
    )
    assert one_shot_diameter_tv_bound(rho, probes, failure) == pytest.approx(expected)


def test_amortized_local_bound_matches_theorem_term() -> None:
    n = 1024
    alpha = 0.48
    gamma = 0.06
    scale = 2.0
    drift = 3.0 * math.sqrt(2.0)
    expected = min(
        1.0,
        2.0**gamma * (1.0 + drift) * scale**-2.0 * n ** (-(2.0 * alpha + gamma))
        + (2.0 ** (2.0 * alpha + 2.0) + 6.0) * scale**-4.0 * n ** (-4.0 * alpha),
    )
    assert amortized_local_tv_bound(
        n, alpha=alpha, gamma=gamma, scale=scale, drift_constant=drift
    ) == pytest.approx(expected)


def test_product_path_budget() -> None:
    assert path_budget([0.1, 0.2]) == pytest.approx(1.0 - 0.9 * 0.8)


def test_product_path_budget_preserves_tiny_sum() -> None:
    # Naive multiplication rounds every factor to one at this scale.
    bounds = [1.0e-16] * 100
    assert path_budget(bounds) == pytest.approx(1.0e-14, rel=1.0e-12)


def test_constant_ratio_achieves_requested_budget() -> None:
    steps = 100
    budget = 0.01
    rho = constant_ratio_for_path_budget(steps, budget)
    step = self_centered_tv_bound(rho)
    assert path_budget([step] * steps) <= budget * (1.0 + 1.0e-10)


def test_constant_ratio_handles_tiny_budget_without_cancellation() -> None:
    steps = 10**6
    budget = 1.0e-12
    rho = constant_ratio_for_path_budget(steps, budget)
    assert math.isfinite(rho)
    assert path_budget([self_centered_tv_bound(rho)] * steps) <= budget * (1.0 + 1.0e-10)


def test_amortized_path_budget_upper_is_finite_and_monotone() -> None:
    coarse = amortized_path_budget_upper(
        alpha=0.48,
        gamma=0.06,
        scale=16.0,
        start=128,
        calibration_failure_budget=1.0e-3,
    )
    fine = amortized_path_budget_upper(
        alpha=0.48,
        gamma=0.06,
        scale=32.0,
        start=128,
        calibration_failure_budget=1.0e-3,
    )
    assert 0.0 < fine < coarse <= 1.0


def test_amortized_path_budget_requires_summability() -> None:
    with pytest.raises(ValueError):
        amortized_path_budget_upper(
            alpha=0.45,
            gamma=0.05,
            scale=2.0,
            start=128,
            calibration_failure_budget=1.0e-3,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0.45, "gamma": 0.06},  # alpha <= 11/24
        {"alpha": 0.48, "gamma": 0.01},  # gamma <= max(0, 1 - 2 alpha)
        {"alpha": 0.48, "gamma": 1.0 / 12.0},
        {"alpha": 0.48, "gamma": 0.06, "scale": math.sqrt(2.0) - 1.0e-12},
        {"alpha": 0.48, "gamma": 0.06, "start": 16},
        {"alpha": 0.48, "gamma": 0.06, "calibration_failure_budget": 0.0},
    ],
)
def test_amortized_path_budget_matches_paper_schedule_validation(kwargs: dict[str, float]) -> None:
    params: dict[str, float | int] = {
        "alpha": 0.48,
        "gamma": 0.06,
        "scale": 2.0,
        "start": 1024,
        "calibration_failure_budget": 1.0e-3,
    }
    params.update(kwargs)
    with pytest.raises(ValueError):
        amortized_path_budget_upper(**params)  # type: ignore[arg-type]


@pytest.mark.parametrize("rho", [1.0, 0.5, math.inf, math.nan])
def test_invalid_rho_rejected(rho: float) -> None:
    with pytest.raises(ValueError):
        self_centered_tv_bound(rho)
