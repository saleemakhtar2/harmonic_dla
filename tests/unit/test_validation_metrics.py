from __future__ import annotations

import numpy as np
import pytest

from harmonic_dla.validation import (
    PRIMARY_RHOS,
    angle_histogram,
    assign_nearest_disk,
    bootstrap_interval,
    bounded_lipschitz_discrepancy,
    fit_power_law,
    histogram_total_variation,
    make_lopsided_comb,
    particle_id_total_variation,
    schedule_seed,
)


def test_lopsided_comb_is_deterministic_connected_and_asymmetric() -> None:
    first = make_lopsided_comb()
    second = make_lopsided_comb()
    np.testing.assert_array_equal(first, second)
    distances = np.sqrt(np.sum((first[:, None] - first[None, :]) ** 2, axis=-1))
    distances[distances == 0.0] = np.inf
    assert np.min(distances) == pytest.approx(1.0)
    centered = first - np.mean(first, axis=0)
    second_moment = centered.T @ centered / first.shape[0]
    assert np.linalg.det(second_moment) > 0.0
    neighbors = distances <= 1.0 + 1.0e-12
    seen = {0}
    pending = [0]
    while pending:
        current = pending.pop()
        for candidate in np.flatnonzero(neighbors[current]):
            if int(candidate) not in seen:
                seen.add(int(candidate))
                pending.append(int(candidate))
    assert len(seen) == first.shape[0]
    assert not np.array_equal(first, first * np.asarray([-1.0, 1.0]))


def test_nearest_disk_assignment_breaks_ties_by_lowest_index() -> None:
    disks = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    attachments = np.asarray([[0.5, 0.0], [1.0, 0.2]], dtype=np.float64)
    np.testing.assert_array_equal(assign_nearest_disk(attachments, disks), [0, 1])


def test_identical_empirical_laws_have_zero_discrepancy() -> None:
    values = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
    assert histogram_total_variation(values, values, bins=16) == 0.0
    assert particle_id_total_variation([0, 1, 1], [0, 1, 1]) == 0.0
    assert bounded_lipschitz_discrepancy(values, values) == 0.0
    np.testing.assert_allclose(angle_histogram(values, (0.0, 0.0), bins=16).sum(), 1.0)


def test_discrepancies_have_exact_fixed_partition_values() -> None:
    first = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    second = np.asarray([[-1.0, 0.0], [-1.0, 0.0]], dtype=np.float64)
    assert histogram_total_variation(first, second, bins=16) == pytest.approx(1.0)
    assert particle_id_total_variation([0, 0, 1, 1], [0, 1, 1, 1]) == pytest.approx(0.25)
    assert bounded_lipschitz_discrepancy([[0.0, 0.0]], [[1.0, 0.0]]) == pytest.approx(1.0)


def test_schedule_seed_is_deterministic_and_streams_are_distinct() -> None:
    key = ("v1", "pilot", "comb", "uniform", 4, 2, "evaluation")
    assert schedule_seed(key) == schedule_seed(key)
    assert schedule_seed((*key[:-1], "reference")) != schedule_seed(key)


def test_bootstrap_interval_is_deterministic() -> None:
    values = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    assert bootstrap_interval(values, resamples=200, seed=11) == bootstrap_interval(
        values,
        resamples=200,
        seed=11,
    )


def test_bootstrap_complete_mask_controls_quantiles() -> None:
    values = np.asarray([0.0, 100.0, 1.0], dtype=np.float64)
    complete = np.asarray([True, False, True])
    masked = bootstrap_interval(values, complete=complete, resamples=40, seed=7)
    rng = np.random.default_rng(7)
    draws = np.asarray([0.0, 1.0])[rng.integers(0, 2, size=(40, 2))]
    estimates = np.median(draws, axis=1)
    expected = np.percentile(estimates, [2.5, 97.5])
    assert masked.lower == pytest.approx(expected[0])
    assert masked.upper == pytest.approx(expected[1])


def test_fit_power_law_requires_preregistered_primary_window() -> None:
    errors = {rho: float(rho) ** -2 for rho in PRIMARY_RHOS}
    fit = fit_power_law(errors)
    assert tuple(fit["ratios"]) == PRIMARY_RHOS
    assert fit["slope"] == pytest.approx(-2.0, abs=0.2)
    with pytest.raises(ValueError, match="primary"):
        fit_power_law(dict.fromkeys(PRIMARY_RHOS[:-1], 1.0))


@pytest.mark.parametrize("kwargs", [{"order": True}, {"order": 2.5}, {"m": False}, {"m": 0}])
def test_fit_power_law_rejects_invalid_order_aliases(kwargs: dict[str, object]) -> None:
    errors = dict.fromkeys(PRIMARY_RHOS, 1.0)
    with pytest.raises(ValueError, match=r"order|positive"):
        fit_power_law(errors, **kwargs)


def test_schedule_seed_rejects_unstable_values() -> None:
    with pytest.raises(TypeError, match="stable"):
        schedule_seed(("v1", object()))
    with pytest.raises(TypeError, match="tuple or list"):
        schedule_seed({"v1", "pilot"})  # type: ignore[arg-type]
