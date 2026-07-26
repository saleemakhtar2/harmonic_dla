from __future__ import annotations

import math

import numpy as np
import pytest

from harmonic_dla.calibration import empirical_center, validate_center


def test_empirical_center_and_independent_validation() -> None:
    search = np.asarray([[1.0, 0.0], [3.0, 2.0]], dtype=np.float64)
    center = empirical_center(search)
    assert center == pytest.approx((2.0, 1.0))

    validation = np.asarray([[2.0, 1.0], [2.2, 1.0]], dtype=np.float64)
    result = validate_center(
        center,
        validation,
        target_radius=4.0,
        death_ratio=3.0,
        failure_probability=0.01,
        search_probes=2,
    )
    assert result.search_probes == 2
    assert result.validation_probes == 2
    assert result.empirical_residual == pytest.approx(0.1)
    assert 0.0 <= result.tv_bound <= 1.0


def test_calibration_rejects_nonfinite_and_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        empirical_center(np.asarray([[math.nan, 0.0]], dtype=np.float64))

    validation = np.asarray([[0.0, 0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="center"):
        validate_center(
            (math.inf, 0.0),
            validation,
            target_radius=1.0,
            death_ratio=2.0,
            failure_probability=0.1,
            search_probes=1,
        )
    with pytest.raises(ValueError, match="target_radius"):
        validate_center(
            (0.0, 0.0),
            validation,
            target_radius=math.nan,
            death_ratio=2.0,
            failure_probability=0.1,
            search_probes=1,
        )
    with pytest.raises(ValueError, match="search_probes"):
        validate_center(
            (0.0, 0.0),
            validation,
            target_radius=1.0,
            death_ratio=2.0,
            failure_probability=0.1,
            search_probes=0,
        )
