from __future__ import annotations

import numpy as np
import pytest

from harmonic_dla.boundaries import poisson_return_delta


def test_poisson_return_fourier_moment() -> None:
    q = 0.4
    rng = np.random.default_rng(1234)
    uniforms = rng.random(300_000)
    delta = poisson_return_delta(q, uniforms)
    # The exterior Poisson kernel has E[cos(m delta)] = q**m.
    assert float(np.mean(np.cos(delta))) == pytest.approx(q, abs=4.0e-3)
    assert float(np.mean(np.cos(2.0 * delta))) == pytest.approx(q**2, abs=4.0e-3)
    assert abs(float(np.mean(np.sin(delta)))) < 4.0e-3


def test_poisson_return_rejects_endpoints() -> None:
    with pytest.raises(ValueError):
        poisson_return_delta(0.5, np.asarray([0.0, 0.5]))


def test_launch_point_validation() -> None:
    from harmonic_dla.boundaries import launch_point

    assert launch_point((1.0, 2.0), 2.0, 0.0) == pytest.approx((3.0, 2.0))
    with pytest.raises(ValueError, match="center"):
        launch_point((float("nan"), 0.0), 1.0, 0.0)
    with pytest.raises(ValueError, match="radius"):
        launch_point((0.0, 0.0), float("inf"), 0.0)
    with pytest.raises(ValueError, match="angle"):
        launch_point((0.0, 0.0), 1.0, float("nan"))
