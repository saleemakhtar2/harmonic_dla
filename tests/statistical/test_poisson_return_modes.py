from __future__ import annotations

import numpy as np
import pytest

from harmonic_dla.boundaries import poisson_return_delta


@pytest.mark.slow
def test_poisson_return_modes_through_four() -> None:
    q = 0.4
    rng = np.random.default_rng(20260726)
    delta = poisson_return_delta(q, rng.random(1_000_000))
    for mode in range(1, 5):
        observed = complex(np.mean(np.exp(1j * mode * delta)))
        assert observed.real == pytest.approx(q**mode, abs=2.5e-3)
        assert abs(observed.imag) < 2.5e-3
