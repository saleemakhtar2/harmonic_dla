from __future__ import annotations

import numpy as np

from harmonic_dla.backends.numba_cpu.rng import seed_state


def test_seed_and_stream_are_not_combined_with_xor_collision() -> None:
    first = seed_state(0, 1)
    second = seed_state(4_950_324_153_757_272_190, 3)
    assert not np.array_equal(first, second)
