from __future__ import annotations

import numpy as np
import pytest

from harmonic_dla.backends.numba_cpu.backend import _allocate_tree
from harmonic_dla.backends.numba_cpu.quadtree import brute_force_nearest, nearest_particle


def test_quadtree_matches_brute_force() -> None:
    rng = np.random.default_rng(42)
    positions = np.ascontiguousarray(rng.normal(size=(200, 2)), dtype=np.float64)
    tree = _allocate_tree(positions, 200, 200, 8, 12, 0.5)
    for query in rng.normal(size=(50, 2)):
        expected_index, expected_sq = brute_force_nearest(
            float(query[0]), float(query[1]), positions, 200
        )
        actual_index, actual_sq = nearest_particle(
            float(query[0]),
            float(query[1]),
            positions,
            200,
            tree.center_x,
            tree.center_y,
            tree.half_size,
            tree.children,
            tree.counts,
            tree.items,
        )
        assert actual_sq == pytest.approx(expected_sq, rel=0.0, abs=1.0e-14)
        assert np.sum((positions[actual_index] - query) ** 2) == pytest.approx(expected_sq)
        assert np.sum((positions[expected_index] - query) ** 2) == pytest.approx(expected_sq)
