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


def test_quadtree_accepts_translated_positions() -> None:
    positions = np.asarray([[100.0, -75.0], [101.0, -75.0]], dtype=np.float64)
    tree = _allocate_tree(positions, 2, 2, 8, 12, 0.5)
    index, squared = brute_force_nearest(100.2, -75.1, positions, 2)
    actual_index, actual_squared = nearest_particle(
        100.2,
        -75.1,
        positions,
        2,
        tree.center_x,
        tree.center_y,
        tree.half_size,
        tree.children,
        tree.counts,
        tree.items,
    )
    assert actual_index == index
    assert actual_squared == pytest.approx(squared)


def test_quadtree_deduplicates_coincident_positions() -> None:
    positions = np.zeros((13, 2), dtype=np.float64)
    tree = _allocate_tree(positions, 13, 13, 12, 10, 0.5)
    index, squared = nearest_particle(
        0.1,
        0.0,
        positions,
        13,
        tree.center_x,
        tree.center_y,
        tree.half_size,
        tree.children,
        tree.counts,
        tree.items,
    )
    assert index >= 0
    assert squared == pytest.approx(0.01)
