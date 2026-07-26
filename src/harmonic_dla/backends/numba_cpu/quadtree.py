"""Packed-array quadtree for exact nearest-neighbour queries."""

from __future__ import annotations

import numpy as np
from numba import njit

MAX_QUERY_STACK = 1024


@njit(cache=True, inline="always")
def _quadrant(x: float, y: float, center_x: float, center_y: float) -> int:
    east = 1 if x >= center_x else 0
    north = 1 if y >= center_y else 0
    return east + 2 * north


@njit(cache=True, inline="always")
def _aabb_distance_sq(
    x: float,
    y: float,
    center_x: float,
    center_y: float,
    half_size: float,
) -> float:
    dx = abs(x - center_x) - half_size
    dy = abs(y - center_y) - half_size
    if dx < 0.0:
        dx = 0.0
    if dy < 0.0:
        dy = 0.0
    return dx * dx + dy * dy


@njit(cache=True)
def initialize_tree(
    node_center_x: np.ndarray,
    node_center_y: np.ndarray,
    node_half_size: np.ndarray,
    children: np.ndarray,
    counts: np.ndarray,
    items: np.ndarray,
    node_count: np.ndarray,
    root_center_x: float,
    root_center_y: float,
    root_half_size: float,
) -> None:
    """Reset packed arrays to one empty root node."""
    children.fill(-1)
    counts.fill(0)
    items.fill(-1)
    node_center_x[0] = root_center_x
    node_center_y[0] = root_center_y
    node_half_size[0] = root_half_size
    node_count[0] = 1


@njit(cache=True, inline="always")
def _create_children(
    node: int,
    node_center_x: np.ndarray,
    node_center_y: np.ndarray,
    node_half_size: np.ndarray,
    children: np.ndarray,
    counts: np.ndarray,
    items: np.ndarray,
    node_count: np.ndarray,
) -> int:
    base = int(node_count[0])
    if base + 4 > node_center_x.shape[0]:
        return -1
    child_half = node_half_size[node] * 0.5
    parent_x = node_center_x[node]
    parent_y = node_center_y[node]
    for quadrant in range(4):
        child = base + quadrant
        dx = child_half if quadrant % 2 == 1 else -child_half
        dy = child_half if quadrant >= 2 else -child_half
        node_center_x[child] = parent_x + dx
        node_center_y[child] = parent_y + dy
        node_half_size[child] = child_half
        children[node, quadrant] = child
        for q in range(4):
            children[child, q] = -1
        counts[child] = 0
        for slot in range(items.shape[1]):
            items[child, slot] = -1
    node_count[0] = base + 4
    return base


@njit(cache=True)
def insert_particle(
    particle_index: int,
    positions: np.ndarray,
    node_center_x: np.ndarray,
    node_center_y: np.ndarray,
    node_half_size: np.ndarray,
    children: np.ndarray,
    counts: np.ndarray,
    items: np.ndarray,
    node_count: np.ndarray,
) -> int:
    """Insert one particle; return zero on success and minus one on capacity failure."""
    x = positions[particle_index, 0]
    y = positions[particle_index, 1]
    root_half = node_half_size[0]
    if abs(x - node_center_x[0]) > root_half or abs(y - node_center_y[0]) > root_half:
        return -1

    node = 0
    bucket_size = items.shape[1]
    while True:
        if children[node, 0] < 0:
            count = int(counts[node])
            if count < bucket_size:
                items[node, count] = particle_index
                counts[node] = count + 1
                return 0

            if _create_children(
                node,
                node_center_x,
                node_center_y,
                node_half_size,
                children,
                counts,
                items,
                node_count,
            ) < 0:
                return -1

            for slot in range(bucket_size):
                old_index = int(items[node, slot])
                old_x = positions[old_index, 0]
                old_y = positions[old_index, 1]
                quadrant = _quadrant(old_x, old_y, node_center_x[node], node_center_y[node])
                child = int(children[node, quadrant])
                child_count = int(counts[child])
                items[child, child_count] = old_index
                counts[child] = child_count + 1
                items[node, slot] = -1
            counts[node] = 0

        quadrant = _quadrant(x, y, node_center_x[node], node_center_y[node])
        node = int(children[node, quadrant])


@njit(cache=True, inline="always")
def brute_force_nearest(
    x: float,
    y: float,
    positions: np.ndarray,
    particle_count: int,
) -> tuple[int, float]:
    """Return the exact nearest particle by a linear scan."""
    best_index = -1
    best_sq = np.inf
    for index in range(particle_count):
        dx = x - positions[index, 0]
        dy = y - positions[index, 1]
        distance_sq = dx * dx + dy * dy
        if distance_sq < best_sq:
            best_sq = distance_sq
            best_index = index
    return best_index, best_sq


@njit(cache=True)
def nearest_particle_workspace(
    x: float,
    y: float,
    positions: np.ndarray,
    particle_count: int,
    node_center_x: np.ndarray,
    node_center_y: np.ndarray,
    node_half_size: np.ndarray,
    children: np.ndarray,
    counts: np.ndarray,
    items: np.ndarray,
    stack_nodes: np.ndarray,
    stack_bounds: np.ndarray,
    local_nodes: np.ndarray,
    local_bounds: np.ndarray,
) -> tuple[int, float]:
    """Nearest query using caller-owned work arrays to avoid per-step allocation."""
    stack_nodes[0] = 0
    stack_bounds[0] = 0.0
    top = 1
    best_index = -1
    best_sq = np.inf

    while top > 0:
        top -= 1
        node = int(stack_nodes[top])
        lower = stack_bounds[top]
        if lower >= best_sq:
            continue

        if children[node, 0] < 0:
            count = int(counts[node])
            for slot in range(count):
                index = int(items[node, slot])
                dx = x - positions[index, 0]
                dy = y - positions[index, 1]
                distance_sq = dx * dx + dy * dy
                if distance_sq < best_sq:
                    best_sq = distance_sq
                    best_index = index
            continue

        local_count = 0
        for quadrant in range(4):
            child = int(children[node, quadrant])
            child_bound = _aabb_distance_sq(
                x,
                y,
                node_center_x[child],
                node_center_y[child],
                node_half_size[child],
            )
            if child_bound < best_sq:
                insert_at = local_count
                while insert_at > 0 and child_bound < local_bounds[insert_at - 1]:
                    local_nodes[insert_at] = local_nodes[insert_at - 1]
                    local_bounds[insert_at] = local_bounds[insert_at - 1]
                    insert_at -= 1
                local_nodes[insert_at] = child
                local_bounds[insert_at] = child_bound
                local_count += 1

        if top + local_count >= stack_nodes.shape[0]:
            return brute_force_nearest(x, y, positions, particle_count)
        for index in range(local_count - 1, -1, -1):
            stack_nodes[top] = local_nodes[index]
            stack_bounds[top] = local_bounds[index]
            top += 1

    if best_index < 0:
        return brute_force_nearest(x, y, positions, particle_count)
    return best_index, best_sq


@njit(cache=True)
def nearest_particle(
    x: float,
    y: float,
    positions: np.ndarray,
    particle_count: int,
    node_center_x: np.ndarray,
    node_center_y: np.ndarray,
    node_half_size: np.ndarray,
    children: np.ndarray,
    counts: np.ndarray,
    items: np.ndarray,
) -> tuple[int, float]:
    """Convenience wrapper used by tests; hot kernels use caller-owned workspaces."""
    stack_nodes = np.empty(MAX_QUERY_STACK, dtype=np.int32)
    stack_bounds = np.empty(MAX_QUERY_STACK, dtype=np.float64)
    local_nodes = np.empty(4, dtype=np.int32)
    local_bounds = np.empty(4, dtype=np.float64)
    return nearest_particle_workspace(
        x,
        y,
        positions,
        particle_count,
        node_center_x,
        node_center_y,
        node_half_size,
        children,
        counts,
        items,
        stack_nodes,
        stack_bounds,
        local_nodes,
        local_bounds,
    )
