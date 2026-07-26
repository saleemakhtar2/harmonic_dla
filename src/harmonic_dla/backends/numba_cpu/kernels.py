"""Compiled walk-on-spheres, frozen-probe, and growth kernels."""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from harmonic_dla.backends.numba_cpu.quadtree import (
    MAX_QUERY_STACK,
    insert_particle,
    nearest_particle_workspace,
)
from harmonic_dla.backends.numba_cpu.rng import seed_state, uniform_open

EXACT_RETURN = 0
UNIFORM_RESTART = 1
TARGET_RADIUS_SCALE = 0
DIAMETER_UPPER_SCALE = 1
_TWO_PI = 2.0 * math.pi


@njit(cache=True, inline="always")
def _poisson_return_delta(q: float, state: np.ndarray) -> float:
    u = uniform_open(state)
    phi = math.pi * (u - 0.5)
    scale = (1.0 - q) / (1.0 + q)
    return 2.0 * math.atan2(scale * math.sin(phi), math.cos(phi))


@njit(cache=True, inline="always")
def _uniform_angle(state: np.ndarray) -> float:
    return _TWO_PI * uniform_open(state)


@njit(cache=True)
def geometry_scan(
    positions: np.ndarray,
    particle_count: int,
    center_x: float,
    center_y: float,
    capture_distance: float,
) -> tuple[float, float, float, float, float]:
    """Return target radius and center-coordinate bounds in one linear pass."""
    min_x = positions[0, 0]
    max_x = min_x
    min_y = positions[0, 1]
    max_y = min_y
    radius_sq = 0.0
    for index in range(particle_count):
        x = positions[index, 0]
        y = positions[index, 1]
        dx = x - center_x
        dy = y - center_y
        candidate = dx * dx + dy * dy
        if candidate > radius_sq:
            radius_sq = candidate
        if x < min_x:
            min_x = x
        elif x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        elif y > max_y:
            max_y = y
    return math.sqrt(radius_sq) + capture_distance, min_x, max_x, min_y, max_y


@njit(cache=True, inline="always")
def target_radius(
    positions: np.ndarray,
    particle_count: int,
    center_x: float,
    center_y: float,
    capture_distance: float,
) -> float:
    """Radius about ``center`` containing the capture target."""
    radius, _min_x, _max_x, _min_y, _max_y = geometry_scan(
        positions,
        particle_count,
        center_x,
        center_y,
        capture_distance,
    )
    return radius


@njit(cache=True, inline="always")
def diameter_upper_from_bounds(
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    capture_distance: float,
) -> float:
    """Bounding-box upper bound on capture-target diameter."""
    width = max_x - min_x + 2.0 * capture_distance
    height = max_y - min_y + 2.0 * capture_distance
    return math.sqrt(width * width + height * height)


@njit(cache=True)
def _walk_attachment_with_workspace(
    master_seed: int,
    stream: int,
    positions: np.ndarray,
    particle_count: int,
    node_center_x: np.ndarray,
    node_center_y: np.ndarray,
    node_half_size: np.ndarray,
    children: np.ndarray,
    counts: np.ndarray,
    items: np.ndarray,
    center_x: float,
    center_y: float,
    capture_distance: float,
    birth_radius: float,
    death_radius: float,
    restart_mode: int,
    tolerance: float,
    max_steps: int,
    max_restarts: int,
    stack_nodes: np.ndarray,
    stack_bounds: np.ndarray,
    local_nodes: np.ndarray,
    local_bounds: np.ndarray,
) -> tuple[float, float, int, int, int]:
    """Simulate one walker using caller-owned nearest-query work arrays."""
    state = seed_state(master_seed, stream)

    angle = _uniform_angle(state)
    x = center_x + birth_radius * math.cos(angle)
    y = center_y + birth_radius * math.sin(angle)
    restarts = 0

    for step in range(1, max_steps + 1):
        nearest_index, nearest_sq = nearest_particle_workspace(
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
        nearest_distance = math.sqrt(nearest_sq)
        target_gap = nearest_distance - capture_distance
        radial_x = x - center_x
        radial_y = y - center_y
        radial_distance = math.sqrt(radial_x * radial_x + radial_y * radial_y)
        death_gap = death_radius - radial_distance

        if target_gap <= tolerance:
            if nearest_distance <= 0.0:
                return x, y, step, restarts, -4
            # Project to exact contact. The WOS tolerance controls only the angular error.
            projection = capture_distance / nearest_distance
            x = positions[nearest_index, 0] + (x - positions[nearest_index, 0]) * projection
            y = positions[nearest_index, 1] + (y - positions[nearest_index, 1]) * projection
            return x, y, step, restarts, 0

        if death_gap <= tolerance:
            if restarts >= max_restarts:
                return x, y, step, restarts, -2
            exit_angle = math.atan2(radial_y, radial_x)
            if restart_mode == EXACT_RETURN:
                q = birth_radius / death_radius
                angle = exit_angle + _poisson_return_delta(q, state)
            else:
                angle = _uniform_angle(state)
            x = center_x + birth_radius * math.cos(angle)
            y = center_y + birth_radius * math.sin(angle)
            restarts += 1
            continue

        sphere_radius = target_gap if target_gap < death_gap else death_gap
        if sphere_radius <= 0.0 or not math.isfinite(sphere_radius):
            return x, y, step, restarts, -4
        angle = _uniform_angle(state)
        x += sphere_radius * math.cos(angle)
        y += sphere_radius * math.sin(angle)

    return x, y, max_steps, restarts, -3


@njit(cache=True)
def walk_attachment(
    master_seed: int,
    stream: int,
    positions: np.ndarray,
    particle_count: int,
    node_center_x: np.ndarray,
    node_center_y: np.ndarray,
    node_half_size: np.ndarray,
    children: np.ndarray,
    counts: np.ndarray,
    items: np.ndarray,
    center_x: float,
    center_y: float,
    capture_distance: float,
    birth_radius: float,
    death_radius: float,
    restart_mode: int,
    tolerance: float,
    max_steps: int,
    max_restarts: int,
) -> tuple[float, float, int, int, int]:
    """Simulate one walker with a private nearest-query workspace."""
    stack_nodes = np.empty(MAX_QUERY_STACK, dtype=np.int32)
    stack_bounds = np.empty(MAX_QUERY_STACK, dtype=np.float64)
    local_nodes = np.empty(4, dtype=np.int32)
    local_bounds = np.empty(4, dtype=np.float64)
    return _walk_attachment_with_workspace(
        master_seed,
        stream,
        positions,
        particle_count,
        node_center_x,
        node_center_y,
        node_half_size,
        children,
        counts,
        items,
        center_x,
        center_y,
        capture_distance,
        birth_radius,
        death_radius,
        restart_mode,
        tolerance,
        max_steps,
        max_restarts,
        stack_nodes,
        stack_bounds,
        local_nodes,
        local_bounds,
    )


@njit(cache=True)
def grow_block(
    master_seed: int,
    positions: np.ndarray,
    start_count: int,
    stop_count: int,
    node_center_x: np.ndarray,
    node_center_y: np.ndarray,
    node_half_size: np.ndarray,
    children: np.ndarray,
    counts: np.ndarray,
    items: np.ndarray,
    node_count: np.ndarray,
    center_x: float,
    center_y: float,
    particle_radius: float,
    launch_margin_units: float,
    death_scale_kind: int,
    ratio_scale: float,
    ratio_alpha: float,
    restart_mode: int,
    tolerance: float,
    max_steps: int,
    max_restarts: int,
    initial_target_radius: float,
    initial_min_x: float,
    initial_max_x: float,
    initial_min_y: float,
    initial_max_y: float,
) -> tuple[int, int, int, int, float, float, float, float, float]:
    """Grow a fixed-center block and update geometry incrementally."""
    capture_distance = 2.0 * particle_radius
    walker_steps = 0
    restarts = 0
    count = start_count
    radius = initial_target_radius
    min_x = initial_min_x
    max_x = initial_max_x
    min_y = initial_min_y
    max_y = initial_max_y
    stack_nodes = np.empty(MAX_QUERY_STACK, dtype=np.int32)
    stack_bounds = np.empty(MAX_QUERY_STACK, dtype=np.float64)
    local_nodes = np.empty(4, dtype=np.int32)
    local_bounds = np.empty(4, dtype=np.float64)

    while count < stop_count:
        ratio = ratio_scale if ratio_alpha == 0.0 else ratio_scale * count**ratio_alpha
        birth_radius = radius + launch_margin_units * particle_radius
        if death_scale_kind == TARGET_RADIUS_SCALE:
            death_radius = ratio * radius
        else:
            diameter_upper = diameter_upper_from_bounds(
                min_x,
                max_x,
                min_y,
                max_y,
                capture_distance,
            )
            death_radius = ratio * diameter_upper
        if death_radius <= birth_radius:
            death_radius = birth_radius * (1.0 + 1.0e-12)

        x, y, steps, walker_restarts, status = _walk_attachment_with_workspace(
            master_seed,
            count,
            positions,
            count,
            node_center_x,
            node_center_y,
            node_half_size,
            children,
            counts,
            items,
            center_x,
            center_y,
            capture_distance,
            birth_radius,
            death_radius,
            restart_mode,
            tolerance,
            max_steps,
            max_restarts,
            stack_nodes,
            stack_bounds,
            local_nodes,
            local_bounds,
        )
        walker_steps += steps
        restarts += walker_restarts
        if status != 0:
            return count, walker_steps, restarts, status, radius, min_x, max_x, min_y, max_y

        positions[count, 0] = x
        positions[count, 1] = y
        if (
            insert_particle(
                count,
                positions,
                node_center_x,
                node_center_y,
                node_half_size,
                children,
                counts,
                items,
                node_count,
            )
            != 0
        ):
            return count, walker_steps, restarts, -5, radius, min_x, max_x, min_y, max_y

        center_dx = x - center_x
        center_dy = y - center_y
        candidate_radius = (
            math.sqrt(center_dx * center_dx + center_dy * center_dy) + capture_distance
        )
        if candidate_radius > radius:
            radius = candidate_radius
        if x < min_x:
            min_x = x
        elif x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        elif y > max_y:
            max_y = y
        count += 1

    return count, walker_steps, restarts, 0, radius, min_x, max_x, min_y, max_y


@njit(cache=True, parallel=True, nogil=True)
def probe_batch(
    master_seed: int,
    stream_offset: int,
    positions: np.ndarray,
    particle_count: int,
    node_center_x: np.ndarray,
    node_center_y: np.ndarray,
    node_half_size: np.ndarray,
    children: np.ndarray,
    counts: np.ndarray,
    items: np.ndarray,
    center_x: float,
    center_y: float,
    particle_radius: float,
    launch_margin_units: float,
    death_scale_kind: int,
    death_ratio: float,
    probes: int,
    tolerance: float,
    max_steps: int,
    max_restarts: int,
    restart_mode: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw independent frozen-cluster probes in parallel."""
    capture_distance = 2.0 * particle_radius
    radius, min_x, max_x, min_y, max_y = geometry_scan(
        positions,
        particle_count,
        center_x,
        center_y,
        capture_distance,
    )
    birth_radius = radius + launch_margin_units * particle_radius
    if death_scale_kind == TARGET_RADIUS_SCALE:
        death_radius = death_ratio * radius
    else:
        death_radius = death_ratio * diameter_upper_from_bounds(
            min_x,
            max_x,
            min_y,
            max_y,
            capture_distance,
        )
    if death_radius <= birth_radius:
        death_radius = birth_radius * (1.0 + 1.0e-12)

    outputs = np.empty((probes, 2), dtype=np.float64)
    statuses = np.empty(probes, dtype=np.int32)
    step_counts = np.empty(probes, dtype=np.int64)
    restart_counts = np.empty(probes, dtype=np.int64)
    for probe_index in prange(probes):  # ty: ignore[not-iterable]
        x, y, steps, restarts, status = walk_attachment(
            master_seed,
            stream_offset + probe_index,
            positions,
            particle_count,
            node_center_x,
            node_center_y,
            node_half_size,
            children,
            counts,
            items,
            center_x,
            center_y,
            capture_distance,
            birth_radius,
            death_radius,
            restart_mode,
            tolerance,
            max_steps,
            max_restarts,
        )
        outputs[probe_index, 0] = x
        outputs[probe_index, 1] = y
        statuses[probe_index] = status
        step_counts[probe_index] = steps
        restart_counts[probe_index] = restarts
    return outputs, statuses, step_counts, restart_counts
