"""Production-oriented Numba CPU backend."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from numba import get_num_threads, set_num_threads

from harmonic_dla.backends.numba_cpu.kernels import (
    DIAMETER_UPPER_SCALE,
    EXACT_RETURN,
    TARGET_RADIUS_SCALE,
    UNIFORM_RESTART,
    geometry_scan,
    grow_block,
    probe_batch,
)
from harmonic_dla.backends.numba_cpu.quadtree import initialize_tree, insert_particle
from harmonic_dla.calibration import empirical_center, validate_center
from harmonic_dla.certificates import amortized_local_tv_bound
from harmonic_dla.config import RunConfig, config_to_dict
from harmonic_dla.enums import CalibrationStrategy, RestartMode
from harmonic_dla.exceptions import SimulationError
from harmonic_dla.models import ProbeResult, SimulationDiagnostics, SimulationResult
from harmonic_dla.provenance import runtime_provenance
from harmonic_dla.schedules import AmortizedSchedule

FloatArray = npt.NDArray[np.float64]


@dataclass(slots=True)
class _TreeStorage:
    center_x: FloatArray
    center_y: FloatArray
    half_size: FloatArray
    children: npt.NDArray[np.int32]
    counts: npt.NDArray[np.int32]
    items: npt.NDArray[np.int32]
    node_count: npt.NDArray[np.int32]


@dataclass(slots=True)
class _Geometry:
    target_radius: float
    min_x: float
    max_x: float
    min_y: float
    max_y: float


def _allocate_tree(
    positions: FloatArray,
    particle_count: int,
    maximum_particles: int,
    bucket_size: int,
    node_factor: int,
    particle_radius: float,
) -> _TreeStorage:
    max_nodes = max(64, node_factor * maximum_particles + 64)
    tree = _TreeStorage(
        center_x=np.empty(max_nodes, dtype=np.float64),
        center_y=np.empty(max_nodes, dtype=np.float64),
        half_size=np.empty(max_nodes, dtype=np.float64),
        children=np.empty((max_nodes, 4), dtype=np.int32),
        counts=np.empty(max_nodes, dtype=np.int32),
        items=np.empty((max_nodes, bucket_size), dtype=np.int32),
        node_count=np.empty(1, dtype=np.int32),
    )
    capture_distance = 2.0 * particle_radius
    # A chain of touching particles cannot extend farther than this conservative root.
    # Centering from the observed bounds keeps the index translation invariant for probes,
    # while the generous half-size still reserves room for future growth from the seed.
    observed = positions[:particle_count]
    min_x = float(np.min(observed[:, 0]))
    max_x = float(np.max(observed[:, 0]))
    min_y = float(np.min(observed[:, 1]))
    max_y = float(np.max(observed[:, 1]))
    root_center_x = 0.5 * (min_x + max_x)
    root_center_y = 0.5 * (min_y + max_y)
    root_half_size = max(
        capture_distance * (maximum_particles + 4),
        0.5 * max(max_x - min_x, max_y - min_y) + capture_distance,
    )
    initialize_tree(
        tree.center_x,
        tree.center_y,
        tree.half_size,
        tree.children,
        tree.counts,
        tree.items,
        tree.node_count,
        root_center_x,
        root_center_y,
        root_half_size,
    )
    for index in range(particle_count):
        status = insert_particle(
            index,
            positions,
            tree.center_x,
            tree.center_y,
            tree.half_size,
            tree.children,
            tree.counts,
            tree.items,
            tree.node_count,
        )
        if status != 0:
            raise SimulationError("quadtree capacity exhausted while building the index")
    return tree


def _check_probe_statuses(statuses: npt.NDArray[np.int32]) -> None:
    failed = np.flatnonzero(statuses)
    if failed.size:
        first = int(failed[0])
        status = int(statuses[first])
        raise SimulationError(f"frozen probe {first} failed with status {status}")


def _pilot_center(
    policy: str,
    positions: FloatArray,
    particle_count: int,
    previous: tuple[float, float],
    has_previous_calibration: bool,
) -> tuple[float, float]:
    if policy == "seed":
        return float(positions[0, 0]), float(positions[0, 1])
    if policy == "mass-centroid":
        mean = np.mean(positions[:particle_count], axis=0)
        return float(mean[0]), float(mean[1])
    if has_previous_calibration:
        return previous
    # The seed is in the target and hence in its convex hull, unlike an arbitrary user center.
    return float(positions[0, 0]), float(positions[0, 1])


def _geometry(
    positions: FloatArray,
    particle_count: int,
    center: tuple[float, float],
    capture_distance: float,
) -> _Geometry:
    radius, min_x, max_x, min_y, max_y = geometry_scan(
        positions,
        particle_count,
        center[0],
        center[1],
        capture_distance,
    )
    return _Geometry(radius, min_x, max_x, min_y, max_y)


def _checkpoint_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.checkpoint.npz")


class NumbaCPUBackend:
    """Sequential aggregate growth with compiled kernels and parallel frozen probes."""

    name = "numba-cpu"

    def _probe_with_tree(
        self,
        positions: FloatArray,
        particle_count: int,
        tree: _TreeStorage,
        particle_radius: float,
        center: tuple[float, float],
        death_scale_kind: int,
        death_ratio: float,
        launch_margin: float,
        probes: int,
        seed: int,
        stream_offset: int,
        tolerance: float,
        max_steps: int,
        max_restarts: int,
        restart_mode: int = UNIFORM_RESTART,
    ) -> tuple[FloatArray, int, int]:
        attachments, statuses, step_counts, restart_counts = probe_batch(
            seed,
            stream_offset,
            positions,
            particle_count,
            tree.center_x,
            tree.center_y,
            tree.half_size,
            tree.children,
            tree.counts,
            tree.items,
            center[0],
            center[1],
            particle_radius,
            launch_margin,
            death_scale_kind,
            death_ratio,
            probes,
            tolerance,
            max_steps,
            max_restarts,
            restart_mode,
        )
        _check_probe_statuses(statuses)
        return attachments, int(np.sum(step_counts)), int(np.sum(restart_counts))

    def probe(
        self,
        positions: FloatArray,
        particle_radius: float,
        center: tuple[float, float],
        death_ratio: float,
        launch_margin: float,
        probes: int,
        seed: int,
    ) -> FloatArray:
        """Draw frozen-cluster target-radius-scaled uniform-restart probes."""
        if probes < 1:
            raise ValueError("probes must be positive")
        array = np.ascontiguousarray(positions, dtype=np.float64)
        tree = _allocate_tree(
            array,
            int(array.shape[0]),
            int(array.shape[0]),
            12,
            10,
            particle_radius,
        )
        attachments, _steps, _restarts = self._probe_with_tree(
            array,
            int(array.shape[0]),
            tree,
            particle_radius,
            center,
            TARGET_RADIUS_SCALE,
            death_ratio,
            launch_margin,
            probes,
            seed,
            1 << 40,
            1.0e-6,
            100_000,
            100_000,
        )
        return attachments

    def probe_detailed(
        self,
        positions: FloatArray,
        particle_radius: float,
        center: tuple[float, float],
        death_ratio: float,
        launch_margin: float,
        probes: int,
        seed: int,
        restart_mode: RestartMode,
    ) -> ProbeResult:
        """Draw probes with an explicit return law and walker diagnostics."""
        if probes < 1:
            raise ValueError("probes must be positive")
        if not isinstance(restart_mode, RestartMode) or restart_mode not in (
            RestartMode.UNIFORM_RESTART,
            RestartMode.EXACT_RETURN,
        ):
            raise ValueError("restart_mode must be exact-return or uniform-restart")
        array = np.ascontiguousarray(positions, dtype=np.float64)
        tree = _allocate_tree(
            array,
            int(array.shape[0]),
            int(array.shape[0]),
            12,
            10,
            particle_radius,
        )
        attachments, walker_steps, restarts = self._probe_with_tree(
            array,
            int(array.shape[0]),
            tree,
            particle_radius,
            center,
            TARGET_RADIUS_SCALE,
            death_ratio,
            launch_margin,
            probes,
            seed,
            1 << 40,
            1.0e-6,
            100_000,
            100_000,
            int(restart_mode),
        )
        return ProbeResult(attachments, walker_steps, restarts)

    def _build_result(
        self,
        positions: FloatArray,
        count: int,
        config: RunConfig,
        diagnostics: SimulationDiagnostics,
        started: float,
        tree: _TreeStorage,
        center: tuple[float, float],
        *,
        complete: bool,
    ) -> SimulationResult:
        metadata = runtime_provenance()
        metadata.update(
            {
                "schema_version": 1,
                "complete": complete,
                "elapsed_seconds": time.perf_counter() - started,
                "death_ratio": config.boundary.death_ratio,
                "launch_margin_particle_radii": config.boundary.launch_margin,
                "walker_tolerance": config.walker.tolerance,
                "quadtree_nodes": int(tree.node_count[0]),
                "numba_threads": int(get_num_threads()),
                "final_center": [center[0], center[1]],
                "numerical_model": "walk-on-spheres with epsilon angular attachment error",
                "paper_certificate_scope": (
                    "one-step and blockwise bounds; see manuscript assumptions"
                ),
                "config": config_to_dict(config),
            }
        )
        return SimulationResult(
            positions=np.ascontiguousarray(positions[:count]),
            particle_radius=config.particle_radius,
            seed=config.seed,
            backend=self.name,
            restart_mode=config.boundary.restart_mode.name.lower(),
            diagnostics=diagnostics,
            metadata=metadata,
        )

    def _simulate(self, config: RunConfig) -> SimulationResult:
        """Grow one aggregate according to ``config`` with the requested thread state."""
        started = time.perf_counter()
        positions = np.empty((config.particles, 2), dtype=np.float64)
        positions[0] = (0.0, 0.0)
        tree = _allocate_tree(
            positions,
            1,
            config.particles,
            config.performance.quadtree_bucket_size,
            config.performance.quadtree_node_factor,
            config.particle_radius,
        )

        count = 1
        total_growth_steps = 0
        total_growth_restarts = 0
        total_probe_steps = 0
        total_probe_restarts = 0
        total_probes = 0
        checkpoint_path = _checkpoint_path(config.output.path)
        checkpoint_initialized = False
        if (
            config.output.checkpoint_every > 0
            and checkpoint_path.exists()
            and not config.output.overwrite
        ):
            raise SimulationError(f"refusing to overwrite existing checkpoint: {checkpoint_path}")
        center = config.boundary.center
        center_history: list[tuple[float, float]] = []
        size_history: list[int] = []
        bound_history: list[float] = []
        failure_history: list[float] = []
        ratio_history: list[float] = []
        block_index_history: list[int] = []
        probe_count_history: list[int] = []
        search_probe_history: list[int] = []
        validation_probe_history: list[int] = []
        calibration_index = 0
        last_checkpoint = 1
        capture_distance = 2.0 * config.particle_radius
        geometry = _geometry(positions, count, center, capture_distance)

        schedule: AmortizedSchedule | None = None
        if (
            config.boundary.restart_mode is RestartMode.CONTROLLED_RESTART
            and config.calibration.strategy is CalibrationStrategy.PAPER_AMORTIZED
        ):
            schedule = AmortizedSchedule(
                alpha=config.calibration.alpha,
                gamma=config.calibration.gamma,
                scale=config.calibration.scale,
                calibration_failure_budget=config.calibration.failure_budget,
                start=config.calibration.start,
            )

        while count < config.particles:
            did_calibrate = False
            death_scale_kind = TARGET_RADIUS_SCALE
            ratio_scale = config.boundary.death_ratio
            ratio_alpha = 0.0
            search_probe_count = 0
            validation_probe_count = 0
            kernel_restart_mode = int(config.boundary.restart_mode)
            stop = config.particles

            if schedule is not None and count < schedule.start:
                stop = min(config.particles, schedule.start)
                kernel_restart_mode = (
                    EXACT_RETURN if config.calibration.exact_prefix else UNIFORM_RESTART
                )
                if config.performance.growth_chunk_size > 0:
                    stop = min(stop, count + config.performance.growth_chunk_size)
            elif config.boundary.restart_mode is RestartMode.CONTROLLED_RESTART:
                pilot = _pilot_center(
                    config.calibration.pilot,
                    positions,
                    count,
                    center,
                    bool(center_history),
                )
                base_stream = (1 << 40) + calibration_index * (1 << 30)

                if schedule is None:
                    search, steps, restarts = self._probe_with_tree(
                        positions,
                        count,
                        tree,
                        config.particle_radius,
                        pilot,
                        TARGET_RADIUS_SCALE,
                        config.boundary.death_ratio,
                        config.boundary.launch_margin,
                        config.calibration.search_probes,
                        config.seed ^ 0x5A17,
                        base_stream,
                        config.walker.tolerance,
                        config.walker.max_steps,
                        config.walker.max_restarts,
                    )
                    total_probe_steps += steps
                    total_probe_restarts += restarts
                    center = empirical_center(search)
                    validation, steps, restarts = self._probe_with_tree(
                        positions,
                        count,
                        tree,
                        config.particle_radius,
                        center,
                        TARGET_RADIUS_SCALE,
                        config.boundary.death_ratio,
                        config.boundary.launch_margin,
                        config.calibration.validation_probes,
                        config.seed ^ 0xA51C,
                        base_stream + config.calibration.search_probes,
                        config.walker.tolerance,
                        config.walker.max_steps,
                        config.walker.max_restarts,
                    )
                    total_probe_steps += steps
                    total_probe_restarts += restarts
                    geometry = _geometry(positions, count, center, capture_distance)
                    calibration = validate_center(
                        center,
                        validation,
                        geometry.target_radius,
                        config.boundary.death_ratio,
                        config.calibration.confidence_failure,
                        search_probes=config.calibration.search_probes,
                    )
                    bound = calibration.tv_bound
                    probes_used = (
                        config.calibration.search_probes + config.calibration.validation_probes
                    )
                    search_probe_count = config.calibration.search_probes
                    validation_probe_count = config.calibration.validation_probes
                    failure_probability = config.calibration.confidence_failure
                    ratio = config.boundary.death_ratio
                    # The fixed sample-split certificate is for the frozen target at
                    # calibration time. Recalibrate before every attachment so the
                    # recorded bound remains conditional on the current target.
                    stop = min(config.particles, count + 1)
                    death_scale_kind = TARGET_RADIUS_SCALE
                    ratio_scale = ratio
                    ratio_alpha = 0.0
                else:
                    block = schedule.block_at(calibration_index, count, config.particles)
                    ratio = block.diameter_ratio
                    search, steps, restarts = self._probe_with_tree(
                        positions,
                        count,
                        tree,
                        config.particle_radius,
                        pilot,
                        DIAMETER_UPPER_SCALE,
                        ratio,
                        config.boundary.launch_margin,
                        block.probes,
                        config.seed ^ 0x5A17,
                        base_stream,
                        config.walker.tolerance,
                        config.walker.max_steps,
                        config.walker.max_restarts,
                    )
                    total_probe_steps += steps
                    total_probe_restarts += restarts
                    center = empirical_center(search)
                    geometry = _geometry(positions, count, center, capture_distance)
                    # The one-shot bound only covers the frozen target. The
                    # amortized schedule's certified local term also pays for
                    # deterministic center drift across the block.
                    bound = amortized_local_tv_bound(
                        count,
                        alpha=schedule.alpha,
                        gamma=schedule.gamma,
                        scale=schedule.scale,
                        start=schedule.start,
                    )
                    probes_used = block.probes
                    search_probe_count = block.probes
                    failure_probability = block.failure_probability
                    stop = block.stop
                    death_scale_kind = DIAMETER_UPPER_SCALE
                    ratio_scale = schedule.scale
                    ratio_alpha = schedule.alpha

                did_calibrate = True
                center_history.append(center)
                size_history.append(count)
                bound_history.append(bound)
                failure_history.append(failure_probability)
                ratio_history.append(ratio)
                block_index_history.append(calibration_index)
                probe_count_history.append(probes_used)
                search_probe_history.append(search_probe_count)
                validation_probe_history.append(validation_probe_count)
                total_probes += probes_used
                kernel_restart_mode = UNIFORM_RESTART
            else:
                chunk = config.performance.growth_chunk_size
                if chunk <= 0 and config.output.checkpoint_every > 0:
                    chunk = config.output.checkpoint_every
                if chunk > 0:
                    stop = min(config.particles, count + chunk)

            (
                new_count,
                steps,
                restarts,
                status,
                target_radius_value,
                min_x,
                max_x,
                min_y,
                max_y,
            ) = grow_block(
                config.seed,
                positions,
                count,
                stop,
                tree.center_x,
                tree.center_y,
                tree.half_size,
                tree.children,
                tree.counts,
                tree.items,
                tree.node_count,
                center[0],
                center[1],
                config.particle_radius,
                config.boundary.launch_margin,
                death_scale_kind,
                ratio_scale,
                ratio_alpha,
                kernel_restart_mode,
                config.walker.tolerance,
                config.walker.max_steps,
                config.walker.max_restarts,
                geometry.target_radius,
                geometry.min_x,
                geometry.max_x,
                geometry.min_y,
                geometry.max_y,
            )
            total_growth_steps += int(steps)
            total_growth_restarts += int(restarts)
            if status != 0:
                raise SimulationError(
                    f"growth failed after {new_count} particles with kernel status {status}"
                )
            count = int(new_count)
            geometry = _Geometry(target_radius_value, min_x, max_x, min_y, max_y)
            if did_calibrate:
                calibration_index += 1

            if (
                config.output.checkpoint_every > 0
                and count - last_checkpoint >= config.output.checkpoint_every
                and count < config.particles
            ):
                diagnostics = SimulationDiagnostics(
                    growth_walker_steps=total_growth_steps,
                    growth_restarts=total_growth_restarts,
                    calibration_walker_steps=total_probe_steps,
                    calibration_restarts=total_probe_restarts,
                    calibration_probes=total_probes,
                    calibration_centers=np.asarray(center_history, dtype=np.float64).reshape(
                        (-1, 2)
                    ),
                    calibration_sizes=np.asarray(size_history, dtype=np.int64),
                    calibration_bounds=np.asarray(bound_history, dtype=np.float64),
                    calibration_failure_probabilities=np.asarray(
                        failure_history,
                        dtype=np.float64,
                    ),
                    calibration_death_ratios=np.asarray(ratio_history, dtype=np.float64),
                    calibration_block_indices=np.asarray(block_index_history, dtype=np.int64),
                    calibration_probe_counts=np.asarray(probe_count_history, dtype=np.int64),
                    calibration_search_probes=np.asarray(search_probe_history, dtype=np.int64),
                    calibration_validation_probes=np.asarray(
                        validation_probe_history, dtype=np.int64
                    ),
                )
                checkpoint = self._build_result(
                    positions,
                    count,
                    config,
                    diagnostics,
                    started,
                    tree,
                    center,
                    complete=False,
                )
                checkpoint.save(
                    checkpoint_path,
                    overwrite=checkpoint_initialized or config.output.overwrite,
                )
                checkpoint_initialized = True
                last_checkpoint = count

        diagnostics = SimulationDiagnostics(
            growth_walker_steps=total_growth_steps,
            growth_restarts=total_growth_restarts,
            calibration_walker_steps=total_probe_steps,
            calibration_restarts=total_probe_restarts,
            calibration_probes=total_probes,
            calibration_centers=np.asarray(center_history, dtype=np.float64).reshape((-1, 2)),
            calibration_sizes=np.asarray(size_history, dtype=np.int64),
            calibration_bounds=np.asarray(bound_history, dtype=np.float64),
            calibration_failure_probabilities=np.asarray(failure_history, dtype=np.float64),
            calibration_death_ratios=np.asarray(ratio_history, dtype=np.float64),
            calibration_block_indices=np.asarray(block_index_history, dtype=np.int64),
            calibration_probe_counts=np.asarray(probe_count_history, dtype=np.int64),
            calibration_search_probes=np.asarray(search_probe_history, dtype=np.int64),
            calibration_validation_probes=np.asarray(validation_probe_history, dtype=np.int64),
        )
        return self._build_result(
            positions,
            count,
            config,
            diagnostics,
            started,
            tree,
            center,
            complete=True,
        )

    def simulate(self, config: RunConfig) -> SimulationResult:
        """Grow one aggregate while restoring the caller's Numba thread state."""
        previous_threads = get_num_threads()
        try:
            if config.performance.threads > previous_threads:
                raise SimulationError(
                    "performance.threads exceeds the active Numba thread capacity "
                    f"({previous_threads})"
                )
            if config.performance.threads > 0:
                set_num_threads(config.performance.threads)
            return self._simulate(config)
        finally:
            if get_num_threads() != previous_threads:
                set_num_threads(previous_threads)
