"""Deliberately simple reference implementation for tests and small runs."""

from __future__ import annotations

import math
import time

import numpy as np
import numpy.typing as npt

from harmonic_dla.boundaries import poisson_return_delta
from harmonic_dla.calibration import empirical_center, validate_center
from harmonic_dla.certificates import one_shot_diameter_tv_bound
from harmonic_dla.config import RunConfig, config_to_dict
from harmonic_dla.enums import CalibrationStrategy, RestartMode
from harmonic_dla.exceptions import SimulationError
from harmonic_dla.models import SimulationDiagnostics, SimulationResult
from harmonic_dla.provenance import runtime_provenance
from harmonic_dla.schedules import AmortizedSchedule

FloatArray = npt.NDArray[np.float64]


def _stream_rng(master_seed: int, stream: int) -> np.random.Generator:
    sequence = np.random.SeedSequence([master_seed, stream])
    return np.random.default_rng(sequence)


def _geometry(
    positions: FloatArray,
    center: tuple[float, float],
    capture_distance: float,
) -> tuple[float, float]:
    offsets = positions - np.asarray(center, dtype=np.float64)
    radius = float(np.sqrt(np.max(np.sum(offsets * offsets, axis=1)))) + capture_distance
    span = np.max(positions, axis=0) - np.min(positions, axis=0) + 2.0 * capture_distance
    diameter_upper = float(math.hypot(float(span[0]), float(span[1])))
    return radius, diameter_upper


def _nearest(
    x: float,
    y: float,
    positions: FloatArray,
) -> tuple[int, float]:
    delta = positions - np.asarray((x, y), dtype=np.float64)
    squared = np.sum(delta * delta, axis=1)
    index = int(np.argmin(squared))
    return index, float(math.sqrt(float(squared[index])))


def _walk_attachment(
    master_seed: int,
    stream: int,
    positions: FloatArray,
    particle_radius: float,
    center: tuple[float, float],
    birth_radius: float,
    death_radius: float,
    restart_mode: RestartMode,
    tolerance: float,
    max_steps: int,
    max_restarts: int,
) -> tuple[tuple[float, float], int, int]:
    rng = _stream_rng(master_seed, stream)
    capture_distance = 2.0 * particle_radius
    angle = 2.0 * math.pi * float(rng.random())
    x = center[0] + birth_radius * math.cos(angle)
    y = center[1] + birth_radius * math.sin(angle)
    restarts = 0

    for step in range(1, max_steps + 1):
        nearest_index, nearest_distance = _nearest(x, y, positions)
        target_gap = nearest_distance - capture_distance
        radial_x = x - center[0]
        radial_y = y - center[1]
        radial_distance = math.hypot(radial_x, radial_y)
        death_gap = death_radius - radial_distance

        if target_gap <= tolerance:
            if nearest_distance <= 0.0:
                raise SimulationError("reference walker reached a particle center")
            scale = capture_distance / nearest_distance
            x = positions[nearest_index, 0] + (x - positions[nearest_index, 0]) * scale
            y = positions[nearest_index, 1] + (y - positions[nearest_index, 1]) * scale
            return (float(x), float(y)), step, restarts

        if death_gap <= tolerance:
            if restarts >= max_restarts:
                raise SimulationError("reference walker exceeded max_restarts")
            exit_angle = math.atan2(radial_y, radial_x)
            if restart_mode is RestartMode.EXACT_RETURN:
                uniform = np.asarray([float(rng.random())], dtype=np.float64)
                angle = exit_angle + float(
                    poisson_return_delta(birth_radius / death_radius, uniform)[0]
                )
            else:
                angle = 2.0 * math.pi * float(rng.random())
            x = center[0] + birth_radius * math.cos(angle)
            y = center[1] + birth_radius * math.sin(angle)
            restarts += 1
            continue

        sphere_radius = min(target_gap, death_gap)
        if sphere_radius <= 0.0 or not math.isfinite(sphere_radius):
            raise SimulationError("reference walker encountered invalid WOS radius")
        angle = 2.0 * math.pi * float(rng.random())
        x += sphere_radius * math.cos(angle)
        y += sphere_radius * math.sin(angle)

    raise SimulationError("reference walker exceeded max_steps")


def _pilot(
    policy: str,
    positions: FloatArray,
    previous: tuple[float, float],
    calibrated: bool,
) -> tuple[float, float]:
    if policy == "seed" or (policy == "previous-center" and not calibrated):
        return float(positions[0, 0]), float(positions[0, 1])
    if policy == "mass-centroid":
        mean = np.mean(positions, axis=0)
        return float(mean[0]), float(mean[1])
    return previous


class ReferenceBackend:
    """Brute-force implementation used as an executable specification."""

    name = "reference"

    def _probe_scaled(
        self,
        positions: FloatArray,
        particle_radius: float,
        center: tuple[float, float],
        ratio: float,
        launch_margin: float,
        probes: int,
        seed: int,
        stream_offset: int,
        *,
        diameter_scaled: bool,
        tolerance: float,
        max_steps: int,
        max_restarts: int,
    ) -> tuple[FloatArray, int, int]:
        radius, diameter_upper = _geometry(positions, center, 2.0 * particle_radius)
        birth_radius = radius + launch_margin * particle_radius
        death_radius = ratio * (diameter_upper if diameter_scaled else radius)
        death_radius = max(death_radius, birth_radius * (1.0 + 1.0e-12))
        attachments = np.empty((probes, 2), dtype=np.float64)
        steps_total = 0
        restarts_total = 0
        for index in range(probes):
            attachment, steps, restarts = _walk_attachment(
                seed,
                stream_offset + index,
                positions,
                particle_radius,
                center,
                birth_radius,
                death_radius,
                RestartMode.UNIFORM_RESTART,
                tolerance,
                max_steps,
                max_restarts,
            )
            attachments[index] = attachment
            steps_total += steps
            restarts_total += restarts
        return attachments, steps_total, restarts_total

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
        """Draw target-radius-scaled uniform-restart probes."""
        attachments, _steps, _restarts = self._probe_scaled(
            np.ascontiguousarray(positions, dtype=np.float64),
            particle_radius,
            center,
            death_ratio,
            launch_margin,
            probes,
            seed,
            1 << 40,
            diameter_scaled=False,
            tolerance=1.0e-6,
            max_steps=100_000,
            max_restarts=100_000,
        )
        return attachments

    def simulate(self, config: RunConfig) -> SimulationResult:
        """Grow a small aggregate using brute-force nearest-neighbour queries."""
        started = time.perf_counter()
        positions = np.empty((config.particles, 2), dtype=np.float64)
        positions[0] = (0.0, 0.0)
        count = 1
        center = config.boundary.center
        growth_steps = 0
        growth_restarts = 0
        probe_steps = 0
        probe_restarts = 0
        probe_count = 0
        centers: list[tuple[float, float]] = []
        sizes: list[int] = []
        bounds: list[float] = []
        failures: list[float] = []
        ratios: list[float] = []
        calibration_index = 0

        schedule: AmortizedSchedule | None = None
        if (
            config.boundary.restart_mode is RestartMode.CONTROLLED_RESTART
            and config.calibration.strategy is CalibrationStrategy.PAPER_AMORTIZED
        ):
            schedule = AmortizedSchedule(
                config.calibration.alpha,
                config.calibration.gamma,
                config.calibration.scale,
                config.calibration.failure_budget,
                config.calibration.start,
            )

        while count < config.particles:
            restart_mode = config.boundary.restart_mode
            ratio = config.boundary.death_ratio
            diameter_scaled = False
            ratio_alpha = 0.0
            stop = config.particles

            if schedule is not None and count < schedule.start:
                stop = min(config.particles, schedule.start)
                restart_mode = (
                    RestartMode.EXACT_RETURN
                    if config.calibration.exact_prefix
                    else RestartMode.UNIFORM_RESTART
                )
            elif config.boundary.restart_mode is RestartMode.CONTROLLED_RESTART:
                pilot = _pilot(config.calibration.pilot, positions[:count], center, bool(centers))
                stream = (1 << 40) + calibration_index * (1 << 30)
                if schedule is None:
                    search, steps, restarts = self._probe_scaled(
                        positions[:count],
                        config.particle_radius,
                        pilot,
                        ratio,
                        config.boundary.launch_margin,
                        config.calibration.search_probes,
                        config.seed ^ 0x5A17,
                        stream,
                        diameter_scaled=False,
                        tolerance=config.walker.tolerance,
                        max_steps=config.walker.max_steps,
                        max_restarts=config.walker.max_restarts,
                    )
                    probe_steps += steps
                    probe_restarts += restarts
                    center = empirical_center(search)
                    validation, steps, restarts = self._probe_scaled(
                        positions[:count],
                        config.particle_radius,
                        center,
                        ratio,
                        config.boundary.launch_margin,
                        config.calibration.validation_probes,
                        config.seed ^ 0xA51C,
                        stream + config.calibration.search_probes,
                        diameter_scaled=False,
                        tolerance=config.walker.tolerance,
                        max_steps=config.walker.max_steps,
                        max_restarts=config.walker.max_restarts,
                    )
                    probe_steps += steps
                    probe_restarts += restarts
                    target_radius, _diameter = _geometry(
                        positions[:count],
                        center,
                        2.0 * config.particle_radius,
                    )
                    calibration = validate_center(
                        center,
                        validation,
                        target_radius,
                        ratio,
                        config.calibration.confidence_failure,
                        search_probes=config.calibration.search_probes,
                    )
                    bound = calibration.tv_bound
                    failure = config.calibration.confidence_failure
                    used = config.calibration.search_probes + config.calibration.validation_probes
                    stop = min(config.particles, count + config.calibration.block_size)
                else:
                    block = schedule.block_at(calibration_index, count, config.particles)
                    ratio = block.diameter_ratio
                    diameter_scaled = True
                    ratio_alpha = schedule.alpha
                    search, steps, restarts = self._probe_scaled(
                        positions[:count],
                        config.particle_radius,
                        pilot,
                        ratio,
                        config.boundary.launch_margin,
                        block.probes,
                        config.seed ^ 0x5A17,
                        stream,
                        diameter_scaled=True,
                        tolerance=config.walker.tolerance,
                        max_steps=config.walker.max_steps,
                        max_restarts=config.walker.max_restarts,
                    )
                    probe_steps += steps
                    probe_restarts += restarts
                    center = empirical_center(search)
                    bound = one_shot_diameter_tv_bound(
                        ratio,
                        block.probes,
                        block.failure_probability,
                    )
                    failure = block.failure_probability
                    used = block.probes
                    stop = block.stop
                centers.append(center)
                sizes.append(count)
                bounds.append(bound)
                failures.append(failure)
                ratios.append(ratio)
                probe_count += used
                restart_mode = RestartMode.UNIFORM_RESTART
                calibration_index += 1

            while count < stop:
                radius, diameter_upper = _geometry(
                    positions[:count],
                    center,
                    2.0 * config.particle_radius,
                )
                current_ratio = (
                    ratio
                    if ratio_alpha == 0.0
                    else config.calibration.scale * count**ratio_alpha
                )
                birth_radius = radius + config.boundary.launch_margin * config.particle_radius
                death_radius = current_ratio * (diameter_upper if diameter_scaled else radius)
                death_radius = max(death_radius, birth_radius * (1.0 + 1.0e-12))
                attachment, steps, restarts = _walk_attachment(
                    config.seed,
                    count,
                    positions[:count],
                    config.particle_radius,
                    center,
                    birth_radius,
                    death_radius,
                    restart_mode,
                    config.walker.tolerance,
                    config.walker.max_steps,
                    config.walker.max_restarts,
                )
                positions[count] = attachment
                count += 1
                growth_steps += steps
                growth_restarts += restarts

        diagnostics = SimulationDiagnostics(
            growth_walker_steps=growth_steps,
            growth_restarts=growth_restarts,
            calibration_walker_steps=probe_steps,
            calibration_restarts=probe_restarts,
            calibration_probes=probe_count,
            calibration_centers=np.asarray(centers, dtype=np.float64).reshape((-1, 2)),
            calibration_sizes=np.asarray(sizes, dtype=np.int64),
            calibration_bounds=np.asarray(bounds, dtype=np.float64),
            calibration_failure_probabilities=np.asarray(failures, dtype=np.float64),
            calibration_death_ratios=np.asarray(ratios, dtype=np.float64),
        )
        metadata = runtime_provenance()
        metadata.update(
            {
                "schema_version": 1,
                "complete": True,
                "elapsed_seconds": time.perf_counter() - started,
                "numerical_model": "reference walk-on-spheres with brute-force nearest queries",
                "config": config_to_dict(config),
            }
        )
        return SimulationResult(
            positions=positions,
            particle_radius=config.particle_radius,
            seed=config.seed,
            backend=self.name,
            restart_mode=config.boundary.restart_mode.name.lower(),
            diagnostics=diagnostics,
            metadata=metadata,
        )
