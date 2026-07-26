"""Boundary and calibration schedules derived from the paper."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Block:
    """One amortized calibration block."""

    index: int
    start: int
    stop: int
    probes: int
    failure_probability: float
    diameter_ratio: float


@dataclass(frozen=True, slots=True)
class AmortizedSchedule:
    """The paper's area-only diameter-scaled block and probe schedule."""

    alpha: float
    gamma: float
    scale: float
    calibration_failure_budget: float
    start: int

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (self.alpha, self.gamma, self.scale, self.calibration_failure_budget)
        ):
            raise ValueError("schedule parameters must be finite")
        if self.alpha <= 11.0 / 24.0:
            raise ValueError("alpha must exceed 11/24 for the area-only theorem")
        lower = max(0.0, 1.0 - 2.0 * self.alpha)
        if not lower < self.gamma < 1.0 / 12.0:
            raise ValueError("gamma must lie in (max(0, 1-2alpha), 1/12)")
        if self.scale < math.sqrt(2.0):
            raise ValueError("scale must be at least sqrt(2)")
        if not 0.0 < self.calibration_failure_budget < 1.0:
            raise ValueError("calibration_failure_budget must lie in (0, 1)")
        minimum_start = max(16, math.ceil(2.0 ** (1.0 / self.kappa)))
        if self.start < minimum_start:
            raise ValueError(
                f"start must be at least {minimum_start} for the equal-disk theorem constants"
            )

    @property
    def kappa(self) -> float:
        """Block-length exponent ``1/4 - gamma``."""
        return 0.25 - self.gamma

    def diameter_ratio(self, particle_count: int) -> float:
        """Return ``varrho_n = L n**alpha``."""
        if particle_count < 1:
            raise ValueError("particle_count must be positive")
        return self.scale * particle_count**self.alpha

    def failure_probability(self, block_index: int) -> float:
        """Summable block failure allocation from the paper."""
        if block_index < 0:
            raise ValueError("block_index cannot be negative")
        return (
            6.0
            * self.calibration_failure_budget
            / (math.pi * math.pi * (block_index + 1) * (block_index + 1))
        )

    def block_at(self, block_index: int, start: int, horizon: int) -> Block:
        """Construct one block beginning at ``start``."""
        if horizon <= start:
            raise ValueError("horizon must exceed start")
        length = max(1, int(start**self.kappa))
        stop = min(horizon, start + length)
        failure = self.failure_probability(block_index)
        probes = math.ceil(4.0 * start ** (2.0 * self.gamma) * math.log(4.0 / failure))
        return Block(
            index=block_index,
            start=start,
            stop=stop,
            probes=probes,
            failure_probability=failure,
            diameter_ratio=self.diameter_ratio(start),
        )

    def blocks(self, horizon: int) -> Iterator[Block]:
        """Yield all calibration blocks intersecting ``[start, horizon)``."""
        if horizon <= self.start:
            return
        block_index = 0
        start = self.start
        while start < horizon:
            block = self.block_at(block_index, start, horizon)
            yield block
            start = block.stop
            block_index += 1
