from __future__ import annotations

import math

from harmonic_dla.schedules import AmortizedSchedule


def test_schedule_allocates_summable_failures_and_increasing_ratios() -> None:
    schedule = AmortizedSchedule(
        alpha=0.48,
        gamma=0.06,
        scale=2.0,
        calibration_failure_budget=0.01,
        start=100,
    )
    blocks = list(schedule.blocks(300))
    assert blocks
    assert sum(block.failure_probability for block in blocks) < 0.01
    assert all(block.stop > block.start for block in blocks)
    assert all(
        blocks[index + 1].diameter_ratio > blocks[index].diameter_ratio
        for index in range(len(blocks) - 1)
    )
    assert schedule.kappa == 0.25 - 0.06
    assert schedule.diameter_ratio(100) == 2.0 * 100**0.48
    assert math.isfinite(blocks[0].probes)
