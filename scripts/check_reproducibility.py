"""CI check for deterministic growth and parallel frozen-probe calibration."""

from __future__ import annotations

import numpy as np

from harmonic_dla.api import simulate
from harmonic_dla.config import (
    BoundaryConfig,
    CalibrationConfig,
    PerformanceConfig,
    RunConfig,
)
from harmonic_dla.enums import RestartMode


def _controlled_config(threads: int) -> RunConfig:
    return RunConfig(
        particles=48,
        seed=1701,
        boundary=BoundaryConfig(
            restart_mode=RestartMode.CONTROLLED_RESTART,
            death_ratio=3.0,
            launch_margin=3.0,
        ),
        calibration=CalibrationConfig(
            enabled=True,
            search_probes=64,
            validation_probes=64,
            block_size=16,
            confidence_failure=0.01,
        ),
        performance=PerformanceConfig(threads=threads),
    )


def main() -> None:
    first = simulate(_controlled_config(threads=1))
    second = simulate(_controlled_config(threads=2))
    np.testing.assert_array_equal(first.positions, second.positions)
    np.testing.assert_array_equal(
        first.diagnostics.calibration_centers,
        second.diagnostics.calibration_centers,
    )
    np.testing.assert_array_equal(
        first.diagnostics.calibration_bounds,
        second.diagnostics.calibration_bounds,
    )
    print("deterministic aggregate and probe calibration reproduced across thread counts")


if __name__ == "__main__":
    main()
