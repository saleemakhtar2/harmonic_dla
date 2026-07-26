"""Backend protocol."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

from harmonic_dla.config import RunConfig
from harmonic_dla.enums import RestartMode
from harmonic_dla.models import ProbeResult, SimulationResult

FloatArray = npt.NDArray[np.float64]


class Backend(Protocol):
    """Simulation backend contract."""

    def simulate(self, config: RunConfig) -> SimulationResult:
        """Grow and return one aggregate."""
        ...

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
        """Draw frozen-cluster attachment probes."""
        ...

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
        """Draw probes and return walker diagnostics."""
        ...
