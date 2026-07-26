"""Accuracy-controlled planar diffusion-limited aggregation."""

from importlib.metadata import PackageNotFoundError, version

from harmonic_dla.api import probe, simulate
from harmonic_dla.certificates import (
    amortized_path_budget_upper,
    constant_ratio_for_path_budget,
    monte_carlo_tv_bound,
    one_shot_diameter_tv_bound,
    path_budget,
    residual_tv_bound,
    self_centered_tv_bound,
)
from harmonic_dla.config import RunConfig, load_config
from harmonic_dla.io import load_result
from harmonic_dla.models import SimulationResult

try:
    __version__ = version("harmonic-dla")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"

__all__ = [
    "RunConfig",
    "SimulationResult",
    "amortized_path_budget_upper",
    "constant_ratio_for_path_budget",
    "load_config",
    "load_result",
    "monte_carlo_tv_bound",
    "one_shot_diameter_tv_bound",
    "path_budget",
    "probe",
    "residual_tv_bound",
    "self_centered_tv_bound",
    "simulate",
]
