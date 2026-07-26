from __future__ import annotations

from pathlib import Path

import pytest

from harmonic_dla.config import CalibrationConfig, RunConfig, load_config
from harmonic_dla.enums import CalibrationStrategy, RestartMode
from harmonic_dla.exceptions import ConfigurationError


def test_controlled_restart_requires_calibration() -> None:
    from harmonic_dla.config import BoundaryConfig

    with pytest.raises(ConfigurationError):
        RunConfig(boundary=BoundaryConfig(restart_mode=RestartMode.CONTROLLED_RESTART))


def test_amortized_parameter_region() -> None:
    config = CalibrationConfig(
        enabled=True,
        strategy=CalibrationStrategy.PAPER_AMORTIZED,
        alpha=0.48,
        gamma=0.06,
        scale=2.0,
    )
    assert config.strategy is CalibrationStrategy.PAPER_AMORTIZED


def test_load_config(tmp_path: Path) -> None:
    path = tmp_path / "run.toml"
    path.write_text(
        """
particles = 32
seed = 7
backend = "reference"

[boundary]
restart_mode = "exact-return"
death_ratio = 3.0
launch_margin = 2.0

[performance]
growth_chunk_size = 8

[output]
path = "result.npz"
""".strip()
    )
    config = load_config(path)
    assert config.particles == 32
    assert config.seed == 7
    assert config.output.path == Path("result.npz")
    assert config.performance.growth_chunk_size == 8


@pytest.mark.parametrize(
    ("toml", "needle"),
    [
        ("unknown = 1", "unknown"),
        ("[walker]\nunknown = 1", "walker"),
        ("[output]\nunknown = 1", "output"),
    ],
)
def test_load_config_rejects_unknown_keys(tmp_path: Path, toml: str, needle: str) -> None:
    path = tmp_path / "run.toml"
    path.write_text(toml)
    with pytest.raises(ConfigurationError, match=needle):
        load_config(path)


@pytest.mark.parametrize(
    "toml",
    [
        "seed = 1.5",
        "particles = 2.5",
        '[calibration]\nenabled = "false"',
        '[output]\noverwrite = "false"',
    ],
)
def test_load_config_rejects_wrong_types(tmp_path: Path, toml: str) -> None:
    path = tmp_path / "run.toml"
    path.write_text(toml)
    with pytest.raises(ConfigurationError):
        load_config(path)


@pytest.mark.parametrize("seed", [-1, 2**63, -(2**63) - 1])
def test_run_config_rejects_seed_outside_nonnegative_int64(seed: int) -> None:
    with pytest.raises(ConfigurationError):
        RunConfig(seed=seed)


def test_run_config_rejects_fractional_particles() -> None:
    with pytest.raises(ConfigurationError):
        RunConfig(particles=2.5)  # type: ignore[arg-type]


def test_paper_amortized_requires_exact_prefix() -> None:
    with pytest.raises(ConfigurationError, match="exact_prefix"):
        CalibrationConfig(
            enabled=True,
            strategy=CalibrationStrategy.PAPER_AMORTIZED,
            exact_prefix=False,
        )


def test_run_config_rejects_unreasonable_particle_capacity() -> None:
    with pytest.raises(ConfigurationError, match=r"memory|particles"):
        RunConfig(particles=10**12)
