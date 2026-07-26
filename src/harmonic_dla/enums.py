"""Enumerations shared by public configuration and internal backends."""

from enum import IntEnum, StrEnum


class BackendKind(StrEnum):
    """Available simulation backends."""

    NUMBA_CPU = "numba-cpu"
    REFERENCE = "reference"


class RestartMode(IntEnum):
    """Boundary treatment after a walker reaches the death circle."""

    EXACT_RETURN = 0
    UNIFORM_RESTART = 1
    CONTROLLED_RESTART = 2

    @classmethod
    def from_text(cls, value: str) -> "RestartMode":
        """Parse a stable command-line or TOML representation."""
        normalized = value.strip().lower().replace("_", "-")
        mapping = {
            "exact": cls.EXACT_RETURN,
            "exact-return": cls.EXACT_RETURN,
            "uniform": cls.UNIFORM_RESTART,
            "uniform-restart": cls.UNIFORM_RESTART,
            "controlled": cls.CONTROLLED_RESTART,
            "controlled-restart": cls.CONTROLLED_RESTART,
        }
        try:
            return mapping[normalized]
        except KeyError as exc:
            allowed = ", ".join(sorted(mapping))
            raise ValueError(f"unknown restart mode {value!r}; choose one of: {allowed}") from exc


class CalibrationStrategy(StrEnum):
    """How a controlled-restart center is estimated and reused."""

    SAMPLE_SPLIT_FIXED = "sample-split-fixed"
    PAPER_AMORTIZED = "paper-amortized"
