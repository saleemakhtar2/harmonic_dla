"""Package-specific exceptions."""


class HarmonicDLAError(Exception):
    """Base class for package errors."""


class ConfigurationError(HarmonicDLAError, ValueError):
    """Raised when a configuration is internally inconsistent."""


class SimulationError(HarmonicDLAError, RuntimeError):
    """Raised when a walker or spatial index cannot complete safely."""


class CheckpointError(HarmonicDLAError, OSError):
    """Raised when checkpoint serialization or restoration fails."""
