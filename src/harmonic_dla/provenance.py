"""Runtime provenance recorded with every result."""

from __future__ import annotations

import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numba
import numpy as np


def runtime_provenance() -> dict[str, Any]:
    """Return stable environment fields useful for reproduction."""
    try:
        package_version = version("harmonic-dla")
    except PackageNotFoundError:
        package_version = "0+unknown"
    return {
        "harmonic_dla_version": package_version,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": np.__version__,
        "numba_version": numba.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "git_commit": os.environ.get("GITHUB_SHA") or os.environ.get("HDLA_GIT_COMMIT"),
    }
