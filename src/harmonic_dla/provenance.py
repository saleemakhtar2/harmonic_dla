"""Runtime provenance recorded with every result."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numba
import numpy as np


def _source_revision() -> str | None:
    """Resolve a source revision from CI metadata or a nearby Git checkout."""
    for name in (
        "GITHUB_SHA",
        "HDLA_SOURCE_REVISION",
        "HDLA_GIT_COMMIT",
        "CI_COMMIT_SHA",
        "BUILD_VCS_NUMBER",
        "VCS_REF",
        "GIT_COMMIT",
    ):
        value = os.environ.get(name)
        if value:
            return value

    # Source distributions and installed wheels need not have a .git directory;
    # probing is therefore best-effort and intentionally silent on failure.
    try:
        root = Path(__file__).resolve().parents[2]
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and revision else None


def runtime_provenance() -> dict[str, Any]:
    """Return stable environment fields useful for reproduction."""
    try:
        package_version = version("harmonic-dla")
    except PackageNotFoundError:
        package_version = "0+unknown"
    source_revision = _source_revision()
    return {
        "harmonic_dla_version": package_version,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": np.__version__,
        "numba_version": numba.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        # Keep ``git_commit`` as a compatibility alias while exposing a
        # source-neutral name for archives generated outside GitHub Actions.
        "git_commit": source_revision,
        "source_revision": source_revision,
    }
