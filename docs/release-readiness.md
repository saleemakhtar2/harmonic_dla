# Release readiness

The repository is designed around a committed universal `uv.lock` file.

## Before the first tag

1. Run `uv lock` on a machine with access to PyPI.
2. Run `uv sync --locked` and commit `uv.lock`.
3. Run `make check`, `uv run python scripts/check_reproducibility.py`, and `uv build`.
4. Install the built wheel into a clean environment and run the CLI smoke test.
5. Replace the placeholder author information in `pyproject.toml` and `CITATION.cff`.
6. Record benchmark JSON on the reference machine.
7. Create a signed tag and let the trusted-publishing workflow build and publish from that tag.

The release workflow deliberately fails when `uv.lock` is absent or stale. Normal CI may run
`uv sync` so a new source archive can be bootstrapped before the first lockfile is committed.

## Scientific release gate

A release that changes random-number generation, geometric kernels, restart laws, certificate
formulas, or calibration schedules must include:

- deterministic regression tests with recorded seeds;
- comparison against the reference backend or a direct mathematical invariant;
- matched-accuracy benchmarks against exact Poisson return;
- updated paper-to-code traceability notes;
- a reproducibility manifest for every result used in a manuscript or release note.
