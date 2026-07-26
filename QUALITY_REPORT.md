# Quality report for the delivered repository

## Executed in the preparation container

- CPython 3.13.5, NumPy 2.3.5, Numba 0.65.1.
- 52 non-slow unit and integration tests passed.
- 75.42% branch-aware coverage outside the private compiled geometry/RNG kernels, above the
  configured 70% gate.
- The stronger Poisson-return mode test passed separately under the `slow` marker.
- Deterministic aggregate replay passed for fixed seeds.
- Controlled calibration reproduced exactly across one and two Numba threads.
- The pure-Python reference backend and Numba CPU backend both passed smoke simulations.
- CLI configuration, run, inspect, certificate, and warm-up paths were exercised.
- Source, tests, benchmarks, and scripts passed byte-code compilation.
- A wheel and source distribution were built with `uv build --offline --no-build-isolation`;
  the wheel was installed into a clean environment and passed an import/simulation smoke test.
- The final 41-page manuscript passed PDF preflight and full-page rendering inspection.

## Tooling status

Ruff and ty are configured as mandatory CI gates in `pyproject.toml`. The preparation container's
Python package mirror returned repeated HTTP 503 responses, so those binaries could not be fetched
and a trustworthy `uv.lock` could not be generated. The repository does not fabricate a lockfile.

Before the first release, generate and commit `uv.lock`, run Ruff and ty, resolve every diagnostic,
and execute the matrix CI in a network-connected checkout. The tag workflow fails closed if the
lockfile is absent or stale.

## Scientific claim boundary

This is a production-structured research alpha, not a completed empirical validation campaign. The
exact-return implementation is the correctness baseline. The controlled modes implement the paper's
formulas and random-stream separation, but no claim of matched-accuracy speed superiority is made
until the preregistered ensemble and benchmark protocol has been completed.
