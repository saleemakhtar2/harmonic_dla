# Validation status for the delivered research-alpha repository

Validated in the build container with CPython 3.13.5, NumPy 2.3.5, and Numba 0.65.1:

- 52 non-slow unit and integration tests passed;
- 75.42% branch-aware coverage outside the private compiled geometry and RNG kernels;
- the separate slow Poisson-return test verified complex Fourier modes one through four;
- deterministic replay for fixed seeds;
- deterministic controlled calibration across one and two Numba threads;
- pure-Python reference and Numba CPU smoke simulations;
- CLI config validation, simulation, inspection, certificate evaluation, and warm-up paths;
- byte-code compilation for source, tests, scripts, and benchmarks;
- offline `uv build` wheel and source-distribution construction;
- wheel installation and import/simulation smoke testing in an isolated environment;
- pickle-free NPZ save/load round trip;
- PDF preflight and complete render inspection of the 41-page minor-revision manuscript.

Ruff and ty are configured as mandatory CI gates. They could not be installed or executed inside
this particular preparation container because its package mirror returned repeated HTTP 503
responses. The same outage prevented generation of `uv.lock`. Generate and commit the universal
lockfile before a tag; the release workflow enforces this requirement.

This is a research-alpha codebase. It is production-structured and defensively validated, but the
paper's preregistered large-ensemble numerical study and matched-accuracy performance campaign
remain future work rather than bundled empirical claims.
