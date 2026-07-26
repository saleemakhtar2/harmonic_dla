# Repository specification

## Scope of version 0.1

Version 0.1 is a production-quality research alpha. It must be capable of reproducing the
paper's transition kernels and schedules, generating exact-return controls, saving sufficient
provenance, and running a predeclared validation matrix. It is not permitted to make a
wall-clock superiority claim without matched-accuracy benchmarks.

## Public interfaces

### Python

- `simulate(RunConfig | path) -> SimulationResult`
- `probe(positions, ...) -> ndarray[(n, 2), float64]`
- certificate functions in `harmonic_dla.certificates`
- `load_result(path) -> SimulationResult`

Public signatures follow semantic versioning after 1.0. Compiled kernel signatures are private.

### Command line

- `hdla run CONFIG`
- `hdla inspect RESULT`
- `hdla check-config CONFIG`
- `hdla certificate ...`
- `hdla warmup`

All commands must return nonzero status on invalid input or incomplete simulation.

## Correctness requirements

- Exact nearest-neighbour queries; approximate spatial searches are forbidden in the reference
  scientific mode.
- New particle centers are projected to exact contact and must not overlap existing particles.
- Exact-return angular samples must pass Fourier-moment tests.
- Uniform-restart calibration samples must be independent of subsequent growth streams.
- Sample-split certificates must use a validation batch not used to select the center.
- Paper-amortized runs must record block size, probe count, failure allocation, and boundary
  ratio for every calibration.
- Any upper-bound geometry substituted for the exact target diameter must be documented and
  conservative.

## Performance requirements

- No Python callback inside a growth walker.
- No full aggregate scan per attached particle for fixed-center runs.
- No allocation per nearest-neighbour query in the compiled hot path.
- Sequential growth and parallel frozen probes must use separate optimization strategies.
- First-use compilation must be excluded from reported steady-state benchmarks.
- Peak memory must scale linearly with the requested particle capacity.

## Reproducibility requirements

Every result records package, Python, NumPy and Numba versions; platform; seed; thread count;
complete validated configuration; numerical tolerance; boundary policy; and calibration
history. Saved files are versioned and pickle-free.

## CI acceptance gates

- Ruff lint and format checks.
- ty type checking.
- deterministic unit and integration tests on supported Python versions.
- statistical tests with fixed seeds and tolerances chosen before execution.
- wheel and source-distribution build.
- separate scheduled benchmarks so timing variance cannot block correctness changes.

## Release process

1. Update `CHANGELOG.md` and `CITATION.cff`.
2. Run the complete validation and benchmark suites on a named machine.
3. Archive raw benchmark and validation outputs.
4. Build with `uv build` from the locked environment.
5. Publish through trusted publishing from a signed `vX.Y.Z` tag.

## Planned extensions

- Resume from block-boundary checkpoints.
- Ensemble runner with process-level parallelism.
- Exact convex-hull diameter tracker for theorem-tight experiments.
- Continuum-sensitive harmonic-measure comparison tools.
- Optional accelerator backend for frozen probes only, after parity tests.
