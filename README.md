# harmonic-dla

`harmonic-dla` is the reproducible software companion to
**Self-Consistent and Amortized Centering for Finite-Boundary Restart in Planar DLA**.
It provides a Python-facing, Numba-compiled implementation of two-dimensional off-lattice
Diffusion-Limited Aggregation together with the paper's finite-boundary error formulas,
calibration schedules, and path-budget utilities.

The repository is production-structured but remains a **research alpha** until the paper's
predeclared ensemble validation and matched-accuracy benchmark campaign has been completed.
It makes no claim that controlled restart is faster than exact Poisson return for circular
boundaries.

## Implemented modes

- **Exact Poisson return** - the exact circular-boundary return baseline; finite
  walk-on-spheres tolerance remains a separately reported numerical approximation.
- **Uniform restart** - the historical approximation, exposed explicitly rather than silently.
- **Sample-split controlled restart** - search probes select a center and an independent
  validation batch supplies the residual certificate.
- **Paper-amortized controlled restart** - exact prefix, blockwise recentering with the
  theorem's stale-center drift budget, diameter-scaled death radii, and the paper's
  probe/block schedule.

The package also includes:

- a compiled Numba CPU engine with exact nearest-neighbour queries through a packed quadtree;
- parallel frozen-cluster probes with deterministic per-walker random streams;
- a deliberately simple pure-Python reference backend;
- versioned, pickle-free NPZ persistence and complete run provenance;
- a CLI, tests, benchmark scaffolding, CI, release automation, and paper-to-code traceability.

## Why Numba rather than JAX

DLA growth is sequential, branch-heavy, and mutates an irregular spatial index after every
attachment. Numba maps that imperative workload directly to native CPU code and can parallelize
independent frozen-target probe walkers with `prange`. JAX is intentionally not a core dependency.
A future JAX backend may be useful for shape-static accelerator probe batches, but only behind the
backend protocol and only after law-parity and matched-accuracy benchmarks.

## Quick start

```bash
uv sync
uv run hdla warmup
uv run hdla run configs/exact.toml
uv run hdla inspect outputs/exact-cluster.npz
```

The first successful `uv sync` creates `uv.lock`. Commit that universal lockfile before the first
tag, then use `uv sync --locked` for release and archival reproduction.

Python API:

```python
from harmonic_dla import load_config, simulate

config = load_config("configs/exact.toml")
result = simulate(config)
result.save("outputs/cluster.npz", overwrite=True)
```

Frozen-cluster probes:

```python
from harmonic_dla import probe

attachments = probe(
    result.positions,
    particle_radius=result.particle_radius,
    death_ratio=4.0,
    probes=100_000,
    seed=123,
)
```

Certificate CLI:

```bash
uv run hdla certificate self-centered --rho 4
uv run hdla certificate monte-carlo \
  --rho 4 \
  --residual-over-radius 0.01 \
  --probes 10000 \
  --failure-probability 0.001
```

## Quality gates

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest -m "not slow" --cov=harmonic_dla
uv run pytest -m slow
uv build
```

Every compiled-kernel change must be checked against the reference backend or a direct
mathematical invariant. Performance changes require before/after benchmark JSON from the same
machine and comparisons at matched accuracy, not merely equal particle count.

## Repository guide

- [`REPO_SPEC.md`](REPO_SPEC.md) - scope, contracts, public interfaces, and release gates.
- [`docs/architecture.md`](docs/architecture.md) - control/data-plane design and backend decision.
- [`docs/paper-mapping.md`](docs/paper-mapping.md) - theorem and algorithm mapping to code.
- [`docs/validation-protocol.md`](docs/validation-protocol.md) - predeclared scientific validation.
- [`docs/performance.md`](docs/performance.md) - matched-accuracy benchmarking rules.
- [`docs/release-readiness.md`](docs/release-readiness.md) - first-tag checklist.
- [`paper/`](paper/) - the final minor-revision manuscript and reproducible figure sources.

The delivered validation status and known release blockers are recorded in
[`QUALITY_REPORT.md`](QUALITY_REPORT.md).
