# ADR 0001: Numba is the primary compiled backend

**Status:** accepted

## Context

The simulator has a sequentially changing target, branch-heavy walk-on-spheres loops, a mutable
spatial index, and independent fixed-target probe batches.

## Decision

Use Numba nopython kernels for CPU growth and `prange` for frozen probes. Keep a backend
protocol so an accelerator implementation can be added later. Do not make JAX a core
runtime dependency.

## Consequences

- Native hot loops without a C++ extension build toolchain.
- Deterministic package-owned RNG streams are required for parallel reproducibility.
- Public objects and I/O stay in ordinary typed Python.
- GPU support is deferred; a future JAX or CUDA backend must reproduce the same statistical
  tests and result schema.
