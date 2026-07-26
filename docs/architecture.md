# Architecture

## Design goals

1. **Scientific correctness before throughput.** Exact Poisson return is the baseline. Every
   approximation must expose its assumptions and certificate.
2. **Compiled hot path.** Aggregate growth, nearest-neighbour queries, walk-on-spheres steps,
   and probe batches run in Numba nopython mode.
3. **Deterministic reproducibility.** Each walker receives a stream derived from
   `(master_seed, stream_id)` using a package-owned xoshiro256** generator. Results do not
   depend on Numba thread scheduling.
4. **Reference implementation.** A brute-force Python backend acts as an executable
   specification for small tests.
5. **No unsafe persistence.** Result archives use `numpy.savez_compressed` with
   `allow_pickle=False` on load and an explicit schema version.

## Layers

```text
CLI / public API
      |
validated dataclass configuration
      |
backend protocol
  +---+-------------------+
  |                       |
Numba CPU             Python reference
  |
compiled WOS + packed quadtree + deterministic RNG
  |
versioned result model and diagnostics
```

## Why Numba is the primary engine

DLA growth is sequential: the next attachment changes the target seen by the next walker.
The hot path also uses data-dependent loops, a mutable packed quadtree, and branch-heavy
boundary logic. Numba maps this structure directly to native CPU code and can parallelize
independent frozen probes with `prange`.

JAX is deliberately not a core dependency. It is excellent for large, shape-static array
programs, but this workload has dynamic control flow, mutable spatial indexing, and one
structural update per particle. A future experimental JAX backend may target fixed-cluster
probe batches on accelerators, behind the backend protocol, without changing the public API.

## Numerical model

The new particle center performs walk-on-spheres in the domain outside capture disks of
radius `2 * particle_radius` around existing centers and inside a circular death boundary.
When the remaining gap to the target is below the configured tolerance, the point is projected
to exact contact with the nearest capture disk. The tolerance therefore controls angular
first-passage error while maintaining a connected, non-overlapping particle aggregate.

## Spatial index

The Numba backend stores a quadtree in flat NumPy arrays. Nearest queries are exact, not
approximate. Caller-owned stack workspaces avoid allocation at every walk-on-spheres step;
a brute-force fallback preserves correctness if the traversal stack would overflow.

The tree is sized before growth. Exhaustion is a hard error, never a silent fallback that could
change the stochastic law.

## Parallelism

Growth remains sequential. Frozen-cluster calibration probes are independent and run through
Numba `prange`. The custom per-walker RNG makes their output deterministic across thread
counts. Process-level parallelism should be used for independent aggregate ensembles.

## Certificate modes

- `exact-return`: exact circular-boundary return law. The configured walk-on-spheres
  tolerance is still a numerical approximation and is not included in the analytical
  restart certificate.
- `uniform-restart`: legacy approximation, no automatic certification.
- `controlled-restart` + `sample-split-fixed`: empirical center search and independent
  validation under the target-radius residual bound.
- `controlled-restart` + `paper-amortized`: one-shot diameter-scaled recentering and the
  paper's block/probe schedule. The implementation uses a bounding-box diameter upper bound;
  this enlarges the death disk and is conservative relative to the theorem.
