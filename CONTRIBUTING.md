# Contributing

1. Install `uv` and run `uv sync`. Once `uv.lock` exists, use `uv sync --locked` for archival reproduction.
2. Create a focused branch and add tests with every behavior change.
3. Run `make check` before opening a pull request.
4. For performance changes, include before/after benchmark JSON from the same machine and
   compare at matched accuracy.
5. Do not change stochastic defaults, RNG logic, or certificate formulas without a convergence
   study, a paper cross-reference, and a release note.
6. Every compiled-kernel change requires either a pure-Python reference comparison or a direct
   mathematical invariant test.
7. Randomized tests must record their seed and avoid post-hoc tolerance changes.

The code of conduct and security policy apply to all contributions.

## Lockfile policy

`uv.lock` is a required release artifact. A contributor may bootstrap a fresh checkout with
`uv sync`, but pull requests that intentionally change dependencies must include the updated
lockfile. Tagged releases fail if the lockfile is absent or stale.
