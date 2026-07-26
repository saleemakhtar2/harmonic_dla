# Reproducibility

- Pin dependencies with a committed universal `uv.lock`; tagged releases use `uv sync --locked`. A fresh pre-lock checkout may bootstrap with `uv sync`.
- Record the config TOML, package version, commit SHA, platform, Numba thread count, and seed.
- Run `hdla warmup` before timing to exclude first-use compilation.
- Never compare backends using only the same integer seed: the reference and compiled engines
  intentionally use different RNG implementations. Compare laws and invariants, not paths.
- Numba growth streams are indexed by particle number. Splitting a fixed-center run into
  checkpoint chunks does not change the generated aggregate.
- Calibration search, validation, and subsequent growth use disjoint random streams.
- For manuscript figures, retain raw attachment samples and the complete replication manifest.
