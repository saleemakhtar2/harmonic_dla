# Performance methodology

Performance claims must be made at matched accuracy, not raw particle count alone.

1. Warm Numba compilation separately.
2. Use identical hardware, thread counts, spatial-index settings, particle radius, and tolerance.
3. Compare exact return, uniform restart, fixed sample-split control, and paper-amortized control.
4. Report growth time, calibration time, WOS steps, restart events, probe count, peak memory,
   and the applicable error bound.
5. Repeat complete runs; do not quote only inner-kernel microbenchmarks.
6. Retain machine information and raw benchmark JSON.

The package makes no pre-emptive claim to be the fastest DLA implementation. Its target is the
fastest transparent Python-facing implementation at a declared accuracy, subject to evidence.
