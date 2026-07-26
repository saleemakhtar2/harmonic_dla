# Asymmetric-Target Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:subagent-driven-development (recommended) or superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run the preregistered asymmetric-target radius sweep and dynamic calibration comparison, then generate a visually verified PDF report.

**Architecture:** Add a detailed frozen-probe path that supports exact and uniform return without changing the existing `probe()` contract. Put deterministic target construction, seed derivation, attachment partitions, discrepancy metrics, and slope fitting in a focused validation module. Run process-isolated resumable shards into a canonical manifest, then render the static and dynamic results through a dedicated ReportLab/matplotlib report generator.

**Tech Stack:** Python 3.13, NumPy, Numba, pytest, ReportLab, matplotlib, Poppler.

**Assumptions:**

- The current circular death-boundary theorem remains the only certified theorem; this plan does not add a non-circular death boundary.
- The exact frozen-probe reference uses the existing circular Poisson-return kernel; it is not a wall-clock claim for arbitrary domains.
- Static attachment distributions are compared through finite partitions and fixed bounded-Lipschitz features; these are reported as empirical diagnostics, not continuum-TV proofs.

---

## File map

- Modify `src/harmonic_dla/models.py`, `src/harmonic_dla/api.py`, `src/harmonic_dla/backends/base.py`, `src/harmonic_dla/backends/reference/backend.py`, `src/harmonic_dla/backends/numba_cpu/backend.py`, and `src/harmonic_dla/backends/numba_cpu/kernels.py` for detailed frozen-probe results and exact/uniform mode selection.
- Create `src/harmonic_dla/validation.py` for deterministic target geometry, seed derivation, attachment partitions, discrepancy metrics, bootstrap summaries, and preregistered fits.
- Create `scripts/run_asymmetric_validation.py` for pilot/production static shards, dynamic runs, resume checks, and canonical-manifest creation.
- Create `scripts/generate_asymmetric_validation_report.py` for charts, tables, and PDF assembly.
- Create `tests/unit/test_detailed_probe.py`, `tests/unit/test_validation_metrics.py`, and `tests/integration/test_asymmetric_validation.py`.
- Generate `output/asymmetric_validation/` and `output/pdf/harmonic_dla_asymmetric_target_validation.pdf` only after the implementation and production manifest pass validation.

### Task 1: Add a detailed frozen-probe contract

**Files:**
- Modify: `src/harmonic_dla/models.py`
- Modify: `src/harmonic_dla/api.py`
- Modify: `src/harmonic_dla/backends/base.py`
- Modify: `src/harmonic_dla/backends/reference/backend.py`
- Modify: `src/harmonic_dla/backends/numba_cpu/backend.py`
- Modify: `src/harmonic_dla/backends/numba_cpu/kernels.py`
- Test: `tests/unit/test_detailed_probe.py`

**Security flag:** `none`

- [ ] **Step 1: Write failing tests**

Add tests asserting that `probe_detailed(..., restart_mode=RestartMode.UNIFORM_RESTART)` and `probe_detailed(..., restart_mode=RestartMode.EXACT_RETURN)` return a `ProbeResult` with `(probes, 2)` attachments and non-negative walker-step/restart counters; assert that the legacy `probe()` result remains an attachment array and that reference/Numba outputs have the same shape on a small frozen target.

- [ ] **Step 2: Verify red**

Run `PYTHONPATH=src:. .venv/bin/pytest -q tests/unit/test_detailed_probe.py`.

Expected: collection or attribute failure because `ProbeResult` and `probe_detailed` do not yet exist.

- [ ] **Step 3: Implement the smallest contract**

Add a validated `ProbeResult` dataclass with `attachments`, `walker_steps`, and `restarts`. Add `probe_detailed()` to the public API and backend protocol. Thread an integer restart-mode argument through the Numba `probe_batch` kernel and reference backend; default the existing `probe()` wrapper to the current uniform behavior. Return counters from both backends without changing simulation diagnostics or serialized result schema.

- [ ] **Step 4: Verify green**

Run `PYTHONPATH=src:. .venv/bin/pytest -q tests/unit/test_detailed_probe.py tests/unit/test_api.py tests/unit/test_boundaries.py`.

Expected: all targeted tests pass and the existing `probe()` behavior remains unchanged.

### Task 2: Implement deterministic validation geometry and metrics

**Files:**
- Create: `src/harmonic_dla/validation.py`
- Test: `tests/unit/test_validation_metrics.py`

**Security flag:** `none`

- [ ] **Step 1: Write failing tests**

Test that `make_lopsided_comb()` is deterministic, connected at particle-radius spacing, and has a nonzero centered second moment; test lowest-index nearest-disk tie-breaking; test that identical empirical laws have zero histogram TV and bounded-Lipschitz discrepancy; test that `schedule_seed((...))` is deterministic and distinct across batch kinds; test that `fit_power_law()` rejects missing primary-window ratios and uses only `{3,4,6,8,12,16}`.

- [ ] **Step 2: Verify red**

Run `PYTHONPATH=src:. .venv/bin/pytest -q tests/unit/test_validation_metrics.py`.

Expected: import failures for the new validation functions.

- [ ] **Step 3: Implement the metric module**

Implement deterministic lopsided-comb coordinates, nearest-disk assignment, 16/32/64/128-bin angle histograms, particle-ID and histogram TV, a fixed clipped-coordinate bounded-Lipschitz feature family, deterministic tuple-based seed derivation, bootstrap intervals over complete replications, and the preregistered two-term/log-log fits.

- [ ] **Step 4: Verify green**

Run `PYTHONPATH=src:. .venv/bin/pytest -q tests/unit/test_validation_metrics.py tests/unit/test_analysis.py`.

Expected: all metric and fit tests pass, including rejection of incomplete primary windows.

### Task 3: Build resumable static pilot/production shards

**Files:**
- Create: `scripts/run_asymmetric_validation.py`
- Test: `tests/integration/test_asymmetric_validation.py`

**Security flag:** `none`

- [ ] **Step 1: Write failing integration tests**

Add a tiny fixture test that runs one target, two ratios, two policies, and two replications into a temporary shard directory; assert that rerunning the same record does not overwrite it, that the merge rejects a missing policy/ratio, and that the canonical JSON contains geometry, seeds, metrics, and completion digests.

- [ ] **Step 2: Verify red**

Run `PYTHONPATH=src:. .venv/bin/pytest -q tests/integration/test_asymmetric_validation.py`.

Expected: missing driver/module failures.

- [ ] **Step 3: Implement the shard driver**

Implement CLI phases `--phase pilot|production`, target generation from the fixed synthetic target plus saved exact DLA aggregates, ratios `{2,3,4,6,8,12,16,24,32}`, exact/uniform/one-shot-centered policies, deterministic seeds, detailed probe diagnostics, mechanical pilot sample-doubling decision, no-clobber shard writes, completeness validation, and canonical merge. Store every geometry value and primary-window membership in each record.

- [ ] **Step 4: Verify green**

Run `PYTHONPATH=src:. .venv/bin/pytest -q tests/integration/test_asymmetric_validation.py` and then `PYTHONPATH=src:. .venv/bin/python scripts/run_asymmetric_validation.py --help`.

Expected: the fixture passes and the CLI displays both phases and manifest options.

### Task 4: Run the pilot, freeze the production manifest, and run dynamic comparisons

**Files:**
- Modify: `scripts/run_asymmetric_validation.py` only if the pilot triggers the preregistered mechanical sample-doubling rule.
- Generate: `output/asymmetric_validation/` and `output/benchmark/asymmetric_dynamic_*.json`.

**Security flag:** `none`

- [ ] **Step 1: Run the pilot**

Run `PYTHONPATH=src:. .venv/bin/python scripts/run_asymmetric_validation.py --phase pilot --output-dir output/asymmetric_validation` and record its machine, package, runtime, split-half noise, and mechanical production-count decision.

- [ ] **Step 2: Freeze and verify the manifest**

Run the production phase with the pilot decision recorded in the manifest. Refuse to start if any target, ratio, policy, or sample-count field differs from the approved specification.

- [ ] **Step 3: Run the dynamic benchmark**

Run `PYTHONPATH=src:. .venv/bin/python scripts/benchmark_four_approaches.py --particles 1200 --seeds 42 137 911 --threads 4 --output output/benchmark/asymmetric_dynamic_1200.json` and repeat with `--particles 3000 --output output/benchmark/asymmetric_dynamic_3000.json`. Record growth/calibration work and certificate fields; do not infer morphology parity from three seeds.

- [ ] **Step 4: Validate raw completeness**

Run the driver's merge/manifest verifier. Expected: every production target/ratio/policy/replication tuple is present exactly once, every dynamic JSON contains all four approaches, and no partial run is treated as complete.

### Task 5: Generate the visual PDF report

**Files:**
- Create: `scripts/generate_asymmetric_validation_report.py`
- Test: extend `tests/integration/test_asymmetric_validation.py` with PDF smoke assertions.

**Security flag:** `none`

- [ ] **Step 1: Write failing report test**

Add a fixture test that calls the report generator on the tiny canonical artifact and asserts that it creates a readable PDF containing the static/dynamic section titles and the exact-method caveat.

- [ ] **Step 2: Verify red**

Run `PYTHONPATH=src:. .venv/bin/pytest -q tests/integration/test_asymmetric_validation.py`.

Expected: missing report generator failure.

- [ ] **Step 3: Implement the report**

Use matplotlib for target panels, error-versus-`rho` log-log plots with bootstrap intervals, slope comparison, directional histograms, and cost/error Pareto plots. Use ReportLab for a polished report with an executive conclusion, static protocol, dynamic results, certificates, limitations, and reproducibility appendix. State that exact Poisson remains dominant whenever its circular closed form is available.

- [ ] **Step 4: Verify green and render**

Run `PYTHONPATH=src:. .venv/bin/pytest -q tests/integration/test_asymmetric_validation.py`, then `PYTHONPATH=src:. .venv/bin/python scripts/generate_asymmetric_validation_report.py --input output/asymmetric_validation/asymmetric_validation_raw.json --dynamic-glob output/benchmark/asymmetric_dynamic_*.json --output output/pdf/harmonic_dla_asymmetric_target_validation.pdf`. Render with Poppler and inspect every page.

### Task 6: Full verification and handoff

**Files:**
- Modify: none unless verification finds a defect.

**Security flag:** `none`

- [ ] Run `PYTHONPATH=src:. .venv/bin/pytest -q` and expect the full suite to pass.
- [ ] Run `.venv/bin/ruff check src tests scripts` and `.venv/bin/ruff format --check src tests scripts`.
- [ ] Run `.venv/bin/ty check src`.
- [ ] Run PDF text extraction and assert the report has no placeholder text, contains all four approaches, and includes the exact-Poisson limitation.
- [ ] Run a stub scan over all files created by this plan.
- [ ] Verify PDF page count, PNG rendering, raw manifest completeness, and output paths before claiming completion.
