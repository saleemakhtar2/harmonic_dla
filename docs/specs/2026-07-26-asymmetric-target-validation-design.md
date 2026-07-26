# Asymmetric-Target Finite-Boundary Validation

Date: 2026-07-26

## Decision

Build and run a theory-faithful validation of the paper's second-order versus
fourth-order finite-boundary prediction using strongly non-circular targets
inside circular death boundaries. Follow it with a dynamic growth comparison
that measures the calibration work saved by paper-amortized tracking.

The experiment will not use a non-circular death boundary. Every current theorem
in the manuscript assumes a circular death boundary; treating a non-circular
boundary as certified would overstate the result.

## Questions and preregistered hypotheses

The experiment answers two separate questions.

1. **Static law accuracy:** As the target-radius death ratio `rho` increases,
   does legacy uniform restart show the predicted second-order error while a
   one-shot centered restart shows fourth-order error?
2. **Dynamic calibration cost:** When controlled restart is used during growth,
   does blockwise paper-amortized tracking preserve explicit path-error
   accounting with substantially less calibration work than fresh
   per-attachment calibration?

Primary hypotheses:

- On the common primary fitting window `rho = {3, 4, 6, 8, 12, 16}`, the
  uniform-restart error is consistent with a leading `rho^-2` term.
- On the same window, the one-shot centered error is consistent with a leading
  `rho^-4` term and has a meaningfully steeper fitted decay than uniform restart.
- In dynamic growth, paper-amortized tracking uses at least 100 times fewer
  calibration probes than per-attachment sample splitting at `N = 1,200`, while
  retaining a nontrivial finite-path bound.

Exact Poisson return is the reference distribution for the static experiment.
It is not expected to lose in circular-boundary wall-clock comparisons.

## Alternatives considered

### A. Non-circular target, circular death boundary - selected

This directly exercises the theorem for arbitrary compact planar targets,
including highly asymmetric disk unions. Exact Poisson return remains an honest
reference, and the expected second- versus fourth-order slopes can be tested.

### B. Non-circular death boundary - rejected for the evidentiary benchmark

This could make exact return require a numerical boundary solve, but the paper's
Fourier identity, fourth-order certificate, and amortized theorem do not cover
that boundary. It would be a separate research extension rather than validation
of the present finding.

### C. Circular target with displaced centers - rejected as the primary case

It is cheap and visually simple, but too symmetric to demonstrate the claimed
arbitrary-target value and can accidentally suppress the moments the experiment
needs to measure.

## Scope

### Static targets

Use four frozen targets:

1. A deterministic connected lopsided-comb target made from touching equal
   disks. It has a long horizontal spine, a tall branch near one end, and a
   shorter branch on the opposite side.
2. Three exact-return DLA aggregates with 256 particles and seeds 42, 137, and
   911.

The synthetic target is fixed by coordinates in source control before any
radius-sweep result is inspected.

### Death-radius sweep

Use the common ratios:

`rho = {2, 3, 4, 6, 8, 12, 16, 24, 32}`

- `rho = 2` is a near-field diagnostic.
- `{3, 4, 6, 8, 12, 16}` is the immutable primary slope-fitting window.
- `{24, 32}` are noise-floor diagnostics and will not be added to or removed
  from the primary fit after results are seen.

Record launch radius, death radius, target radius about each tested center,
target diameter, `rho`, and death-to-diameter ratio for every result.

### Static policies

For each target and ratio:

1. **Exact Poisson:** exact exterior return at the same circular boundary.
2. **Uniform restart:** uniform relaunch around a fixed seed center.
3. **One-shot centered:** estimate the finite-boundary attachment barycenter
   from an independent search batch, move the computational-circle center once,
   and draw a fresh evaluation batch.

At a frozen target, per-step controlled restart and paper-amortized tracking
have the same attachment law once their centers and radii are fixed. The static
experiment therefore tests the centering law, while the dynamic experiment
separates their calibration schedules.

### Replication

Run a non-evidentiary pilot first to validate runtime, output shape, and noise:

- 5 independent replications
- 5,000 search probes for centered restart
- 20,000 evaluation attachments per policy, target, ratio, and replication

The pilot may change production sample counts only for feasibility or a
predeclared noise criterion; it may not change targets, radii, metrics, or the
fitting window.

Production defaults:

- 20 independent replications
- 8,192 independent search probes for centered restart
- 25,000 evaluation attachments per policy, target, ratio, and replication

If the pilot shows that a production point would exceed 120 seconds per
replication or that its split-half reference standard error exceeds 25% of the
measured uniform-versus-reference discrepancy, double all evaluation counts for
that entire ratio across every target and policy. Apply this rule mechanically
before production results are inspected.

### Dynamic growth

Run all four growth approaches at:

- Horizons `N = {1,200, 3,000}`
- Seeds `{42, 137, 911}`
- Four Numba threads

Use the package's post-fix configurations:

- Exact Poisson return.
- Uniform restart at fixed ratio 4.
- Per-attachment sample-split calibration with 4,096 search and 4,096
  validation probes.
- Paper-amortized tracking with `alpha = 0.48`, `gamma = 0.06`, `scale = 32`,
  `start = 1,024`, failure budget `0.001`, and an exact prefix.

The production report will distinguish measured wall-clock results from
deterministic schedule projections.

## Metrics

No single discretized statistic is treated as continuum total variation.

For every static replication report:

- Particle-ID attachment total variation, assigning an attachment to its
  nearest target disk with lowest-index deterministic tie-breaking.
- Polar-angle histogram total variation at 16, 32, 64, and 128 bins.
- Normalized attachment-barycenter displacement.
- A bounded-Lipschitz feature discrepancy based on a fixed, source-controlled
  family of clipped coordinate projections.
- Walker steps, restarts, calibration probes, and wall time.

For each metric:

- Aggregate independent complete-replication estimates.
- Report medians, central 95% bootstrap intervals, and all raw replication
  values.
- Fit the preregistered two-term form
  `E(rho) = a rho^(-2m) + b rho^(-2m-2)` on the common primary window.
- Also report a descriptive log-log slope with a replication bootstrap
  interval.

For dynamic runs report:

- Runtime and throughput.
- Growth and calibration walker steps.
- Calibration probes and events.
- Maximum local bound, conditional finite-path bound, and analytic
  infinite-history upper bound where defined.
- Standard cluster morphology summaries, explicitly labeled descriptive at
  three seeds.

## Architecture and data flow

1. Add a detailed frozen-probe API capable of selecting exact or uniform
   restart and returning attachment coordinates plus walker diagnostics. Keep
   the existing `probe()` behavior backward compatible.
2. Add an experiment module containing deterministic target construction,
   attachment partitioning, discrepancy metrics, fit inputs, and seed
   derivation.
3. Add a process-isolated benchmark driver. Each target/policy/ratio/replication
   is a separately reproducible record and can be resumed without overwriting a
   completed record.
4. Write append-safe per-record JSON files, then merge them into a validated
   canonical JSON artifact. Never treat an incomplete shard set as complete.
5. Generate figures and the final PDF only from the canonical merged artifact.

Seed derivation must be deterministic from the tuple:

`(experiment_version, phase, target, policy, rho, replication, batch_kind)`

Search, validation, reference, and evaluation streams must never overlap.

## Output contracts

Final artifacts:

- `output/asymmetric_validation/raw/` - resumable record shards
- `output/asymmetric_validation/asymmetric_validation_raw.json`
- `output/asymmetric_validation/figures/`
- `output/pdf/harmonic_dla_asymmetric_target_validation.pdf`

The canonical JSON records:

- Exact configuration and environment provenance.
- Source revision when available.
- Target coordinates or an immutable digest plus a lossless target artifact.
- Every seed and sample count.
- Complete geometry and boundary values.
- Raw replication metrics and fit-window membership.
- Pilot/production phase labels.
- Completion manifest and file digests.

## PDF report

The final report will contain:

1. Executive conclusion and honest novelty boundary.
2. Diagrams of the four frozen targets.
3. Log-log error-versus-radius plots with independent-replication intervals.
4. Fitted slope and two-term-model summaries.
5. Directional attachment-distribution comparisons at representative radii.
6. Error-versus-work Pareto plots.
7. Dynamic four-approach runtime and calibration-work comparisons.
8. Certificate interpretation and limitations.
9. Full reproducibility appendix.

The report must state that exact Poisson remains dominant when its circular
closed form is available. “Optimal” may be used only for a clearly defined
Pareto objective among approximate or calibration-based schemes.

## Error handling

- Reject incomplete, duplicated, or mixed-configuration record sets.
- Refuse to fit if any primary-window target/policy/replication is missing.
- Preserve failed shards with their exception and environment metadata.
- Never silently replace non-finite metrics, failed walkers, or exhausted
  restart budgets.
- Do not overwrite raw production records without an explicit flag.

## Testing strategy

- Unit tests for exact/uniform detailed probe-mode selection and deterministic
  seed derivation.
- Unit tests for target geometry, nearest-disk assignment, all metrics, fitting
  window enforcement, and manifest validation.
- Reference-versus-Numba parity tests on small frozen targets.
- Statistical smoke tests showing exact mode is invariant to circular death
  ratio within Monte Carlo uncertainty.
- Resume/no-clobber integration tests for record shards.
- A tiny end-to-end experiment that produces a canonical JSON file and a
  renderable PDF.
- Existing full test, lint, format, and type gates remain mandatory.

## Failure-mode review

### Monte Carlo noise hides fourth-order decay

Severity: critical to the experiment.

Mitigation: independent complete replications, fixed fitting window, explicit
noise-floor points, and a mechanical sample-doubling rule decided from the
pilot before production.

### A discretized TV plot exaggerates or hides continuum error

Severity: critical to interpretation.

Mitigation: particle partitions, four angular resolutions, barycenter error,
and a fixed bounded-Lipschitz feature family. Every discretized TV estimate is
labeled a lower bound.

### The chosen target accidentally cancels the leading moment

Severity: important.

Mitigation: use one fixed asymmetric synthetic target plus three independently
grown DLA targets. Do not select targets after inspecting fitted slopes.

### The exact prefix makes the dynamic paper run look artificially cheap

Severity: important.

Mitigation: report prefix and post-prefix work separately, run beyond 1,024 at
both horizons, and pair measured results with schedule counts through much
larger deterministic horizons.

### A claim of superiority silently excludes exact Poisson

Severity: critical to scientific honesty.

Mitigation: exact Poisson remains in every relevant table and Pareto plot.
Conclusions distinguish “best certified approximation” from “fastest method
when a closed-form exact kernel exists.”

## Non-goals

- Proving a theorem for non-circular death boundaries.
- Claiming morphology-law parity from three dynamic seeds.
- Selecting targets, radii, fit windows, or metrics after viewing production
  slopes.
- Claiming wall-clock superiority over exact circular Poisson return.

## Rollout

1. Implement and verify the detailed frozen-probe and metric infrastructure.
2. Run the tiny end-to-end fixture.
3. Run the pilot and publish its feasibility/noise decision before production.
4. Freeze the production manifest.
5. Run resumable production shards.
6. Validate completeness and generate the PDF.
7. Render every PDF page and perform visual and text-integrity checks.
