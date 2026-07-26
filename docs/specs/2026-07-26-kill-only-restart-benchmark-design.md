# Kill-Only Restart Benchmark and Report

Date: 2026-07-26

## Decision

Create a constrained-interface benchmark in which a circular death boundary
reports only `ESCAPED`, not the angle at which the walker escaped. Under this
interface, exact Poisson return remains the mathematical oracle but is not an
eligible implementation because it requires the discarded escape angle.

The benchmark will compare the three feasible restart policies and explain why
paper-amortized controlled restart is useful when correlated Poisson return
cannot be implemented through the available interface.

## Scientific claim

The report may claim:

- The classical Poisson kernel remains exact for ordinary Brownian motion
  between concentric circles.
- A kill-only interface lacks the escape angle required to condition a Poisson
  return sample, so exact return is operationally unavailable.
- Uniform restart, per-step controlled restart, and paper-amortized controlled
  restart remain feasible because they require only fresh launches and
  attachment samples.
- Under the paper's circular-boundary assumptions, centering controls the
  finite-boundary bias, while amortized tracking reduces the calibration work
  relative to recalibrating after every attachment.

The report must not claim that the Poisson kernel is mathematically undefined,
that it is slower than controlled restart, or that the benchmark proves a
theorem for non-circular boundaries or non-Brownian motion.

## Interface contract

The constrained boundary adapter exposes:

```text
ATTACHED(position)
ESCAPED
```

It deliberately does not expose:

```text
ESCAPED(angle)
```

The benchmark records a capability matrix for every policy:

| Policy | Needs escape angle | Needs attachment samples | Kill-only eligible |
|---|---:|---:|---:|
| Exact Poisson return | yes | no | no |
| Uniform restart | no | no | yes |
| Per-step controlled restart | no | yes | yes |
| Paper-amortized controlled restart | no | yes | yes |

The oracle runs in a separately instrumented process that retains the angle.
Its samples define the reference attachment law but its runtime is not ranked
on the constrained-interface Pareto frontier.

## Experiment

### Static validation

Reuse the canonical asymmetric-target production data:

- deterministic lopsided comb;
- three 256-particle exact-return DLA targets, seeds 42, 137, and 911;
- death-radius ratios `rho = {2, 3, 4, 6, 8, 12, 16, 24, 32}`;
- 20 complete replications per target and radius;
- exact Poisson oracle, uniform restart, and independent one-shot centering;
- particle-ID total variation, angular histogram total variation,
  bounded-Lipschitz discrepancy, barycenter error, walker work, and runtime.

The constrained-interface report reinterprets exact Poisson as an oracle rather
than an eligible policy. Reusing the existing immutable samples avoids rerunning
an identical stochastic experiment under a wrapper that merely hides a field.

### Dynamic comparison

Reuse the existing `N = 1,200` and `N = 3,000`, seeds 42, 137, and 911,
four-approach outputs for measured work and runtime. Exact Poisson is displayed
as the unconstrained oracle baseline. The constrained eligible frontier contains
uniform restart, per-step controlled restart, and paper-amortized controlled
restart.

The current paper-amortized runs contain an exact prefix. They therefore cannot
be labelled end-to-end kill-only executions. The report will:

1. disclose the exact prefix prominently;
2. use the measured post-prefix calibration schedule only as evidence for
   amortized probe savings;
3. present a separately budgeted uniform-restart prefix as the theorem-compatible
   end-to-end kill-only construction; and
4. label its prefix error as an analytic budget, not a measured runtime result.

No new end-to-end runtime claim will be made until a separately budgeted prefix
is implemented and run.

## Report structure and visualizations

Generate:

`output/pdf/harmonic_dla_kill_only_restart_benchmark.pdf`

The PDF will contain:

1. A plain-English explanation of what information the Poisson method needs.
2. A boundary-event flow diagram contrasting instrumented and kill-only APIs.
3. A capability matrix separating mathematical correctness from operational
   eligibility.
4. Static error-versus-radius plots showing the price of uniform restart and
   the benefit of centering against the oracle law.
5. A constrained Pareto chart using error, calibration probes, and measured
   runtime, with exact Poisson visually marked as oracle-only.
6. Dynamic calibration-work plots comparing per-step and amortized control.
7. A conclusion and limitations page that distinguishes measured findings,
   analytic certificates, and proposed kill-only deployment behavior.

Every plot is generated from committed canonical JSON artifacts. The report
generator validates required records and fails rather than silently omitting a
policy or dataset.

## Implementation isolation

- Add one report generator:
  `scripts/generate_kill_only_restart_report.py`.
- Add focused validation tests for the capability matrix, input validation,
  report text, and generated page count.
- Reuse existing data and plotting utilities where their contracts match.
- Do not change simulation kernels, public package APIs, or the existing
  asymmetric report.
- Store render-check PNGs under `tmp/pdfs/`; do not commit them.

## Verification

- Run the report generator against the canonical static and dynamic artifacts.
- Extract PDF text and assert that it includes `oracle only`,
  `escape angle`, `kill-only`, and the exact-prefix disclosure.
- Assert that prohibited claims such as `Poisson kernel is unavailable
  mathematically` do not appear.
- Render every page with Poppler and visually inspect all pages for clipping,
  overlap, unreadable labels, or missing figures.
- Run focused tests, then the complete test suite and static quality gates.
- Commit the generator, tests, design/plan documents, and final PDF.
- Push the existing branch and verify the remote tip.

## Failure-mode check

### The benchmark manufactures an artificial disadvantage for Poisson

Severity: minor, if disclosed.

Mitigation: describe the interface as a legacy/stateless/GPU integration
constraint, retain exact Poisson as the oracle, and never rank it as an eligible
implementation. The report demonstrates value under a stated interface, not
universal superiority.

### The exact prefix violates the kill-only contract

Severity: critical if hidden.

Mitigation: disclose that the existing dynamic run is not end-to-end kill-only,
use it only for post-prefix scheduling evidence, and specify the separately
budgeted approximate prefix required for a future end-to-end run.

### Conservative paper schedules remain slower in wall-clock time

Severity: minor.

Mitigation: make eligibility, accuracy, and calibration-probe savings the
primary findings. Report wall time honestly and do not promise that version 4
is the fastest feasible implementation on this machine.

### Hiding an already computed angle is not representative of all systems

Severity: minor.

Mitigation: make the loss of the angle an explicit input contract and list the
settings it represents. Do not generalize the result to simulators that already
retain the crossing location.

## Non-goals

- Changing the paper's theorem.
- Extending the result to irregular death boundaries, drift, or heterogeneous
  diffusion.
- Claiming that exact Poisson return should be removed from unconstrained DLA
  simulators.
- Re-running the 720-record static experiment when the existing artifact
  already contains the required policy samples.
- Claiming an end-to-end kill-only runtime for the current exact-prefix dynamic
  artifact.
