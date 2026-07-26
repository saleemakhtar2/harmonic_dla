# Kill-Only Restart Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate, verify, commit, and push a polished PDF showing when paper-amortized controlled restart is useful under a circular kill-only boundary interface that discards escape angles.

**Architecture:** A standalone report module loads and validates the already committed canonical static and dynamic JSON artifacts. Pure helpers define the policy capability contract and derived summaries; plotting and ReportLab composition consume only those validated summaries. Exact Poisson remains the instrumented oracle, while the eligible Pareto comparison contains uniform, per-step controlled, and paper-amortized policies.

**Tech Stack:** Python 3.12, NumPy, Matplotlib, ReportLab, pypdf/pdfplumber, Poppler, pytest, Ruff, ty, Git.

**Assumptions:**

- Assumes the committed 720-record production artifact and both dynamic JSON artifacts are complete - will NOT silently regenerate or accept incomplete data.
- Assumes ordinary planar Brownian motion and a circular death boundary - will NOT generalize the paper's theorem to irregular boundaries, drift, or heterogeneous diffusion.
- Assumes a kill-only API discards the escape angle - will NOT claim Poisson return is ineligible when an integration retains that angle.
- Assumes the existing dynamic run's exact prefix is disclosed and used only for post-prefix scheduling evidence - will NOT label it an end-to-end kill-only runtime.

---

## File structure

- `scripts/generate_kill_only_restart_report.py`: capability contract, JSON validation, summary derivation, charts, PDF composition, and CLI.
- `tests/unit/test_kill_only_restart_report.py`: pure-helper and PDF smoke/regression tests.
- `output/asymmetric_validation/figures_kill_only/`: generated committed chart assets used by the report.
- `output/pdf/harmonic_dla_kill_only_restart_benchmark.pdf`: final report.
- `docs/specs/2026-07-26-kill-only-restart-benchmark-design.md`: approved scientific and reporting contract.
- `docs/plans/2026-07-26-kill-only-restart-benchmark-plan.md`: implementation and verification record.

### Task 1: Lock the constrained-interface contract and artifact validation

**Files:**
- Create: `scripts/generate_kill_only_restart_report.py`
- Create: `tests/unit/test_kill_only_restart_report.py`

**Security flag:** none

**Does NOT cover:** It validates report inputs and policy eligibility; it does not alter walker execution, restart laws, or package APIs.

- [ ] **Step 1: Write failing capability and validation tests**

```python
from pathlib import Path

import pytest

from scripts.generate_kill_only_restart_report import (
    CAPABILITIES,
    load_inputs,
    validate_inputs,
)


def test_kill_only_capability_matrix() -> None:
    assert CAPABILITIES["exact"]["needs_escape_angle"] is True
    assert CAPABILITIES["exact"]["kill_only_eligible"] is False
    for policy in ("uniform", "controlled-fixed", "paper-amortized"):
        assert CAPABILITIES[policy]["needs_escape_angle"] is False
        assert CAPABILITIES[policy]["kill_only_eligible"] is True


def test_validate_inputs_rejects_missing_static_policy() -> None:
    static = {
        "records": [{"target_name": "target", "rho": 2, "policies": {"uniform": {}}}],
        "targets": {},
    }
    with pytest.raises(ValueError, match="one-shot-centered"):
        validate_inputs(static, [{"records": [], "summaries": {}}])


def test_committed_inputs_validate() -> None:
    static, dynamics = load_inputs(
        Path("output/asymmetric_validation/asymmetric_validation_production.json"),
        (
            Path("output/benchmark/asymmetric_dynamic_1200.json"),
            Path("output/benchmark/asymmetric_dynamic_3000.json"),
        ),
    )
    validate_inputs(static, dynamics)
    assert len(static["records"]) == 720
```

- [ ] **Step 2: Run the tests and confirm the module is absent**

Run: `PYTHONPATH=src:. .venv/bin/pytest -q tests/unit/test_kill_only_restart_report.py`

Expected: FAIL during collection with `ModuleNotFoundError`.

- [ ] **Step 3: Add the capability contract and strict loaders**

```python
CAPABILITIES = {
    "exact": {
        "label": "Exact Poisson oracle",
        "needs_escape_angle": True,
        "needs_attachment_samples": False,
        "kill_only_eligible": False,
    },
    "uniform": {
        "label": "Uniform restart",
        "needs_escape_angle": False,
        "needs_attachment_samples": False,
        "kill_only_eligible": True,
    },
    "controlled-fixed": {
        "label": "Per-step controlled restart",
        "needs_escape_angle": False,
        "needs_attachment_samples": True,
        "kill_only_eligible": True,
    },
    "paper-amortized": {
        "label": "Paper-amortized control",
        "needs_escape_angle": False,
        "needs_attachment_samples": True,
        "kill_only_eligible": True,
    },
}


def validate_inputs(static: dict[str, Any], dynamics: list[dict[str, Any]]) -> None:
    records = static.get("records")
    if not isinstance(records, list) or len(records) != 720:
        raise ValueError("static artifact must contain exactly 720 records")
    for record in records:
        policies = record.get("policies", {})
        for policy in ("uniform", "one-shot-centered"):
            if policy not in policies:
                raise ValueError(f"static record is missing {policy}")
    if len(dynamics) != 2:
        raise ValueError("exactly two dynamic artifacts are required")
    for data in dynamics:
        summaries = data.get("summaries", {})
        for policy in CAPABILITIES:
            if policy not in summaries:
                raise ValueError(f"dynamic artifact is missing {policy}")
```

- [ ] **Step 4: Run focused tests**

Run: `PYTHONPATH=src:. .venv/bin/pytest -q tests/unit/test_kill_only_restart_report.py`

Expected: PASS.

### Task 2: Build the charts and PDF

**Files:**
- Modify: `scripts/generate_kill_only_restart_report.py`
- Modify: `tests/unit/test_kill_only_restart_report.py`

**Security flag:** none

**Does NOT cover:** The report labels exact Poisson as oracle-only under the constrained interface; it does not remove exact Poisson from the unconstrained benchmark or claim universal superiority.

- [ ] **Step 1: Add a failing PDF content test**

```python
from pypdf import PdfReader

from scripts.generate_kill_only_restart_report import build_report


def test_build_report_discloses_oracle_and_exact_prefix(tmp_path: Path) -> None:
    static, dynamics = load_inputs(
        Path("output/asymmetric_validation/asymmetric_validation_production.json"),
        (
            Path("output/benchmark/asymmetric_dynamic_1200.json"),
            Path("output/benchmark/asymmetric_dynamic_3000.json"),
        ),
    )
    output = tmp_path / "report.pdf"
    build_report(static, dynamics, output, tmp_path / "figures")
    text = "\n".join(page.extract_text() or "" for page in PdfReader(output).pages)
    assert "oracle only" in text.lower()
    assert "escape angle" in text.lower()
    assert "kill-only" in text.lower()
    assert "exact prefix" in text.lower()
    assert "mathematically unavailable" not in text.lower()
    assert 5 <= len(PdfReader(output).pages) <= 8
```

- [ ] **Step 2: Run the PDF test and confirm `build_report` is absent**

Run: `PYTHONPATH=src:. .venv/bin/pytest -q tests/unit/test_kill_only_restart_report.py`

Expected: FAIL importing or calling `build_report`.

- [ ] **Step 3: Implement derived summaries and four sharp visualizations**

Implement:

```python
def static_summary(static: dict[str, Any], policy: str, rho: int) -> dict[str, float]:
    rows = [
        record["policies"][policy]
        for record in static["records"]
        if int(record["rho"]) == rho
    ]
    return {
        "particle_id_tv": float(np.median([row["particle_id_tv"] for row in rows])),
        "bounded_lipschitz": float(np.median([row["bounded_lipschitz"] for row in rows])),
        "seconds": float(np.median([row["seconds"] for row in rows])),
    }


def dynamic_rows(dynamics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "particles": int(data["records"][0]["particles"]),
            "policy": policy,
            "runtime": float(summary["runtime_median_seconds"]),
            "probes": float(summary["calibration_probes_median"]),
            "eligible": bool(CAPABILITIES[policy]["kill_only_eligible"]),
        }
        for data in dynamics
        for policy, summary in data["summaries"].items()
    ]
```

Create charts:

1. `boundary_information_flow.png`: two-panel event-flow diagram showing that Poisson needs `escape angle -> conditioned return`, while kill-only exposes only `ESCAPED -> fresh restart`.
2. `static_accuracy.png`: median particle-ID TV and bounded-Lipschitz discrepancy against `rho` for uniform and one-shot centered restart.
3. `eligible_pareto.png`: static accuracy versus measured policy time with exact Poisson annotated outside the eligible frontier as oracle-only.
4. `dynamic_calibration.png`: runtime and calibration probes at `N=1,200` and `N=3,000`, using hollow markers for oracle-only exact Poisson.

- [ ] **Step 4: Compose a polished 6-page ReportLab document**

Build these pages:

1. Executive result and constrained-interface diagram.
2. Capability matrix and exact mathematical/operational distinction.
3. Static accuracy results from the 720-record experiment.
4. Eligible accuracy-cost frontier.
5. Dynamic calibration work with the exact-prefix disclosure.
6. Findings, limitations, artifact provenance, and reproducibility commands.

Add a page callback with consistent header, footer, and `Page N` numbering.
Use ASCII hyphens throughout the source text.

- [ ] **Step 5: Run focused tests**

Run: `PYTHONPATH=src:. .venv/bin/pytest -q tests/unit/test_kill_only_restart_report.py`

Expected: PASS with a 5-8 page PDF containing all required disclosures.

### Task 3: Generate and visually verify the production artifact

**Files:**
- Create: `output/asymmetric_validation/figures_kill_only/boundary_information_flow.png`
- Create: `output/asymmetric_validation/figures_kill_only/static_accuracy.png`
- Create: `output/asymmetric_validation/figures_kill_only/eligible_pareto.png`
- Create: `output/asymmetric_validation/figures_kill_only/dynamic_calibration.png`
- Create: `output/pdf/harmonic_dla_kill_only_restart_benchmark.pdf`

**Security flag:** none

- [ ] **Step 1: Generate the report from canonical inputs**

Run:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/generate_kill_only_restart_report.py \
  --static output/asymmetric_validation/asymmetric_validation_production.json \
  --dynamic-1200 output/benchmark/asymmetric_dynamic_1200.json \
  --dynamic-3000 output/benchmark/asymmetric_dynamic_3000.json \
  --figure-dir output/asymmetric_validation/figures_kill_only \
  --output output/pdf/harmonic_dla_kill_only_restart_benchmark.pdf
```

Expected: the final PDF and four PNG figures are written without warnings or omitted data.

- [ ] **Step 2: Run logical PDF checks**

Run:

```bash
.venv/bin/python -c "from pathlib import Path; from pypdf import PdfReader; p=Path('output/pdf/harmonic_dla_kill_only_restart_benchmark.pdf'); r=PdfReader(p); t=' '.join((x.extract_text() or '') for x in r.pages).lower(); assert 5 <= len(r.pages) <= 8; assert all(x in t for x in ('oracle only','escape angle','kill-only','exact prefix')); assert 'mathematically unavailable' not in t; print(len(r.pages), p.stat().st_size)"
```

Expected: page count and byte size are printed; every assertion passes.

- [ ] **Step 3: Render every page**

Run:

```bash
mkdir -p tmp/pdfs/kill_only
pdftoppm -png -r 150 \
  output/pdf/harmonic_dla_kill_only_restart_benchmark.pdf \
  tmp/pdfs/kill_only/page
```

Expected: one non-empty PNG per PDF page.

- [ ] **Step 4: Inspect all rendered pages**

Inspect each page PNG at original or high detail. Check headings, body text,
tables, captions, axes, legends, margins, page numbers, and line breaks.
If any defect exists, update the generator, regenerate, rerender, and inspect
the entire changed page set again.

### Task 4: Verify, commit, push, and confirm the PR branch

**Files:**
- Modify: `docs/plans/2026-07-26-kill-only-restart-benchmark-plan.md`
- Commit only the design, plan, generator, focused tests, four figures, and final PDF.

**Security flag:** none

- [ ] **Step 1: Run complete verification**

Run:

```bash
PYTHONPATH=src:. .venv/bin/pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
.venv/bin/ty check src scripts
git diff --check
```

Expected: all tests pass; Ruff, formatting, ty, and whitespace checks are clean.

- [ ] **Step 2: Confirm the intended commit set**

Run:

```bash
git status --short
git diff --stat
```

Expected: existing local-only workflow, older report, benchmark, and `tmp/`
files remain untracked and are not staged.

- [ ] **Step 3: Commit the report**

Run:

```bash
git add \
  docs/specs/2026-07-26-kill-only-restart-benchmark-design.md \
  docs/plans/2026-07-26-kill-only-restart-benchmark-plan.md \
  scripts/generate_kill_only_restart_report.py \
  tests/unit/test_kill_only_restart_report.py \
  output/asymmetric_validation/figures_kill_only \
  output/pdf/harmonic_dla_kill_only_restart_benchmark.pdf
git commit -m "Add kill-only restart benchmark report"
```

Expected: one commit containing only the intended benchmark/report artifacts.

- [ ] **Step 4: Push and verify the remote tip**

Run:

```bash
git push origin codex/asymmetric-target-validation-20260726
git rev-parse HEAD
git ls-remote --heads origin codex/asymmetric-target-validation-20260726
```

Expected: local and remote commit hashes match.
