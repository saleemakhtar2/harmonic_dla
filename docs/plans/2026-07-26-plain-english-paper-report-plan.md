# Plain-English Paper Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a polished six-page, equation-free PDF that explains the paper to a non-technical reader, then commit, push, and open a GitHub PR for the branch.

**Architecture:** A standalone generator turns a fixed plain-English narrative into five locally drawn diagrams and a six-page ReportLab PDF. A small test module verifies the reader-language contract and PDF text. The report has no dependency on benchmark data or simulator execution.

**Tech Stack:** Python 3.12, Matplotlib, ReportLab, pypdf, Poppler, pytest, Ruff, ty, GitHub web interface.

**Assumptions:**

- Assumes a reader does not know the paper's field - will NOT teach mathematical proofs, notation, or source code.
- Assumes the “wanderer, pile, and safety fence” metaphor is a teaching device - will NOT represent every physical or mathematical detail literally.
- Assumes exact Poisson return remains the gold-standard method when its input is available - will NOT make a speed or superiority claim for the new method.
- Assumes a logged-in GitHub browser session is available to open the PR - will NOT use or expose credentials in source code or shell history.

---

## File structure

- `scripts/generate_plain_english_paper_report.py`: fixed narrative, locally generated diagrams, PDF composition, and command-line entry point.
- `tests/unit/test_plain_english_paper_report.py`: narrative-language and generated-PDF regression checks.
- `output/plain_english_paper/figures/`: five committed diagrams.
- `output/pdf/harmonic_dla_paper_plain_english.pdf`: final six-page reader report.
- `docs/specs/2026-07-26-plain-english-paper-report-design.md`: approved content contract.
- `docs/plans/2026-07-26-plain-english-paper-report-plan.md`: execution record.

### Task 1: Define and test the reader-language contract

**Files:**
- Create: `scripts/generate_plain_english_paper_report.py`
- Create: `tests/unit/test_plain_english_paper_report.py`

**Security flag:** none

**Does NOT cover:** This task validates explanatory content; it does not test simulation accuracy or paper proofs.

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path

from pypdf import PdfReader

from scripts.generate_plain_english_paper_report import build_report, report_text


def test_report_text_is_plain_english() -> None:
    text = report_text().lower()
    for phrase in ("safety fence", "balance point", "gold-standard", "one minute"):
        assert phrase in text
    for forbidden in ("rho", "gamma", "total variation", "barycenter", "o(", "="):
        assert forbidden not in text


def test_report_pdf_has_six_reader_pages(tmp_path: Path) -> None:
    output = tmp_path / "plain-english.pdf"
    build_report(output, tmp_path / "figures")
    reader = PdfReader(output)
    text = "\n".join(page.extract_text() or "" for page in reader.pages).lower()
    assert len(reader.pages) == 6
    assert all(phrase in text for phrase in ("safety fence", "does not", "one minute"))
```

- [ ] **Step 2: Confirm tests fail before the generator exists**

Run: `PYTHONPATH=src:. .venv/bin/pytest -q tests/unit/test_plain_english_paper_report.py`

Expected: FAIL with `ModuleNotFoundError` for the new generator.

- [ ] **Step 3: Add the fixed narrative contract**

```python
def report_text() -> str:
    return """A random wanderer helps grow a branching pile.
    A safety fence keeps the journey manageable. A gold-standard return rule
    uses the direction in which the wanderer left. The new idea moves the
    fence to a better balance point and checks that point in batches.
    The paper in one minute: improve a useful shortcut without pretending it
    beats the gold-standard method when that method is available."""
```

Ensure every final paragraph uses only the approved metaphor vocabulary and
states both limits: exact return is best when available, and the paper does not
promise an identical final picture.

- [ ] **Step 4: Run focused tests**

Run: `PYTHONPATH=src:. .venv/bin/pytest -q tests/unit/test_plain_english_paper_report.py`

Expected: PASS.

### Task 2: Build the illustrated six-page PDF

**Files:**
- Modify: `scripts/generate_plain_english_paper_report.py`
- Modify: `tests/unit/test_plain_english_paper_report.py`

**Security flag:** none

**Does NOT cover:** Visuals illustrate the mechanism; they do not constitute new empirical evidence.

- [ ] **Step 1: Add five locally generated diagrams**

Implement these named outputs under the supplied figure directory:

```python
FIGURES = (
    "growing_pile.png",
    "safety_fence.png",
    "lost_direction.png",
    "moved_centre.png",
    "batch_checkpoints.png",
)
```

Draw circles, dots, arrows, a lopsided pile, a moved fence centre, and calendar
checkpoints with Matplotlib. Label each graphic only with plain-English words.

- [ ] **Step 2: Compose exactly six ReportLab pages**

Build pages titled:

1. `A tiny story about a growing pile`
2. `Why computers add a safety fence`
3. `The small mistake in a random restart`
4. `The paper's key idea: move the centre`
5. `Checking in batches instead of every second`
6. `The paper in one minute`

Use a consistent header, footer, page number, large type, short paragraphs,
and one diagram per page. Do not insert equations, Greek letters, or technical
words from the manuscript without an immediate plain-English explanation.

- [ ] **Step 3: Add a negative wording regression test**

```python
def test_pdf_keeps_the_two_honest_limits(tmp_path: Path) -> None:
    output = tmp_path / "plain-english.pdf"
    build_report(output, tmp_path / "figures")
    text = " ".join(page.extract_text() or "" for page in PdfReader(output).pages).lower()
    assert "best choice when it is available" in text
    assert "does not promise" in text
    assert "faster than" not in text
```

- [ ] **Step 4: Run focused tests**

Run: `PYTHONPATH=src:. .venv/bin/pytest -q tests/unit/test_plain_english_paper_report.py`

Expected: PASS.

### Task 3: Generate and inspect the production PDF

**Files:**
- Create: `output/plain_english_paper/figures/growing_pile.png`
- Create: `output/plain_english_paper/figures/safety_fence.png`
- Create: `output/plain_english_paper/figures/lost_direction.png`
- Create: `output/plain_english_paper/figures/moved_centre.png`
- Create: `output/plain_english_paper/figures/batch_checkpoints.png`
- Create: `output/pdf/harmonic_dla_paper_plain_english.pdf`

**Security flag:** none

- [ ] **Step 1: Generate the report**

Run:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/generate_plain_english_paper_report.py \
  --figure-dir output/plain_english_paper/figures \
  --output output/pdf/harmonic_dla_paper_plain_english.pdf
```

Expected: five diagrams and a six-page PDF are written.

- [ ] **Step 2: Perform logical PDF checks**

Run:

```bash
.venv/bin/python -c "from pathlib import Path; from pypdf import PdfReader; p=Path('output/pdf/harmonic_dla_paper_plain_english.pdf'); r=PdfReader(p); t=' '.join((x.extract_text() or '') for x in r.pages).lower(); assert len(r.pages)==6; assert all(x in t for x in ('safety fence','balance point','gold-standard','one minute','does not promise')); assert all(x not in t for x in ('total variation','barycenter','rho','gamma')); print(p.stat().st_size)"
```

Expected: one byte size is printed and every assertion passes.

- [ ] **Step 3: Render and inspect every page**

Run:

```bash
mkdir -p tmp/pdfs/plain_english
pdftoppm -png -r 150 output/pdf/harmonic_dla_paper_plain_english.pdf tmp/pdfs/plain_english/page
```

Inspect all six PNGs. Check visual hierarchy, readable diagrams, short line
lengths, generous margins, page numbering, and that no page looks like a
technical paper.

### Task 4: Verify, push, and open the PR

**Files:**
- Modify: `docs/plans/2026-07-26-plain-english-paper-report-plan.md`
- Commit: the design, plan, generator, tests, five figures, and final PDF.

**Security flag:** security

**Does NOT cover:** PR creation must use an existing authenticated GitHub browser session; it must not add credentials to the repository or shell commands.

- [ ] **Step 1: Run complete verification**

Run:

```bash
PYTHONPATH=src:. .venv/bin/pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
.venv/bin/ty check src scripts
git diff --check
```

Expected: all tests and static checks pass.

- [ ] **Step 2: Commit and push only intended files**

Run:

```bash
git add docs/specs/2026-07-26-plain-english-paper-report-design.md \
  docs/plans/2026-07-26-plain-english-paper-report-plan.md \
  scripts/generate_plain_english_paper_report.py \
  tests/unit/test_plain_english_paper_report.py \
  output/plain_english_paper/figures \
  output/pdf/harmonic_dla_paper_plain_english.pdf
git commit -m "Add plain-English paper report"
git push origin codex/asymmetric-target-validation-20260726
```

Expected: the remote branch contains the new report while unrelated local-only
workflow files, older outputs, and `tmp/` remain untracked.

- [ ] **Step 3: Open the GitHub pull request in the authenticated browser**

Create a PR from `codex/asymmetric-target-validation-20260726` into `main`.

Title: `Add validation reports and plain-English paper guide`

Description:

```markdown
## What changed

Adds reproducible asymmetric-target and kill-only restart validation reports,
plus a six-page plain-English guide to the paper.

## Why

The technical report explains the constrained-interface use case. The new guide
makes the paper understandable without maths or programming background.

## How to verify

Run `PYTHONPATH=src:. .venv/bin/pytest -q` and open
`output/pdf/harmonic_dla_paper_plain_english.pdf`.

## Notable decisions

Exact Poisson remains the gold-standard method when available. The reports do
not make an unsupported universal wall-clock speed claim.
```

Expected: GitHub returns a PR URL with `main` as base and the feature branch as
head. Read the PR page once to confirm its title, description, and changed-file
summary.
