# Plain-English Paper Report

Date: 2026-07-26

## Decision

Create a polished six-page PDF for a reader with no maths, coding, or physics
background. It explains the paper through a single everyday story: a wanderer
trying to reach a growing pile of snowballs while a safety fence occasionally
sends the wanderer back to try again.

The report will be written in plain English. It will not show equations,
notation, asymptotic names, proof steps, code, or unexplained specialist words.

## Reader promise

After reading the report, a non-technical reader should be able to answer:

1. What is being simulated?
2. Why is a safety fence used?
3. What goes wrong when a wanderer is restarted in a random direction?
4. What does it mean to move the centre of the fence?
5. Why does checking the best centre in batches matter?
6. What the paper proves, and what it deliberately does not claim.

## Narrative structure

### Page 1 - The picture in everyday terms

Introduce a random wanderer building a coral-like pile of snowballs. A dot is
released, wanders randomly, and sticks when it first reaches the pile. Repeat
this many times and the pile grows into a branching shape.

### Page 2 - The safety fence and the shortcut

Explain why a computer places a large circular safety fence around the pile.
When a wanderer reaches it, continuing the full journey is inconvenient. A
common shortcut throws the wanderer away and starts a fresh one from a random
point on an inner ring.

### Page 3 - Why random restart can lean the answer

Use a lopsided pile and a simple map-like sketch. A wanderer that escaped on the
right is more likely to matter on the right when it comes back. Replacing that
memory with a random restart can introduce a gentle directional lean. Describe
the exact Poisson return as the gold-standard method when the escape direction
is available, not as a method this paper defeats.

### Page 4 - The new centering idea

Explain the key insight as moving the fence so it is centred on where the
shortcut actually tends to attach, rather than on an arbitrary starting point.
This cancels the largest left-versus-right error. Use a before-and-after target
diagram and an intuitive “balance point” explanation.

### Page 5 - Checking in batches

Explain why finding that balance point from scratch after every new snowball is
wasteful. The paper proves that the balance point cannot jump wildly when only
one bounded snowball is added. Therefore the simulator can check occasionally,
reuse the answer for a while, and still keep a written error allowance. Use a
calendar/checkpoint diagram.

### Page 6 - What this means and what it does not mean

State the narrow contribution:

- it gives a trustworthy way to improve an approximate restart rule;
- it reduces the amount of extra checking that a cautious method needs;
- it helps legacy or constrained systems that cannot use an exact return;
- it does not beat exact Poisson return when exact return is readily available;
- it does not guarantee a perfect picture or prove every simulation error has
  disappeared.

End with a five-line summary labelled “The paper in one minute.”

## Visual and language rules

- Use one friendly visual per page: a growing pile, circular fence, directional
  arrows, moved centre, and calendar checkpoints.
- Use short paragraphs, familiar words, large headings, and generous spacing.
- Prefer “wanderer”, “pile”, “fence”, “balance point”, and “checking” over
  “Brownian particle”, “aggregate”, “death boundary”, “barycenter”, and
  “calibration”.
- If an unavoidable technical term appears, explain it in the same sentence.
- Mention the exact Poisson method by name only once, with the plain-English
  label “gold-standard return rule”.
- All graphics must be generated locally from simple geometric shapes; no
  external images or claims based on decorative evidence.

## Scope and non-goals

Final artifact:

`output/pdf/harmonic_dla_paper_plain_english.pdf`

Supporting source:

`scripts/generate_plain_english_paper_report.py`

The report is explanatory, not a new experiment. It will not present the
earlier benchmark as proof of the paper, discuss the kill-only interface in
detail, or ask the reader to interpret a graph of technical metrics.

## Verification

- Extract the PDF text and check it contains the narrative landmarks: `safety
  fence`, `balance point`, `gold-standard`, `does not`, and `one minute`.
- Check it contains no mathematical notation, no equations, and no undefined
  technical terms from the paper's title.
- Render every page with Poppler and inspect it visually for clipping, overlap,
  unreadable labels, and page-flow problems.
- Add focused tests for report text and page count.
- Run the complete project test and quality suite before committing.

## Failure-mode check

### It implies the shortcut is always better than the gold-standard method

Severity: critical.

Mitigation: say on pages 3 and 6 that exact Poisson return is the best practical
choice when it is available and the escape direction has been retained.

### It turns a probability guarantee into a promise of an identical final shape

Severity: critical.

Mitigation: explain that the paper keeps the chance of a disagreement within a
chosen allowance; it does not promise that every simulated branching picture is
visually identical.

### It hides that this is a theoretical contribution rather than a universal
speed result

Severity: minor.

Mitigation: say that the result reduces extra checking work and gives an error
allowance. Do not call it universally faster.

### It becomes too childish or inaccurate

Severity: minor.

Mitigation: use adult, clear visual design and immediately pair each metaphor
with its computational role.
