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


def test_pdf_keeps_the_two_honest_limits(tmp_path: Path) -> None:
    output = tmp_path / "plain-english.pdf"
    build_report(output, tmp_path / "figures")
    text = " ".join(
        " ".join((page.extract_text() or "").lower().split()) for page in PdfReader(output).pages
    )
    assert "best choice when it is available" in text
    assert "does not promise" in text
    assert "faster than" not in text
