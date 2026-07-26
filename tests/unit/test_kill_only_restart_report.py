from pathlib import Path

import pytest
from pypdf import PdfReader
from scripts.generate_kill_only_restart_report import (
    CAPABILITIES,
    build_report,
    load_inputs,
    validate_inputs,
)

STATIC_PATH = Path("output/asymmetric_validation/asymmetric_validation_production.json")
DYNAMIC_PATHS = (
    Path("output/benchmark/asymmetric_dynamic_1200.json"),
    Path("output/benchmark/asymmetric_dynamic_3000.json"),
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
    static, dynamics = load_inputs(STATIC_PATH, DYNAMIC_PATHS)
    validate_inputs(static, dynamics)
    assert len(static["records"]) == 720


def test_build_report_discloses_oracle_and_exact_prefix(tmp_path: Path) -> None:
    static, dynamics = load_inputs(STATIC_PATH, DYNAMIC_PATHS)
    output = tmp_path / "report.pdf"
    build_report(static, dynamics, output, tmp_path / "figures")
    reader = PdfReader(output)
    text = "\n".join(page.extract_text() or "" for page in reader.pages).lower()
    assert "oracle only" in text
    assert "escape angle" in text
    assert "kill-only" in text
    assert "exact prefix" in text
    assert "mathematically unavailable" not in text
    assert 5 <= len(reader.pages) <= 8
