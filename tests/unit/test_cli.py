from __future__ import annotations

import json
from pathlib import Path

from harmonic_dla.cli import main


def test_certificate_cli(capsys) -> None:
    assert main(["certificate", "self-centered", "--rho", "3"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert 0.0 < payload["value"] < 1.0


def test_check_config_cli(tmp_path: Path, capsys) -> None:
    path = tmp_path / "config.toml"
    path.write_text("particles = 4\n")
    assert main(["check-config", str(path)]) == 0
    assert "particles=4" in capsys.readouterr().out
