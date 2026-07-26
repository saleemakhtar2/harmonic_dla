from __future__ import annotations

import json
from pathlib import Path

import numba
import pytest

from harmonic_dla.cli import main
from harmonic_dla.models import SimulationDiagnostics, SimulationResult


def test_certificate_cli(capsys) -> None:
    assert main(["certificate", "self-centered", "--rho", "3"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert 0.0 < payload["value"] < 1.0


def test_check_config_cli(tmp_path: Path, capsys) -> None:
    path = tmp_path / "config.toml"
    path.write_text("particles = 4\n")
    assert main(["check-config", str(path)]) == 0
    assert "particles=4" in capsys.readouterr().out


def test_check_config_rejects_threads_above_numba_capacity(tmp_path: Path, capsys) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f"particles = 4\n[performance]\nthreads = {numba.get_num_threads() + 1}\n")
    assert main(["check-config", str(path)]) == 2
    assert "thread capacity" in capsys.readouterr().err


def test_inspect_incomplete_archive_returns_nonzero(tmp_path: Path, capsys) -> None:
    path = SimulationResult(
        positions=[[0.0, 0.0], [1.0, 0.0]],
        particle_radius=0.5,
        seed=1,
        backend="reference",
        restart_mode="exact_return",
        diagnostics=SimulationDiagnostics(),
        metadata={"complete": False, "final_center": [0.0, 0.0]},
    ).save(tmp_path / "checkpoint.npz")

    assert main(["inspect", str(path), "--json"]) != 0
    assert "incomplete" in capsys.readouterr().err.lower()


def test_summary_uses_persisted_final_center(tmp_path: Path, capsys) -> None:
    path = SimulationResult(
        positions=[[2.0, 0.0], [3.0, 0.0]],
        particle_radius=0.5,
        seed=1,
        backend="reference",
        restart_mode="exact_return",
        diagnostics=SimulationDiagnostics(),
        metadata={"complete": True, "final_center": [2.0, 0.0]},
    ).save(tmp_path / "result.npz")

    assert main(["inspect", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["target_radius_about_final_center"] == 2.0


def test_run_rejects_existing_output_before_simulating(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    config_path = tmp_path / "config.toml"
    output = tmp_path / "result.npz"
    output.write_bytes(b"keep me")
    config_path.write_text(f'particles = 2\nbackend = "reference"\n[output]\npath = "{output}"\n')
    monkeypatch.setattr("harmonic_dla.cli.simulate", lambda _config: pytest.fail("simulated"))
    assert main(["run", str(config_path)]) == 2
    assert output.read_bytes() == b"keep me"
    assert "overwrite" in capsys.readouterr().err.lower()
