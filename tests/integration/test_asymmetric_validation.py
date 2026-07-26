from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from scripts import run_asymmetric_validation as driver
from scripts.run_asymmetric_validation import (
    _load_or_create_targets,
    _methodology,
    merge_static_shards,
    pilot_decision,
    run_static_cell,
    write_static_shard,
)


def test_tiny_static_cell_has_reference_and_policy_metrics(tmp_path: Path) -> None:
    positions = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.0, 1.0]],
        dtype=np.float64,
    )

    record = run_static_cell(
        target_name="fixture",
        positions=positions,
        particle_radius=0.5,
        rho=3,
        replication=0,
        search_probes=4,
        evaluation_probes=8,
        seed=123,
        backend="reference",
    )

    assert record["complete"] is True
    assert set(record["policies"]) == {"uniform", "one-shot-centered"}
    assert record["reference"]["samples"] == 8
    assert record["policies"]["uniform"]["samples"] == 8
    assert record["policies"]["one-shot-centered"]["search_probes"] == 4
    assert set(record["seeds"]) == {
        "exact_evaluation",
        "uniform_evaluation",
        "center_search",
        "center_evaluation",
    }
    assert record["geometry"]["metric"]["launch_radius"] > 0.0
    assert record["geometry"]["metric"]["death_radius"] > 0.0
    assert record["geometry"]["metric"]["death_over_diameter"] > 0.0
    assert record["geometry"]["one_shot"]["launch_radius"] > 0.0
    assert record["provenance"]["python_version"]
    assert record["config"]["launch_margin"] == 4.0


def test_static_shards_are_no_clobber_and_merge_requires_completeness(tmp_path: Path) -> None:
    positions = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    record = run_static_cell(
        target_name="fixture",
        positions=positions,
        particle_radius=0.5,
        rho=3,
        replication=0,
        search_probes=2,
        evaluation_probes=4,
        seed=123,
        backend="reference",
    )
    first = write_static_shard(tmp_path, record)
    with pytest.raises(FileExistsError):
        write_static_shard(tmp_path, record)

    merged = merge_static_shards([first], expected_keys={("fixture", 3, 0)})
    assert merged["records"] == [record]

    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps({**record, "rho": 4, "complete": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        merge_static_shards([first, missing], expected_keys={("fixture", 3, 0), ("fixture", 4, 0)})


def test_incomplete_shard_requires_explicit_retry_and_can_then_be_replaced(tmp_path: Path) -> None:
    positions = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    complete = run_static_cell(
        target_name="fixture",
        positions=positions,
        particle_radius=0.5,
        rho=3,
        replication=0,
        search_probes=2,
        evaluation_probes=4,
        seed=123,
        backend="reference",
    )
    incomplete = {"target_name": "fixture", "rho": 3, "replication": 0, "complete": False}
    write_static_shard(tmp_path, incomplete)
    with pytest.raises(FileExistsError):
        write_static_shard(tmp_path, complete)
    replaced = write_static_shard(tmp_path, complete, replace_incomplete=True)
    assert json.loads(replaced.read_text(encoding="utf-8"))["complete"] is True

    stale = dict(complete)
    stale["replication"] = 1
    write_static_shard(
        tmp_path, {"target_name": "fixture", "rho": 3, "replication": 1, "complete": False}
    )
    lock = tmp_path / "fixture__rho3__rep1.json.retry-lock"
    lock.write_text("stale", encoding="utf-8")
    os.utime(lock, (0, 0))
    assert write_static_shard(tmp_path, stale, replace_incomplete=True).exists()


def test_merge_rejects_mixed_target_configuration(tmp_path: Path) -> None:
    positions = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    first = run_static_cell(
        target_name="fixture",
        positions=positions,
        particle_radius=0.5,
        rho=3,
        replication=0,
        search_probes=2,
        evaluation_probes=4,
        seed=123,
        backend="reference",
    )
    second = dict(first)
    second["rho"] = 4
    second["target_digest"] = "not-the-same-target"
    first_path = write_static_shard(tmp_path, first)
    second_path = write_static_shard(tmp_path, second)
    with pytest.raises(ValueError, match=r"target digest|configuration"):
        merge_static_shards(
            [first_path, second_path],
            expected_keys={("fixture", 3, 0), ("fixture", 4, 0)},
        )


def test_pilot_decision_and_methodology_are_explicit() -> None:
    assert _methodology()["policies"] == ("exact", "uniform", "one-shot-centered")
    decision = pilot_decision(
        [
            {
                "complete": True,
                "rho": 3,
                "reference": {"seconds": 0.1},
                "policies": {"uniform": {"particle_id_tv": 0.2}},
            }
        ],
        evaluation_probes=20_000,
    )
    assert decision["evaluation_probes"] == 20_000
    assert isinstance(decision["double_evaluation_counts"], bool)


def test_pilot_uses_reference_split_half_noise_against_uniform_discrepancy() -> None:
    decision = pilot_decision(
        [
            {
                "complete": True,
                "cell_seconds": 0.1,
                "reference": {"split_half_standard_error": 0.4},
                "policies": {
                    "uniform": {"particle_id_tv": 0.1, "seconds": 0.1},
                    "one-shot-centered": {"seconds": 0.1},
                },
            }
        ],
        evaluation_probes=10,
    )
    assert decision["max_reference_split_half_se_over_uniform"] == pytest.approx(4.0)
    assert decision["double_evaluation_counts"] is True


def test_pilot_probe_doubling_is_per_rho() -> None:
    def record(rho: int, noise: float) -> dict[str, object]:
        return {
            "complete": True,
            "rho": rho,
            "cell_seconds": 0.1,
            "config": {"search_probes": 5_000, "evaluation_probes": 20_000},
            "reference": {"split_half_standard_error": noise, "seconds": 0.1},
            "policies": {
                "uniform": {"particle_id_tv": 0.4, "seconds": 0.1},
                "one-shot-centered": {
                    "seconds": 0.1,
                    "search_seconds": 0.05,
                    "evaluation_seconds": 0.05,
                },
            },
        }

    decision = pilot_decision(
        [record(3, 0.01), record(4, 0.4)],
        evaluation_probes=20_000,
    )
    assert decision["by_rho"]["3"]["double_evaluation_counts"] is False
    assert decision["by_rho"]["4"]["double_evaluation_counts"] is True


def test_merge_records_shard_digest_inventory(tmp_path: Path) -> None:
    positions = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    record = run_static_cell(
        target_name="fixture",
        positions=positions,
        particle_radius=0.5,
        rho=3,
        replication=0,
        search_probes=2,
        evaluation_probes=4,
        seed=123,
        backend="reference",
    )
    path = write_static_shard(tmp_path, record)
    merged = merge_static_shards([path], expected_keys={("fixture", 3, 0)})
    assert merged["expected_shard_count"] == 1
    assert merged["shard_inventory"] == [
        {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    ]


def test_target_cache_hit_does_not_simulate_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _load_or_create_targets(tmp_path)

    def fail_simulate(*args: object, **kwargs: object) -> object:
        raise AssertionError("target cache hit unexpectedly simulated a target")

    monkeypatch.setattr(driver, "simulate", fail_simulate)
    second = _load_or_create_targets(tmp_path)
    assert sorted(first) == sorted(second)
    for name in first:
        np.testing.assert_array_equal(first[name][0], second[name][0])


def test_merge_rejects_mixed_phase_and_backend(tmp_path: Path) -> None:
    positions = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    first = run_static_cell(
        target_name="fixture",
        positions=positions,
        particle_radius=0.5,
        rho=3,
        replication=0,
        search_probes=2,
        evaluation_probes=4,
        seed=123,
        backend="reference",
        phase="pilot",
    )
    second = run_static_cell(
        target_name="fixture",
        positions=positions,
        particle_radius=0.5,
        rho=4,
        replication=0,
        search_probes=2,
        evaluation_probes=4,
        seed=123,
        backend="reference",
        phase="production",
    )
    first_path = write_static_shard(tmp_path, first)
    second_path = write_static_shard(tmp_path, second)
    with pytest.raises(ValueError, match="phase"):
        merge_static_shards(
            [first_path, second_path],
            expected_keys={("fixture", 3, 0), ("fixture", 4, 0)},
        )
