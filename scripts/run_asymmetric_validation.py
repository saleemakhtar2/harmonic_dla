"""Run the preregistered asymmetric-target finite-boundary validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from harmonic_dla.analysis import bounding_box_diameter_upper
from harmonic_dla.api import probe_detailed, simulate
from harmonic_dla.config import RunConfig
from harmonic_dla.enums import RestartMode
from harmonic_dla.models import ProbeResult
from harmonic_dla.provenance import runtime_provenance
from harmonic_dla.validation import (
    ANGLE_BINS,
    PRIMARY_RHOS,
    assign_nearest_disk,
    bounded_lipschitz_discrepancy,
    histogram_total_variation,
    make_lopsided_comb,
    particle_id_total_variation,
    schedule_seed,
)

EXPERIMENT_VERSION = "asymmetric-v1"
STATIC_RHOS = (2, 3, 4, 6, 8, 12, 16, 24, 32)
STATIC_POLICIES = ("exact", "uniform", "one-shot-centered")
TARGET_SEEDS = (42, 137, 911)
LAUNCH_MARGIN = 4.0


def _target_digest(positions: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(positions, dtype=np.float64).tobytes()).hexdigest()


def _methodology() -> dict[str, Any]:
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "ratios": STATIC_RHOS,
        "primary_fit_window": PRIMARY_RHOS,
        "policies": STATIC_POLICIES,
        "angle_bins": ANGLE_BINS,
        "metric_note": "finite partitions and fixed bounded-Lipschitz diagnostics; not continuum-TV proofs",
    }


def _batch_seed(
    phase: str,
    target_name: str,
    policy: str,
    rho: int,
    replication: int,
    batch_kind: str,
) -> int:
    """Derive every stream from the immutable preregistered tuple."""
    return schedule_seed(
        (EXPERIMENT_VERSION, phase, target_name, policy, int(rho), int(replication), batch_kind)
    )


def _boundary_geometry(
    positions: np.ndarray,
    center: tuple[float, float],
    particle_radius: float,
    rho: int,
) -> dict[str, float | list[float]]:
    diameter = bounding_box_diameter_upper(positions, particle_radius)
    target_radius = _target_radius(positions, center, particle_radius)
    launch_radius = target_radius + LAUNCH_MARGIN * particle_radius
    death_radius = max(
        float(rho) * target_radius,
        launch_radius * (1.0 + 1.0e-12),
    )
    return {
        "center": [float(center[0]), float(center[1])],
        "target_radius": target_radius,
        "launch_radius": launch_radius,
        "death_radius": death_radius,
        "diameter_upper": diameter,
        "death_over_diameter": death_radius / max(diameter, np.finfo(float).eps),
    }


def _record_config(
    particle_radius: float,
    backend: str,
    *,
    phase: str,
    rho: int,
    search_probes: int,
    evaluation_probes: int,
) -> dict[str, Any]:
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "backend": backend,
        "phase": phase,
        "rho": int(rho),
        "search_probes": int(search_probes),
        "evaluation_probes": int(evaluation_probes),
        "particle_radius": float(particle_radius),
        "launch_margin": LAUNCH_MARGIN,
        "angle_bins": list(ANGLE_BINS),
        "primary_fit_window": list(PRIMARY_RHOS),
    }


def _target_radius(
    positions: np.ndarray, center: tuple[float, float], particle_radius: float
) -> float:
    offsets = positions - np.asarray(center, dtype=np.float64)
    return float(np.sqrt(np.max(np.sum(offsets * offsets, axis=1))) + 2.0 * particle_radius)


def _probe_summary(
    samples: ProbeResult,
    reference: ProbeResult,
    positions: np.ndarray,
    metric_center: tuple[float, float],
    particle_radius: float,
) -> dict[str, Any]:
    reference_ids = assign_nearest_disk(reference.attachments, positions)
    sample_ids = assign_nearest_disk(samples.attachments, positions)
    diameter = bounding_box_diameter_upper(positions, particle_radius)
    reference_mean = np.mean(reference.attachments, axis=0)
    sample_mean = np.mean(samples.attachments, axis=0)
    return {
        "samples": int(samples.attachments.shape[0]),
        "walker_steps": int(samples.walker_steps),
        "restarts": int(samples.restarts),
        "particle_id_tv": particle_id_total_variation(
            reference_ids,
            sample_ids,
            particle_count=positions.shape[0],
        ),
        "angle_tv": {
            str(bins): histogram_total_variation(
                reference.attachments,
                samples.attachments,
                bins=bins,
                center=metric_center,
            )
            for bins in ANGLE_BINS
        },
        "bounded_lipschitz": bounded_lipschitz_discrepancy(
            reference.attachments,
            samples.attachments,
            metric_center,
            clip_radius=diameter,
        ),
        "barycenter_error_over_diameter": float(
            np.linalg.norm(sample_mean - reference_mean) / max(diameter, np.finfo(float).eps)
        ),
    }


def _reference_split_half_noise(
    reference: ProbeResult,
    positions: np.ndarray,
) -> float:
    """Measure reference sampling noise without consulting policy outputs."""
    ids = assign_nearest_disk(reference.attachments, positions)
    midpoint = ids.shape[0] // 2
    if midpoint < 1 or midpoint == ids.shape[0]:
        return 0.0
    split_half_tv = particle_id_total_variation(
        ids[:midpoint],
        ids[midpoint:],
        particle_count=positions.shape[0],
    )
    # Two disjoint half-sample estimates contribute independent variance; the
    # corresponding standard error is the half-difference scaled by sqrt(2).
    return float(split_half_tv / np.sqrt(2.0))


def _run_probe(
    positions: np.ndarray,
    particle_radius: float,
    center: tuple[float, float],
    rho: int,
    probes: int,
    seed: int,
    backend: str,
    restart_mode: RestartMode,
) -> tuple[ProbeResult, float]:
    started = time.perf_counter()
    result = probe_detailed(
        positions,
        particle_radius=particle_radius,
        center=center,
        death_ratio=float(rho),
        launch_margin=4.0,
        probes=probes,
        seed=seed,
        backend=backend,
        restart_mode=restart_mode,
    )
    return result, time.perf_counter() - started


def run_static_cell(
    *,
    target_name: str,
    positions: np.ndarray,
    particle_radius: float,
    rho: int,
    replication: int,
    search_probes: int,
    evaluation_probes: int,
    seed: int,
    backend: str = "numba-cpu",
    phase: str = "pilot",
) -> dict[str, Any]:
    """Run one complete exact/uniform/centered frozen-target comparison."""
    target = np.ascontiguousarray(positions, dtype=np.float64)
    metric_center = (float(target[0, 0]), float(target[0, 1]))
    batch_seeds = {
        "exact_evaluation": _batch_seed(
            phase, target_name, "exact", rho, replication, "evaluation"
        ),
        "uniform_evaluation": _batch_seed(
            phase, target_name, "uniform", rho, replication, "evaluation"
        ),
        "center_search": _batch_seed(
            phase, target_name, "one-shot-centered", rho, replication, "search"
        ),
        "center_evaluation": _batch_seed(
            phase, target_name, "one-shot-centered", rho, replication, "evaluation"
        ),
    }
    reference, reference_seconds = _run_probe(
        target,
        particle_radius,
        metric_center,
        rho,
        evaluation_probes,
        batch_seeds["exact_evaluation"],
        backend,
        RestartMode.EXACT_RETURN,
    )
    uniform, uniform_seconds = _run_probe(
        target,
        particle_radius,
        metric_center,
        rho,
        evaluation_probes,
        batch_seeds["uniform_evaluation"],
        backend,
        RestartMode.UNIFORM_RESTART,
    )
    search, search_seconds = _run_probe(
        target,
        particle_radius,
        metric_center,
        rho,
        search_probes,
        batch_seeds["center_search"],
        backend,
        RestartMode.UNIFORM_RESTART,
    )
    estimated_mean = np.mean(search.attachments, axis=0)
    estimated_center = (float(estimated_mean[0]), float(estimated_mean[1]))
    centered, centered_seconds = _run_probe(
        target,
        particle_radius,
        estimated_center,
        rho,
        evaluation_probes,
        batch_seeds["center_evaluation"],
        backend,
        RestartMode.UNIFORM_RESTART,
    )
    metric_geometry = _boundary_geometry(target, metric_center, particle_radius, rho)
    one_shot_geometry = _boundary_geometry(target, estimated_center, particle_radius, rho)
    reference_split_half_noise = _reference_split_half_noise(reference, target)
    total_seconds = reference_seconds + uniform_seconds + search_seconds + centered_seconds
    replication_seed = _batch_seed(phase, target_name, "record", rho, replication, "record")
    return {
        "schema_version": 1,
        "experiment_version": EXPERIMENT_VERSION,
        "phase": phase,
        "target_name": target_name,
        "target_digest": _target_digest(target),
        "target_particle_count": int(target.shape[0]),
        "particle_radius": float(particle_radius),
        "rho": int(rho),
        "primary_fit_window": int(rho) in PRIMARY_RHOS,
        "replication": int(replication),
        "requested_seed": int(seed),
        "replication_seed": replication_seed,
        "complete": True,
        "backend": backend,
        "seeds": batch_seeds,
        "provenance": runtime_provenance(),
        "config": _record_config(
            particle_radius,
            backend,
            phase=phase,
            rho=rho,
            search_probes=search_probes,
            evaluation_probes=evaluation_probes,
        ),
        "geometry": {
            "metric_center": list(metric_center),
            "one_shot_center": list(estimated_center),
            "metric": metric_geometry,
            "one_shot": one_shot_geometry,
            "target_radius_metric_center": metric_geometry["target_radius"],
            "target_radius_one_shot_center": one_shot_geometry["target_radius"],
            "diameter_upper": bounding_box_diameter_upper(target, particle_radius),
        },
        "reference": {
            "samples": int(reference.attachments.shape[0]),
            "walker_steps": int(reference.walker_steps),
            "restarts": int(reference.restarts),
            "seconds": reference_seconds,
            "split_half_particle_id_tv": reference_split_half_noise * np.sqrt(2.0),
            "split_half_standard_error": reference_split_half_noise,
            "seed": batch_seeds["exact_evaluation"],
        },
        "cell_seconds": total_seconds,
        "policies": {
            "uniform": {
                **_probe_summary(uniform, reference, target, metric_center, particle_radius),
                "seconds": uniform_seconds,
                "calibration_probes": 0,
                "seed": batch_seeds["uniform_evaluation"],
            },
            "one-shot-centered": {
                **_probe_summary(centered, reference, target, metric_center, particle_radius),
                "seconds": centered_seconds + search_seconds,
                "search_seconds": search_seconds,
                "evaluation_seconds": centered_seconds,
                "search_probes": int(search.attachments.shape[0]),
                "calibration_probes": int(search.attachments.shape[0]),
                "center_residual_over_diameter": float(
                    np.linalg.norm(np.asarray(estimated_center) - np.asarray(metric_center))
                    / max(
                        bounding_box_diameter_upper(target, particle_radius),
                        np.finfo(float).eps,
                    )
                ),
                "seed": batch_seeds["center_evaluation"],
            },
        },
    }


def _shard_name(record: dict[str, Any]) -> str:
    target = str(record["target_name"]).replace("/", "_")
    return f"{target}__rho{int(record['rho'])}__rep{int(record['replication'])}.json"


def write_static_shard(
    output_dir: Path,
    record: dict[str, Any],
    *,
    replace_incomplete: bool = False,
) -> Path:
    """Create a shard atomically, optionally retrying a known incomplete shard."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / _shard_name(record)
    if path.exists():
        if not replace_incomplete:
            raise FileExistsError(path)
        lock = path.with_suffix(path.suffix + ".retry-lock")
        try:
            lock_descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError as exc:
            try:
                stale = time.time() - lock.stat().st_mtime > 3_600.0
            except OSError:
                stale = False
            if not stale:
                raise FileExistsError(
                    f"incomplete shard retry already in progress: {path}"
                ) from exc
            lock.unlink(missing_ok=True)
            lock_descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"cannot replace invalid existing shard: {path}") from exc
            if not isinstance(existing, dict):
                raise ValueError(f"cannot replace invalid existing shard: {path}")
            if existing.get("complete") is True:
                raise FileExistsError(path)
            with os.fdopen(lock_descriptor, "wb"):
                lock_descriptor = -1
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output_dir,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                try:
                    stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
            temporary.replace(path)
            return path
        finally:
            if lock_descriptor >= 0:
                os.close(lock_descriptor)
            lock.unlink(missing_ok=True)
    payload = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        raise
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def _validate_static_record(
    record: dict[str, Any],
    *,
    expected_phase: str | None = None,
    expected_backend: str | None = None,
    expected_config: dict[str, Any] | None = None,
) -> None:
    required = {
        "schema_version",
        "experiment_version",
        "phase",
        "target_name",
        "target_digest",
        "target_particle_count",
        "particle_radius",
        "rho",
        "primary_fit_window",
        "replication",
        "requested_seed",
        "replication_seed",
        "complete",
        "cell_seconds",
        "backend",
        "seeds",
        "provenance",
        "config",
        "geometry",
        "reference",
        "policies",
    }
    missing = sorted(required.difference(record))
    if missing:
        raise ValueError(f"record missing required fields: {', '.join(missing)}")
    if record["schema_version"] != 1 or record["experiment_version"] != EXPERIMENT_VERSION:
        raise ValueError("record schema/configuration mismatch")
    if not isinstance(record["phase"], str) or record["phase"] not in {"pilot", "production"}:
        raise ValueError("record phase is invalid")
    if expected_phase is not None and record["phase"] != expected_phase:
        raise ValueError("record phase does not match the requested phase")
    if record["complete"] is not True:
        raise ValueError("incomplete static shard")
    try:
        cell_seconds = float(record["cell_seconds"])
    except (TypeError, ValueError) as exc:
        raise ValueError("record timing diagnostics are invalid") from exc
    if not np.isfinite(cell_seconds) or cell_seconds < 0.0:
        raise ValueError("record timing diagnostics are invalid")
    if not isinstance(record["target_name"], str) or not record["target_name"]:
        raise ValueError("target_name must be a non-empty string")
    if not isinstance(record["target_digest"], str) or len(record["target_digest"]) != 64:
        raise ValueError("target digest must be a SHA-256 hex digest")
    if any(character not in "0123456789abcdefABCDEF" for character in record["target_digest"]):
        raise ValueError("target digest must be a SHA-256 hex digest")
    integer_fields = (
        "target_particle_count",
        "rho",
        "replication",
        "requested_seed",
        "replication_seed",
    )
    if any(
        isinstance(record[field], bool) or not isinstance(record[field], int)
        for field in integer_fields
    ):
        raise ValueError("record geometry identifiers are invalid")
    particle_count = int(record["target_particle_count"])
    radius = float(record["particle_radius"])
    rho = int(record["rho"])
    replication = int(record["replication"])
    requested_seed = int(record["requested_seed"])
    replication_seed = int(record["replication_seed"])
    if particle_count < 1 or not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("record target geometry is invalid")
    if rho not in STATIC_RHOS or replication < 0 or requested_seed < 0 or replication_seed < 0:
        raise ValueError("record rho or replication is outside the preregistered design")
    if replication_seed != _batch_seed(
        record["phase"], record["target_name"], "record", rho, replication, "record"
    ):
        raise ValueError("record replication seed does not match the preregistered seed tuple")
    if not isinstance(record["primary_fit_window"], bool):
        raise ValueError("primary_fit_window must be boolean")
    if record["primary_fit_window"] != (rho in PRIMARY_RHOS):
        raise ValueError("primary_fit_window does not match rho")
    if not isinstance(record["backend"], str) or record["backend"] not in {
        "reference",
        "numba-cpu",
    }:
        raise ValueError("record backend is invalid")
    if expected_backend is not None and record["backend"] != expected_backend:
        raise ValueError("record backend does not match the requested backend")
    if not isinstance(record["config"], dict):
        raise ValueError("record configuration is invalid")
    config_integer_fields = ("rho", "search_probes", "evaluation_probes")
    if any(
        isinstance(record["config"].get(field), bool)
        or not isinstance(record["config"].get(field), int)
        or int(record["config"].get(field)) <= 0
        for field in config_integer_fields
    ):
        raise ValueError("record configuration probe counts are invalid")
    config_expected = _record_config(
        radius,
        record["backend"],
        phase=record["phase"],
        rho=rho,
        search_probes=int(record["config"]["search_probes"]),
        evaluation_probes=int(record["config"]["evaluation_probes"]),
    )
    if record["config"] != config_expected:
        raise ValueError("record configuration does not match the preregistered design")
    if expected_config is not None and any(
        record["config"].get(key) != value for key, value in expected_config.items()
    ):
        raise ValueError("record configuration does not match the requested run")
    if not isinstance(record["policies"], dict) or set(record["policies"]) != {
        "uniform",
        "one-shot-centered",
    }:
        raise ValueError("record policy set is incomplete")
    reference = record["reference"]
    uniform = record["policies"]["uniform"]
    centered = record["policies"]["one-shot-centered"]
    if (
        not isinstance(reference, dict)
        or not isinstance(uniform, dict)
        or not isinstance(centered, dict)
    ):
        raise ValueError("record sampling diagnostics are incomplete")
    expected_evaluations = int(record["config"]["evaluation_probes"])
    expected_search = int(record["config"]["search_probes"])
    if (
        reference.get("samples") != expected_evaluations
        or uniform.get("samples") != expected_evaluations
        or centered.get("samples") != expected_evaluations
        or centered.get("search_probes") != expected_search
        or centered.get("calibration_probes") != expected_search
    ):
        raise ValueError("record sample/search counts do not match configuration")
    try:
        uniform_tv = float(uniform["particle_id_tv"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("record uniform discrepancy is incomplete") from exc
    if not np.isfinite(uniform_tv) or uniform_tv < 0.0:
        raise ValueError("record uniform discrepancy is invalid")
    timing_fields = (
        (reference, "seconds"),
        (uniform, "seconds"),
        (centered, "seconds"),
        (centered, "search_seconds"),
        (centered, "evaluation_seconds"),
    )
    for source, field in timing_fields:
        try:
            value = float(source[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("record timing diagnostics are incomplete") from exc
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("record timing diagnostics are invalid")
    try:
        split_half_se = float(reference["split_half_standard_error"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("record reference split-half diagnostics are incomplete") from exc
    if not np.isfinite(split_half_se) or split_half_se < 0.0:
        raise ValueError("record reference split-half diagnostics are invalid")
    if (
        not isinstance(record["seeds"], dict)
        or set(record["seeds"])
        != {
            "exact_evaluation",
            "uniform_evaluation",
            "center_search",
            "center_evaluation",
        }
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in record["seeds"].values()
        )
    ):
        raise ValueError("record batch seeds are incomplete or invalid")
    expected_seeds = {
        "exact_evaluation": _batch_seed(
            str(record["phase"]),
            str(record["target_name"]),
            "exact",
            rho,
            replication,
            "evaluation",
        ),
        "uniform_evaluation": _batch_seed(
            str(record["phase"]),
            str(record["target_name"]),
            "uniform",
            rho,
            replication,
            "evaluation",
        ),
        "center_search": _batch_seed(
            str(record["phase"]),
            str(record["target_name"]),
            "one-shot-centered",
            rho,
            replication,
            "search",
        ),
        "center_evaluation": _batch_seed(
            str(record["phase"]),
            str(record["target_name"]),
            "one-shot-centered",
            rho,
            replication,
            "evaluation",
        ),
    }
    if record["seeds"] != expected_seeds:
        raise ValueError("record batch seeds do not match the preregistered seed tuple")
    if not isinstance(record["geometry"], dict):
        raise ValueError("record geometry is incomplete")
    for key in ("metric", "one_shot"):
        geometry = record["geometry"].get(key)
        if not isinstance(geometry, dict) or any(
            field not in geometry
            for field in (
                "center",
                "target_radius",
                "launch_radius",
                "death_radius",
                "diameter_upper",
                "death_over_diameter",
            )
        ):
            raise ValueError("record geometry is incomplete")
        center = geometry["center"]
        if (
            not isinstance(center, list)
            or len(center) != 2
            or any(not np.isfinite(float(value)) for value in center)
        ):
            raise ValueError("record geometry center is invalid")
        if any(not np.isfinite(float(geometry[field])) for field in geometry if field != "center"):
            raise ValueError("record geometry contains non-finite values")
    if not isinstance(record["provenance"], dict) or "source_revision" not in record["provenance"]:
        raise ValueError("record provenance is incomplete")


def merge_static_shards(
    paths: list[Path],
    *,
    expected_keys: set[tuple[str, int, int]],
    expected_phase: str | None = None,
    expected_backend: str | None = None,
    expected_config: dict[str, Any] | None = None,
    expected_shard_digests: dict[str, str] | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    shard_inventory: list[dict[str, Any]] = []
    phase = expected_phase
    backend = expected_backend
    for path in paths:
        try:
            payload = path.read_bytes()
            item = json.loads(payload.decode("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid static shard: {path}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"static shard must contain an object: {path}")
        if phase is None:
            phase = item.get("phase")
        if backend is None:
            backend = item.get("backend")
        _validate_static_record(
            item,
            expected_phase=phase,
            expected_backend=backend,
            expected_config=expected_config,
        )
        digest = hashlib.sha256(payload).hexdigest()
        inventory_entry = {"path": str(path), "sha256": digest}
        if expected_shard_digests is not None:
            expected_digest = expected_shard_digests.get(str(path))
            if expected_digest is None or expected_digest != digest:
                raise ValueError(f"shard digest mismatch: {path}")
        shard_inventory.append(inventory_entry)
        records.append(item)
    actual_keys = {
        (str(item["target_name"]), int(item["rho"]), int(item["replication"])) for item in records
    }
    if actual_keys != expected_keys:
        raise ValueError("incomplete static shard set")
    if len(records) != len(actual_keys):
        raise ValueError("duplicate static shard key")
    digests: dict[str, str] = {}
    configs: dict[str, Any] = {}
    for item in records:
        name = str(item["target_name"])
        digest = str(item["target_digest"])
        if name in digests and digests[name] != digest:
            raise ValueError("mixed target digest/configuration")
        config = item["config"]
        global_config = {
            key: value
            for key, value in config.items()
            if key not in {"rho", "search_probes", "evaluation_probes"}
        }
        if name in configs and configs[name] != global_config:
            raise ValueError("mixed target configuration")
        digests[name] = digest
        configs[name] = global_config
    if phase is None or backend is None:
        raise ValueError("static shard set has no phase/backend")
    shard_inventory.sort(key=lambda entry: entry["path"])
    return {
        "schema_version": 1,
        "methodology": _methodology(),
        "phase": phase,
        "backend": backend,
        "expected_shard_count": len(expected_keys),
        "shard_inventory": shard_inventory,
        "records": sorted(
            records,
            key=lambda item: (str(item["target_name"]), int(item["rho"]), int(item["replication"])),
        ),
    }


def pilot_decision(
    records: list[dict[str, Any]],
    *,
    evaluation_probes: int,
    production_search_probes: int = 8_192,
    production_evaluation_probes: int = 25_000,
) -> dict[str, Any]:
    """Apply the preregistered per-rho feasibility and noise rule."""
    complete = [item for item in records if item.get("complete") is True]
    if not complete:
        raise ValueError("pilot contains no complete records")
    groups: dict[int, list[dict[str, Any]]] = {}
    for item in complete:
        try:
            rho = int(item.get("rho", 0))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("pilot record lacks a valid rho") from exc
        groups.setdefault(rho, []).append(item)

    def group_decision(group: list[dict[str, Any]]) -> dict[str, Any]:
        runtimes: list[float] = []
        projected_runtimes: list[float] = []
        discrepancies: list[float] = []
        reference_noise: list[float] = []
        for item in group:
            policies = item.get("policies")
            reference = item.get("reference")
            config = item.get("config")
            if not isinstance(policies, dict) or not isinstance(reference, dict):
                raise ValueError("pilot record lacks reference/policy diagnostics")
            uniform = policies.get("uniform")
            centered = policies.get("one-shot-centered")
            if not isinstance(uniform, dict):
                raise ValueError("pilot record lacks policy diagnostics")
            if not isinstance(centered, dict):
                centered = {}
            if not isinstance(config, dict):
                # Keep the helper useful for compact diagnostic fixtures; full
                # manifests are validated strictly before reaching this path.
                pilot_search = 5_000
                pilot_eval = int(evaluation_probes)
            else:
                pilot_search = int(config["search_probes"])
                pilot_eval = int(config["evaluation_probes"])
            runtime = float(item.get("cell_seconds", 0.0))
            try:
                discrepancy = float(uniform["particle_id_tv"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("pilot record lacks uniform discrepancy") from exc
            noise = float(
                reference.get(
                    "split_half_standard_error",
                    reference.get("split_half_particle_id_tv", 0.0),
                )
            )
            projected = (
                float(reference.get("seconds", 0.0)) * production_evaluation_probes / pilot_eval
                + float(uniform.get("seconds", 0.0)) * production_evaluation_probes / pilot_eval
                + float(centered.get("search_seconds", 0.0))
                * production_search_probes
                / pilot_search
                + float(centered.get("evaluation_seconds", 0.0))
                * production_evaluation_probes
                / pilot_eval
            )
            if not all(
                np.isfinite(value) and value >= 0.0
                for value in (runtime, discrepancy, noise, projected)
            ):
                raise ValueError("pilot diagnostics contain non-finite or negative values")
            runtimes.append(runtime)
            projected_runtimes.append(projected)
            discrepancies.append(discrepancy)
            reference_noise.append(noise)
        ratios = [
            noise / max(discrepancy, np.finfo(float).eps)
            for noise, discrepancy in zip(reference_noise, discrepancies, strict=True)
        ]
        max_reference_noise = max(ratios)
        max_projected_runtime = max(projected_runtimes)
        return {
            "complete_records": len(group),
            "max_cell_seconds": max(runtimes),
            "max_projected_production_seconds": max_projected_runtime,
            "max_reference_split_half_se_over_uniform": max_reference_noise,
            "split_half_noise_fraction": max_reference_noise,
            "double_evaluation_counts": bool(
                max_projected_runtime > 120.0 or max_reference_noise > 0.25
            ),
        }

    by_rho = {str(rho): group_decision(group) for rho, group in sorted(groups.items())}
    double = any(item["double_evaluation_counts"] for item in by_rho.values())
    max_noise = max(item["max_reference_split_half_se_over_uniform"] for item in by_rho.values())
    return {
        "complete_records": len(complete),
        "by_rho": by_rho,
        "max_reference_split_half_se_over_uniform": max_noise,
        "split_half_noise_fraction": max_noise,
        "evaluation_probes": int(evaluation_probes) * (2 if double else 1),
        "base_evaluation_probes": int(evaluation_probes),
        "double_evaluation_counts": double,
        "rule": "double per rho if projected runtime > 120s or reference split-half SE > 25% of uniform discrepancy",
    }


def _target_cache_path(cache_dir: Path, name: str) -> Path:
    return cache_dir / f"{name.replace('/', '_')}.npz"


def _read_target_cache(
    path: Path, *, expected_radius: float | None = None
) -> tuple[np.ndarray, float]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            cached = np.asarray(archive["positions"], dtype=np.float64)
            cached_radius = float(archive["particle_radius"].item())
            cached_digest = str(archive["digest"].item())
    except Exception as exc:
        raise ValueError(f"invalid target cache entry: {path}") from exc
    if cached.ndim != 2 or cached.shape[1] != 2 or cached.shape[0] < 1:
        raise ValueError(f"invalid target cache geometry: {path}")
    if not np.isfinite(cached).all() or not np.isfinite(cached_radius) or cached_radius <= 0.0:
        raise ValueError(f"invalid target cache geometry: {path}")
    if cached_digest != _target_digest(cached):
        raise ValueError(f"target cache digest mismatch: {path.stem}")
    if expected_radius is not None and cached_radius != float(expected_radius):
        raise ValueError(f"target cache radius mismatch: {path.stem}")
    return np.ascontiguousarray(cached), cached_radius


def _write_target_cache(
    path: Path,
    positions: np.ndarray,
    radius: float,
) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        positions=np.ascontiguousarray(positions, dtype=np.float64),
        particle_radius=np.asarray(radius, dtype=np.float64),
        digest=np.asarray(_target_digest(positions)),
    )
    temporary.replace(path)


def _load_or_create_targets(output_dir: Path) -> dict[str, tuple[np.ndarray, float]]:
    """Load target cache entries first, simulating only targets that are missing."""
    cache_dir = output_dir / "targets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    targets: dict[str, tuple[np.ndarray, float]] = {}
    for name in ("lopsided-comb", *(f"dla-{seed}" for seed in TARGET_SEEDS)):
        path = _target_cache_path(cache_dir, name)
        if path.exists():
            targets[name] = _read_target_cache(
                path,
                expected_radius=0.5 if name == "lopsided-comb" else None,
            )
            continue
        if name == "lopsided-comb":
            positions, radius = make_lopsided_comb(), 0.5
        else:
            seed = int(name.removeprefix("dla-"))
            result = simulate(RunConfig(particles=256, seed=seed))
            positions, radius = result.positions, result.particle_radius
        _write_target_cache(path, positions, radius)
        targets[name] = (positions, radius)
    return targets


def _target_manifest(targets: dict[str, tuple[np.ndarray, float]]) -> dict[str, dict[str, Any]]:
    """Return the compact target identity used to bind pilot and production."""
    return {
        name: {
            "particle_radius": float(radius),
            "digest": _target_digest(positions),
            "particle_count": int(positions.shape[0]),
        }
        for name, (positions, radius) in sorted(targets.items())
    }


def _json_methodology(value: dict[str, Any]) -> dict[str, Any]:
    """Normalize tuple/list differences introduced by JSON serialization."""
    if not isinstance(value, dict):
        raise ValueError("methodology must be an object")
    return {
        key: list(item) if isinstance(item, (tuple, list)) else item for key, item in value.items()
    }


def _load_pilot_manifest(
    output_dir: Path,
    *,
    backend: str,
    targets: dict[str, tuple[np.ndarray, float]],
) -> dict[str, Any]:
    path = output_dir / "asymmetric_validation_pilot.json"
    if not path.exists():
        raise ValueError(
            "production requires asymmetric_validation_pilot.json; run the pilot first"
        )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid pilot manifest: {path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("pilot manifest must contain an object")
    try:
        methodology = _json_methodology(manifest.get("methodology", {}))
    except ValueError as exc:
        raise ValueError("production pilot methodology/configuration mismatch") from exc
    if methodology != _json_methodology(_methodology()):
        raise ValueError("production pilot methodology/configuration mismatch")
    if manifest.get("phase") != "pilot":
        raise ValueError("pilot manifest has the wrong phase")
    if isinstance(manifest.get("shard_inventory"), list):
        for entry in manifest["shard_inventory"]:
            if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                shard_path = Path(entry["path"])
                if not shard_path.is_absolute():
                    entry["path"] = str((output_dir / shard_path).resolve())
    _verify_shard_inventory(manifest)
    environment = manifest.get("environment")
    if not isinstance(environment, dict) or environment.get("backend") != backend:
        raise ValueError("production backend does not match the pilot manifest")
    expected_targets = _target_manifest(targets)
    recorded_targets = manifest.get("target_identity")
    if recorded_targets != expected_targets:
        raise ValueError("production target digest/configuration does not match the pilot manifest")
    records: list[dict[str, Any]] = []
    expected_keys = {
        (name, rho, replication)
        for name in expected_targets
        for rho in STATIC_RHOS
        for replication in range(5)
    }
    inventory = manifest["shard_inventory"]
    for entry in inventory:
        path = Path(entry["path"])
        if not path.is_absolute():
            path = (output_dir / path).resolve()
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid pilot shard: {path}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"invalid pilot shard: {path}")
        _validate_static_record(
            item,
            expected_phase="pilot",
            expected_backend=backend,
            expected_config={
                "phase": "pilot",
                "backend": backend,
                "search_probes": 5_000,
                "evaluation_probes": 20_000,
            },
        )
        target_identity = expected_targets.get(str(item["target_name"]))
        if target_identity is None:
            raise ValueError("pilot shard target is not in the expected target manifest")
        if (
            item["target_digest"] != target_identity["digest"]
            or item["target_particle_count"] != target_identity["particle_count"]
            or item["particle_radius"] != target_identity["particle_radius"]
        ):
            raise ValueError("pilot shard target digest/configuration mismatch")
        records.append(item)
    actual_keys = {
        (str(item["target_name"]), int(item["rho"]), int(item["replication"])) for item in records
    }
    if actual_keys != expected_keys or len(records) != len(expected_keys):
        raise ValueError("pilot manifest shard keys do not match the preregistered design")
    recomputed = pilot_decision(
        records,
        evaluation_probes=20_000,
        production_search_probes=8_192,
        production_evaluation_probes=25_000,
    )
    decision = manifest.get("pilot_decision")
    if not isinstance(decision, dict) or not isinstance(
        decision.get("double_evaluation_counts"), bool
    ):
        raise ValueError("pilot manifest has no valid probe-count decision")
    if decision != recomputed:
        raise ValueError("pilot manifest decision does not match verified pilot shards")
    by_rho = decision.get("by_rho")
    if not isinstance(by_rho, dict) or set(by_rho) != {str(rho) for rho in STATIC_RHOS}:
        raise ValueError("pilot manifest has no complete per-rho probe-count decision")
    if any(
        not isinstance(item, dict) or not isinstance(item.get("double_evaluation_counts"), bool)
        for item in by_rho.values()
    ):
        raise ValueError("pilot manifest has invalid per-rho probe-count decision")
    return manifest


def _verify_shard_inventory(manifest: dict[str, Any]) -> None:
    entries = manifest.get("shard_inventory")
    expected_count = manifest.get("expected_shard_count")
    if not isinstance(entries, list) or not isinstance(expected_count, int):
        raise ValueError("pilot manifest shard inventory is missing")
    if len(entries) != expected_count:
        raise ValueError("pilot manifest shard inventory count mismatch")
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("pilot manifest shard inventory is invalid")
        path_value = entry.get("path")
        expected_digest = entry.get("sha256")
        if not isinstance(path_value, str) or not isinstance(expected_digest, str):
            raise ValueError("pilot manifest shard inventory is invalid")
        path = Path(path_value)
        try:
            actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ValueError(f"pilot shard is unavailable: {path}") from exc
        if actual_digest != expected_digest:
            raise ValueError(f"pilot shard digest mismatch: {path}")


def _targets() -> dict[str, tuple[np.ndarray, float]]:
    targets: dict[str, tuple[np.ndarray, float]] = {
        "lopsided-comb": (make_lopsided_comb(), 0.5),
    }
    for seed in TARGET_SEEDS:
        result = simulate(RunConfig(particles=256, seed=seed))
        targets[f"dla-{seed}"] = (result.positions, result.particle_radius)
    return targets


def run_phase(
    phase: str,
    output_dir: Path,
    backend: str,
    *,
    retry_incomplete: bool = False,
) -> Path:
    if phase not in {"pilot", "production"}:
        raise ValueError("phase must be pilot or production")
    output_dir.mkdir(parents=True, exist_ok=True)
    if phase == "production" and not (output_dir / "asymmetric_validation_pilot.json").exists():
        raise ValueError(
            "production requires asymmetric_validation_pilot.json; run the pilot first"
        )
    repetitions = 5 if phase == "pilot" else 20
    search_probes = 5_000 if phase == "pilot" else 8_192
    base_evaluation_probes = 20_000 if phase == "pilot" else 25_000
    targets = _load_or_create_targets(output_dir)
    pilot_manifest = (
        None
        if phase == "pilot"
        else _load_pilot_manifest(output_dir, backend=backend, targets=targets)
    )
    pilot_by_rho = (
        None if pilot_manifest is None else pilot_manifest["pilot_decision"].get("by_rho")
    )
    if pilot_by_rho is not None and set(pilot_by_rho) != {str(rho) for rho in STATIC_RHOS}:
        raise ValueError("pilot manifest lacks per-rho probe-count decisions")
    shard_dir = output_dir / phase / "shards"
    expected_keys = {
        (name, rho, replication)
        for name in targets
        for rho in STATIC_RHOS
        for replication in range(repetitions)
    }
    for name, (positions, particle_radius) in targets.items():
        for rho in STATIC_RHOS:
            for replication in range(repetitions):
                evaluation_probes = base_evaluation_probes
                if pilot_by_rho is not None:
                    evaluation_probes *= (
                        2 if pilot_by_rho[str(rho)]["double_evaluation_counts"] else 1
                    )
                path = shard_dir / _shard_name(
                    {"target_name": name, "rho": rho, "replication": replication}
                )
                replace_incomplete = False
                if path.exists():
                    try:
                        existing = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as exc:
                        raise ValueError(f"invalid existing static shard: {path}") from exc
                    if not isinstance(existing, dict):
                        raise ValueError(f"existing static shard must contain an object: {path}")
                    if existing.get("complete") is True:
                        expected_config = _record_config(
                            particle_radius,
                            backend,
                            phase=phase,
                            rho=rho,
                            search_probes=search_probes,
                            evaluation_probes=evaluation_probes,
                        )
                        _validate_static_record(
                            existing,
                            expected_phase=phase,
                            expected_backend=backend,
                            expected_config=expected_config,
                        )
                        if (
                            existing["target_name"] != name
                            or existing["rho"] != rho
                            or existing["replication"] != replication
                            or existing["target_digest"] != _target_digest(positions)
                            or existing["target_particle_count"] != int(positions.shape[0])
                            or existing["particle_radius"] != float(particle_radius)
                        ):
                            raise ValueError(
                                f"existing shard target configuration does not match: {path}"
                            )
                        continue
                    # An incomplete shard may only be replaced deliberately; this
                    # is the explicit retry path for interrupted workers.
                    if not retry_incomplete:
                        raise ValueError(f"incomplete shard requires --retry-incomplete: {path}")
                    replace_incomplete = True
                try:
                    record = run_static_cell(
                        target_name=name,
                        positions=positions,
                        particle_radius=particle_radius,
                        rho=rho,
                        replication=replication,
                        search_probes=search_probes,
                        evaluation_probes=evaluation_probes,
                        seed=_batch_seed(phase, name, "record", rho, replication, "record"),
                        backend=backend,
                        phase=phase,
                    )
                except Exception as error:  # preserve failed work for diagnosis
                    record = {
                        "schema_version": 1,
                        "target_name": name,
                        "rho": rho,
                        "replication": replication,
                        "complete": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
                write_static_shard(shard_dir, record, replace_incomplete=replace_incomplete)
    paths = sorted(shard_dir.glob("*.json"))
    merged = merge_static_shards(
        paths,
        expected_keys=expected_keys,
        expected_phase=phase,
        expected_backend=backend,
        expected_config={
            "phase": phase,
            "backend": backend,
            "search_probes": search_probes,
        },
    )
    for entry in merged["shard_inventory"]:
        resolved = Path(entry["path"]).resolve()
        entry["path"] = str(resolved.relative_to(output_dir.resolve()))
    if phase == "pilot":
        decision = pilot_decision(
            merged["records"],
            evaluation_probes=base_evaluation_probes,
            production_search_probes=8_192,
            production_evaluation_probes=25_000,
        )
    else:
        assert pilot_manifest is not None
        decision = pilot_manifest["pilot_decision"]
    merged.update(
        {
            "phase": phase,
            "experiment_version": EXPERIMENT_VERSION,
            "provenance": runtime_provenance(),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "backend": backend,
            },
            "target_identity": _target_manifest(targets),
            "pilot_decision": decision,
            "targets": {
                name: {
                    "particle_radius": radius,
                    "positions": positions.tolist(),
                    "digest": _target_digest(positions),
                }
                for name, (positions, radius) in targets.items()
            },
        }
    )
    canonical = output_dir / f"asymmetric_validation_{phase}.json"
    temporary = canonical.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(canonical)
    return canonical


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "production"), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/asymmetric_validation"))
    parser.add_argument("--backend", choices=("reference", "numba-cpu"), default="numba-cpu")
    parser.add_argument(
        "--retry-incomplete",
        action="store_true",
        help="replace incomplete shards left by an interrupted run",
    )
    args = parser.parse_args()
    print(
        run_phase(args.phase, args.output_dir, args.backend, retry_incomplete=args.retry_incomplete)
    )


if __name__ == "__main__":
    main()
