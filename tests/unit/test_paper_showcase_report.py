from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_paper_showcase_report import load_benchmark_inputs, schedule_scaling


def test_load_benchmark_inputs_merges_records_and_recomputes_summaries(tmp_path: Path) -> None:
    records = []
    for seed, exact_time, paper_time in ((1, 1.0, 4.0), (2, 3.0, 8.0)):
        for approach, elapsed in (
            ("exact", exact_time),
            ("uniform", exact_time * 2),
            ("controlled-fixed", exact_time * 3),
            ("paper-amortized", paper_time),
        ):
            records.append(
                {
                    "approach": approach,
                    "seed": seed,
                    "elapsed_seconds": elapsed,
                    "particles_per_second": 100 / elapsed,
                    "peak_rss_bytes": 1000,
                    "growth_walker_steps": 10,
                    "growth_restarts": 1,
                    "calibration_walker_steps": 0,
                    "calibration_restarts": 0,
                    "calibration_probes": 0,
                    "calibration_events": 0,
                    "max_one_step_bound": 0.0,
                    "conditional_path_bound": 0.0,
                    "shape": {
                        "radius_of_gyration": 1.0,
                        "max_seed_radius": 1.0,
                        "center_of_mass_offset": 1.0,
                        "anisotropy_ratio": 1.0,
                    },
                    "representative_positions": [[0.0, 0.0]],
                }
            )
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    base = {
        "methodology": {"particles": 100, "seeds": [], "repeats": 0},
        "environment": {"threads": 4},
    }
    first.write_text(json.dumps({**base, "records": records[:4]}), encoding="utf-8")
    second.write_text(json.dumps({**base, "records": records[4:]}), encoding="utf-8")

    merged = load_benchmark_inputs([first, second])

    assert merged["methodology"]["seeds"] == [1, 2]
    assert merged["summaries"]["exact"]["runtime_median_seconds"] == 2.0
    assert merged["summaries"]["paper-amortized"]["runtime_median_seconds"] == 6.0


def test_schedule_scaling_matches_the_shipped_1200_particle_schedule() -> None:
    row = schedule_scaling([1200])[0]

    assert row["paper_events"] == 59
    assert row["paper_probes"] == 8271
    assert row["fixed_probes"] == 9_822_208
