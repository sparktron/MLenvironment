"""Tests for multi-seed runner failure isolation (plan §5c).

Verifies that when one seed's training raises an exception the remaining
seeds still complete, the aggregate CSV is written with both successful
and failed rows, and the returned dict accurately reflects which seeds
succeeded and which failed.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from rl_framework.training import multi_seed_runner as ms_module


def _base_cfg(tmp_path: Path) -> dict:
    return {
        "experiment_name": "ms_test",
        "seed": 0,
        "output": {"base_dir": str(tmp_path)},
        "environment": {"type": "walker_bullet"},
        "training": {"total_timesteps": 100},
        "evaluation": {"episodes": 1},
    }


def _fake_run_one_seed_factory(fail_seed: int | None = None):
    """Return a _run_one_seed stub that raises for *fail_seed* and succeeds for others."""

    def _stub(args: tuple) -> tuple:
        seed, cfg = args
        if seed == fail_seed:
            raise RuntimeError(f"Intentional failure for seed {seed}")
        return seed, f"/fake/model_seed_{seed}", {"mean_return": float(seed) + 0.5}

    return _stub


def test_one_failed_seed_does_not_block_others(tmp_path: Path, monkeypatch) -> None:
    """A single failing seed must not prevent the other seeds from completing."""
    monkeypatch.setattr(ms_module, "_run_one_seed", _fake_run_one_seed_factory(fail_seed=1))

    cfg = _base_cfg(tmp_path)
    result = ms_module.run_multi_seed(cfg, seeds=[0, 1, 2], max_workers=1)

    assert result["successful_seeds"] == [0, 2]
    assert 1 in result["failed_seeds"]
    assert "Intentional failure" in result["failed_seeds"][1]


def test_aggregate_csv_contains_all_seeds(tmp_path: Path, monkeypatch) -> None:
    """aggregate.csv must have a row for every seed — success or failure."""
    monkeypatch.setattr(ms_module, "_run_one_seed", _fake_run_one_seed_factory(fail_seed=2))

    cfg = _base_cfg(tmp_path)
    ms_module.run_multi_seed(cfg, seeds=[0, 1, 2], max_workers=1)

    csv_path = Path(tmp_path) / "ms_test" / "multi_seed_summary" / "aggregate.csv"
    assert csv_path.exists(), f"aggregate.csv not written to {csv_path}"

    rows = list(csv.reader(csv_path.open(encoding="utf-8")))
    seed_rows = [r for r in rows if r and r[0].lstrip("-").isdigit()]
    written_seeds = {int(r[0]) for r in seed_rows}
    assert {0, 1, 2} == written_seeds, f"Expected all seeds in CSV; got {written_seeds}"


def test_all_seeds_failed_raises_runtime_error(tmp_path: Path, monkeypatch) -> None:
    """run_multi_seed raises RuntimeError when every seed fails."""

    def _always_fail(args: tuple) -> tuple:
        seed, _ = args
        raise RuntimeError(f"seed {seed} failed")

    monkeypatch.setattr(ms_module, "_run_one_seed", _always_fail)

    cfg = _base_cfg(tmp_path)
    with pytest.raises(RuntimeError, match="All seeds failed"):
        ms_module.run_multi_seed(cfg, seeds=[0, 1], max_workers=1)


def test_aggregate_mean_uses_only_successful_seeds(tmp_path: Path, monkeypatch) -> None:
    """mean_return_mean must be computed from successful seeds only."""
    monkeypatch.setattr(ms_module, "_run_one_seed", _fake_run_one_seed_factory(fail_seed=1))

    cfg = _base_cfg(tmp_path)
    # seed 0 → 0.5, seed 1 → fails, seed 2 → 2.5  ⟹  mean of [0.5, 2.5] = 1.5
    result = ms_module.run_multi_seed(cfg, seeds=[0, 1, 2], max_workers=1)

    assert abs(result["mean_return_mean"] - 1.5) < 1e-9
