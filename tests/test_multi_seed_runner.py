"""Tests for multi-seed runner failure isolation (plan §5c).

Verifies that when one seed's training raises an exception the remaining
seeds still complete, the aggregate CSV is written with both successful
and failed rows, and the returned dict accurately reflects which seeds
succeeded and which failed.
"""

from __future__ import annotations

import csv
import warnings
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
    monkeypatch.setattr(
        ms_module, "_run_one_seed", _fake_run_one_seed_factory(fail_seed=1)
    )

    cfg = _base_cfg(tmp_path)
    result = ms_module.run_multi_seed(cfg, seeds=[0, 1, 2], max_workers=1)

    assert result["successful_seeds"] == [0, 2]
    assert 1 in result["failed_seeds"]
    assert "Intentional failure" in result["failed_seeds"][1]


def test_aggregate_csv_contains_all_seeds(tmp_path: Path, monkeypatch) -> None:
    """aggregate.csv must have a row for every seed — success or failure."""
    monkeypatch.setattr(
        ms_module, "_run_one_seed", _fake_run_one_seed_factory(fail_seed=2)
    )

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
    monkeypatch.setattr(
        ms_module, "_run_one_seed", _fake_run_one_seed_factory(fail_seed=1)
    )

    cfg = _base_cfg(tmp_path)
    # seed 0 → 0.5, seed 1 → fails, seed 2 → 2.5  ⟹  mean of [0.5, 2.5] = 1.5
    result = ms_module.run_multi_seed(cfg, seeds=[0, 1, 2], max_workers=1)

    assert abs(result["mean_return_mean"] - 1.5) < 1e-9


def test_parallel_rollouts_default_multi_seed_to_sequential(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ms_module, "_run_one_seed", _fake_run_one_seed_factory())
    cfg = _base_cfg(tmp_path)
    cfg["training"]["num_envs"] = 8

    with pytest.warns(UserWarning, match="max_workers=1"):
        result = ms_module.run_multi_seed(cfg, seeds=[0, 1])

    assert result["successful_seeds"] == [0, 1]


def test_per_seed_configs_update_environment_seed(tmp_path: Path, monkeypatch) -> None:
    """Arena env construction reads environment.seed, so it must vary too."""
    seen: dict[int, int] = {}

    def _capture(args: tuple) -> tuple:
        seed, cfg = args
        seen[seed] = cfg["environment"]["seed"]
        return seed, f"/fake/model_seed_{seed}", {"mean_return": float(seed)}

    monkeypatch.setattr(ms_module, "_run_one_seed", _capture)

    cfg = _base_cfg(tmp_path)
    cfg["environment"]["seed"] = 999
    ms_module.run_multi_seed(cfg, seeds=[3, 4], max_workers=1)

    assert seen == {3: 3, 4: 4}


def test_default_max_workers_is_one_when_num_envs_oversubscribes(
    tmp_path: Path, monkeypatch
) -> None:
    """Each seed's own training fans out num_envs SubprocVecEnv workers, so
    the default parallelism across seeds must not also multiply by seed
    count (5 seeds x num_envs=24 = 120 processes on a 24-core box)."""
    seen_max_workers: list[int] = []

    monkeypatch.setattr(
        ms_module, "_run_one_seed", _fake_run_one_seed_factory(fail_seed=None)
    )
    real_executor = ms_module.ProcessPoolExecutor

    def _capture_executor(max_workers=None, **kwargs):
        seen_max_workers.append(max_workers)
        return real_executor(max_workers=max_workers, **kwargs)

    monkeypatch.setattr(ms_module, "ProcessPoolExecutor", _capture_executor)

    cfg = _base_cfg(tmp_path)
    cfg["training"]["num_envs"] = 24
    with pytest.warns(UserWarning, match="max_workers=1"):
        result = ms_module.run_multi_seed(cfg, seeds=[0, 1, 2])

    assert result["successful_seeds"] == [0, 1, 2]
    # max_workers=1 takes the sequential path — ProcessPoolExecutor never used.
    assert seen_max_workers == []


def test_default_max_workers_unchanged_for_single_env_seeds(
    tmp_path: Path, monkeypatch
) -> None:
    """The pre-existing min(len(seeds), cpu_count) default is preserved when
    each seed's own training does not itself parallelize (num_envs == 1).

    cpu_count is pinned to 1 so the resolved default (min(3, 1) == 1) takes
    the sequential path deterministically, without depending on the actual
    core count of the machine running the test or spawning real subprocesses
    (the stub below is a local closure and isn't picklable)."""
    monkeypatch.setattr(
        ms_module, "_run_one_seed", _fake_run_one_seed_factory(fail_seed=None)
    )
    monkeypatch.setattr(ms_module.os, "cpu_count", lambda: 1)

    cfg = _base_cfg(tmp_path)
    assert cfg["training"].get("num_envs", 1) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = ms_module.run_multi_seed(cfg, seeds=[0, 1, 2])

    assert result["successful_seeds"] == [0, 1, 2]
