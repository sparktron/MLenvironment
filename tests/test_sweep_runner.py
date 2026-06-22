from pathlib import Path

import pytest

from rl_framework.training import sweep as sweep_module
from rl_framework.utils.config_merge import set_nested


def test_run_sweep_dry_run_writes_manifest_and_skips_training(
    tmp_path, monkeypatch
) -> None:
    calls = {"count": 0}

    def _fake_train(_cfg):
        calls["count"] += 1

    monkeypatch.setattr(sweep_module, "train", _fake_train)
    cfg = {
        "experiment_name": "exp",
        "output": {"base_dir": str(tmp_path)},
        "training": {"learning_rate": 3e-4},
        "sweep": {"parameters": {"training.learning_rate": [1e-4, 2e-4]}},
    }
    planned = sweep_module.run_sweep(cfg, dry_run=True)
    assert len(planned) == 2
    assert calls["count"] == 0

    manifest = Path(tmp_path) / "exp" / "sweep_summary" / "planned_runs.csv"
    assert manifest.exists()


def test_run_sweep_executes_training_when_not_dry_run(tmp_path, monkeypatch) -> None:
    calls = {"count": 0}

    def _fake_train(_cfg):
        calls["count"] += 1

    monkeypatch.setattr(sweep_module, "train", _fake_train)
    cfg = {
        "experiment_name": "exp",
        "output": {"base_dir": str(tmp_path)},
        "training": {"learning_rate": 3e-4},
        "sweep": {"parameters": {"training.learning_rate": [1e-4, 2e-4, 3e-4]}},
    }
    planned = sweep_module.run_sweep(cfg, dry_run=False)
    assert len(planned) == 3
    assert calls["count"] == 3


def test_run_sweep_routes_variants_through_run_id_not_name(
    tmp_path, monkeypatch
) -> None:
    seen: list[dict] = []

    def _fake_train(cfg):
        seen.append(cfg)

    monkeypatch.setattr(sweep_module, "train", _fake_train)
    cfg = {
        "experiment_name": "exp",
        "output": {"base_dir": str(tmp_path)},
        "training": {"learning_rate": 3e-4},
        "sweep": {"parameters": {"training.learning_rate": [1e-4, 2e-4]}},
    }
    planned = sweep_module.run_sweep(cfg, dry_run=False)

    # experiment_name is never mutated; the variant lives in output.run_id.
    assert {c["experiment_name"] for c in seen} == {"exp"}
    assert [c["output"]["run_id"] for c in seen] == [
        "learning_rate_0.0001",
        "learning_rate_0.0002",
    ]
    assert all(p["experiment_name"] == "exp" for p in planned)
    assert [p["run_id"] for p in planned] == [
        "learning_rate_0.0001",
        "learning_rate_0.0002",
    ]


def test_set_nested_rejects_missing_intermediate_key() -> None:
    cfg = {"training": {"learning_rate": 3e-4}}
    with pytest.raises(KeyError, match="intermediate key 'optimizer' not found"):
        set_nested(cfg, "training.optimizer.lr", 1e-4)
