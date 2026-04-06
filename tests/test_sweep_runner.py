from pathlib import Path

from rl_framework.training import sweep as sweep_module


def test_run_sweep_dry_run_writes_manifest_and_skips_training(tmp_path, monkeypatch) -> None:
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
