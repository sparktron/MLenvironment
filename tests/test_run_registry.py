from __future__ import annotations

from pathlib import Path

from rl_framework.utils.run_registry import RunRegistry


def _cfg() -> dict:
    return {"experiment_name": "registry", "seed": 3, "output": {"base_dir": "outputs"}}


def test_registry_preserves_identity_and_records_lineage(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path)
    parent_dir = tmp_path / "parent" / "seed_0"
    child_dir = tmp_path / "child" / "seed_0"
    registry.register_run("parent", _cfg(), parent_dir)
    registry.register_run("child", _cfg(), child_dir, resume_from=parent_dir / "checkpoints" / "model.zip")

    child = registry.get_run("child")
    assert child is not None
    assert child["parent_run_id"] == "parent"
    assert child["config"] == _cfg()


def test_registry_tuning_queue_is_durable_and_ordered(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path)
    registry.enqueue_tuning("run", {"learning_rate": 3e-4})
    registry.enqueue_tuning("run", {"learning_rate": 1e-4, "reward.target_velocity": 1.0})

    reopened = RunRegistry(tmp_path)
    assert reopened.claim_tuning("run") == {
        "learning_rate": 1e-4,
        "reward.target_velocity": 1.0,
    }
    assert reopened.claim_tuning("run") is None

