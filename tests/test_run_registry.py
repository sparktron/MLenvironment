from __future__ import annotations

from pathlib import Path

import pytest

from rl_framework.utils.run_registry import RunRegistry


def _cfg() -> dict:
    return {"experiment_name": "registry", "seed": 3, "output": {"base_dir": "outputs"}}


def test_registry_preserves_identity_and_records_lineage(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path)
    parent_dir = tmp_path / "parent" / "seed_0"
    child_dir = tmp_path / "child" / "seed_0"
    registry.register_run("parent", _cfg(), parent_dir)
    registry.register_run(
        "child", _cfg(), child_dir, resume_from=parent_dir / "checkpoints" / "model.zip"
    )

    child = registry.get_run("child")
    assert child is not None
    assert child["parent_run_id"] == "parent"
    assert child["config"] == _cfg()


def test_registry_keeps_variant_id_separate_from_execution_id(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path)
    cfg = _cfg()
    cfg["output"]["run_id"] = "morph_000"
    registry.register_run("run_execution_a", cfg, tmp_path / "exp_a")

    row = registry.get_run("run_execution_a")
    assert row is not None
    assert row["run_id"] == "run_execution_a"
    assert row["variant_id"] == "morph_000"
    assert registry.list_runs()[0]["variant_id"] == "morph_000"


def test_registry_rejects_execution_id_reuse_for_different_run(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path)
    cfg = _cfg()
    cfg["output"]["run_id"] = "morph_000"
    registry.register_run("run_execution", cfg, tmp_path / "exp_a")

    conflicting_cfg = _cfg()
    conflicting_cfg["experiment_name"] = "other_experiment"
    conflicting_cfg["output"]["run_id"] = "morph_000"
    with pytest.raises(ValueError, match="different training execution"):
        registry.register_run("run_execution", conflicting_cfg, tmp_path / "exp_b")

    assert registry.get_run("run_execution")["run_dir"] == str(tmp_path / "exp_a")


def test_registry_existing_identity_requires_explicit_orchestrator_handoff(
    tmp_path: Path,
) -> None:
    registry = RunRegistry(tmp_path)
    run_dir = tmp_path / "exp"
    registry.register_run("run_execution", _cfg(), run_dir)

    with pytest.raises(ValueError, match="different training execution"):
        registry.register_run("run_execution", _cfg(), run_dir)

    assert (
        registry.register_run(
            "run_execution", _cfg(), run_dir, allow_existing=True
        )
        == "run_execution"
    )


def test_registry_tuning_queue_is_durable_and_ordered(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path)
    registry.enqueue_tuning("run", {"learning_rate": 3e-4})
    registry.enqueue_tuning(
        "run", {"learning_rate": 1e-4, "reward.target_velocity": 1.0}
    )

    reopened = RunRegistry(tmp_path)
    assert reopened.claim_tuning("run") == {
        "learning_rate": 1e-4,
        "reward.target_velocity": 1.0,
    }
    assert reopened.claim_tuning("run") is None


def test_analysis_jobs_persist_across_reopen(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path)
    registry.create_analysis_job("job_done", "replay", {"path": "exp/seed_0"})
    registry.finish_analysis_job(
        "job_done", status="completed", result={"saved_replay": "replay.gif"}
    )
    registry.create_analysis_job(
        "job_orphan", "league_ratings", {"path": "arena/seed_0"}
    )

    reopened = RunRegistry(tmp_path)
    done = reopened.get_analysis_job("job_done")
    assert done is not None
    assert done["status"] == "completed"
    assert done["result"] == {"saved_replay": "replay.gif"}
    assert done["params"] == {"path": "exp/seed_0"}
    assert done["finished_at"] is not None
    assert reopened.get_analysis_job("missing") is None
    assert [job["job_id"] for job in reopened.list_analysis_jobs()] == [
        "job_orphan",
        "job_done",
    ]


def test_recover_interrupted_analysis_jobs_marks_only_running(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path)
    registry.create_analysis_job("job_done", "replay")
    registry.finish_analysis_job("job_done", status="completed", result={})
    registry.create_analysis_job("job_failed", "replay")
    registry.finish_analysis_job("job_failed", status="failed", error="boom")
    registry.create_analysis_job("job_orphan", "replay")

    reopened = RunRegistry(tmp_path)  # simulates GUI restart
    assert reopened.recover_interrupted_analysis_jobs() == 1
    orphan = reopened.get_analysis_job("job_orphan")
    assert orphan is not None
    assert orphan["status"] == "interrupted"
    assert orphan["error"]
    assert reopened.get_analysis_job("job_done")["status"] == "completed"
    assert reopened.get_analysis_job("job_failed")["status"] == "failed"
    assert reopened.recover_interrupted_analysis_jobs() == 0


def test_registry_summary_export_and_missing_artifact_prune(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path)
    run_dir = tmp_path / "exp" / "seed_3"
    artifact = run_dir / "checkpoints" / "model.zip"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"model")
    registry.register_run("run", _cfg(), run_dir)
    registry.record_artifact("run", "checkpoint", artifact)
    registry.create_analysis_job("job", "replay")
    registry.finish_analysis_job("job", status="failed", error="boom")
    artifact.unlink()

    summary = registry.summary()
    assert summary["runs_total"] == 1
    assert summary["analysis_jobs_by_status"] == {"failed": 1}
    assert summary["run_artifacts_total"] == 1
    assert summary["missing_artifacts_total"] == 1
    exported = registry.export()
    assert {row["run_id"] for row in exported["runs"]} == {"run"}
    assert {row["job_id"] for row in exported["analysis_jobs"]} == {"job"}

    preview = registry.prune_artifacts(missing_only=True, dry_run=True)
    assert len(preview) == 1
    assert registry.summary()["run_artifacts_total"] == 1
    assert registry.prune_artifacts(missing_only=True) == preview
    assert registry.summary()["run_artifacts_total"] == 0


def test_registry_prunes_stale_jobs_and_run_relations(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path)
    run_dir = tmp_path / "exp" / "seed_3"
    artifact = run_dir / "model.zip"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"model")
    registry.register_run("run", _cfg(), run_dir)
    registry.enqueue_tuning("run", {"learning_rate": 1e-4})
    registry.record_artifact("run", "final_model", artifact)
    registry.update_run("run", status="completed")
    registry.create_analysis_job("job_done", "replay")
    registry.finish_analysis_job("job_done", status="completed", result={})
    registry.create_analysis_job("job_live", "replay")

    cutoff_after_all_rows = float("inf")
    assert registry.prune_analysis_jobs(
        statuses=["completed"], older_than=cutoff_after_all_rows, dry_run=True
    ) == ["job_done"]
    assert registry.get_analysis_job("job_done") is not None
    assert registry.prune_analysis_jobs(
        statuses=["completed"], older_than=cutoff_after_all_rows
    ) == ["job_done"]
    assert registry.get_analysis_job("job_done") is None
    assert registry.get_analysis_job("job_live") is not None

    assert registry.prune_runs(statuses=["completed"]) == ["run"]
    assert registry.get_run("run") is None
    exported = registry.export()
    assert exported["run_events"] == []
    assert exported["tuning_commands"] == []
    assert exported["run_artifacts"] == []
    assert artifact.exists(), "registry maintenance must not delete artifact files"


def test_metric_history_is_oldest_first_and_supports_key_filtering(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path)
    registry.register_run("run", _cfg(), tmp_path / "run")
    registry.record_event(
        "run", "metrics", {"timesteps": 10, "rollout/ep_rew_mean": 1.5, "other": 4}
    )
    registry.record_event(
        "run", "metrics", {"timesteps": 20, "rollout/ep_rew_mean": 2.5, "other": 5}
    )

    history = registry.metric_history(
        "run", ["timesteps", "rollout/ep_rew_mean"]
    )
    assert [point["timesteps"] for point in history] == [10, 20]
    assert [point["rollout/ep_rew_mean"] for point in history] == [1.5, 2.5]
    assert all("other" not in point and isinstance(point["t"], float) for point in history)
