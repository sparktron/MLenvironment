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
    registry.register_run(
        "child", _cfg(), child_dir, resume_from=parent_dir / "checkpoints" / "model.zip"
    )

    child = registry.get_run("child")
    assert child is not None
    assert child["parent_run_id"] == "parent"
    assert child["config"] == _cfg()


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
