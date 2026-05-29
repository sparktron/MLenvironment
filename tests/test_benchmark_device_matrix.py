from __future__ import annotations

import json

import pytest

from scripts import benchmark_device_matrix as bdm


def test_regimes_run_from_smaller_to_larger_worker_counts() -> None:
    assert [(regime.name, regime.device, regime.max_workers) for regime in bdm.REGIMES] == [
        ("CPU-4workers", "cpu", 4),
        ("CPU-8workers", "cpu", 8),
        ("GPU-1worker", "cuda", 1),
        ("GPU-2workers", "cuda", 2),
    ]


def test_matrix_metadata_identifies_runtime_script_and_order() -> None:
    metadata = bdm._matrix_metadata()

    assert metadata["matrix_version"] == "2026-05-28-small-first-v2"
    assert metadata["script_path"].endswith("scripts/benchmark_device_matrix.py")
    assert metadata["order"] == [
        "CPU-4workers",
        "CPU-8workers",
        "GPU-1worker",
        "GPU-2workers",
    ]


def test_append_progress_log_writes_jsonl_events(tmp_path) -> None:
    progress_log = tmp_path / "nested" / "progress.jsonl"

    bdm._append_progress_log(progress_log, {"event": "regime_completed", "name": "CPU-4workers"})
    bdm._append_progress_log(progress_log, {"event": "regime_started", "name": "CPU-8workers"})

    events = [json.loads(line) for line in progress_log.read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == ["regime_completed", "regime_started"]
    assert [event["name"] for event in events] == ["CPU-4workers", "CPU-8workers"]
    assert all("ts" in event for event in events)


def test_run_regime_timeout_writes_progress_event(monkeypatch, tmp_path) -> None:
    class FakeProcess:
        pid = 12345

        def __init__(self) -> None:
            self.killed = False

        def poll(self) -> None:
            return None

        def kill(self) -> None:
            self.killed = True

    fake_process = FakeProcess()
    times = iter([0.0, 301.0])
    progress_log = tmp_path / "progress.jsonl"

    monkeypatch.setattr(bdm.subprocess, "Popen", lambda _cmd: fake_process)
    monkeypatch.setattr(bdm.time, "perf_counter", lambda: next(times))

    with pytest.raises(RuntimeError, match="timed out"):
        bdm._run_regime(
            config_name="robot_walk_basic",
            seeds="0",
            config_dir="src/rl_framework/configs/experiments",
            regime=bdm.Regime(name="CPU-4workers", device="cpu", max_workers=4),
            inactivity_timeout_s=300.0,
            heartbeat_s=999.0,
            total_timesteps=20000,
            debug=False,
            progress_log=progress_log,
            ordinal=1,
            total_regimes=4,
        )

    events = [json.loads(line) for line in progress_log.read_text(encoding="utf-8").splitlines()]
    assert fake_process.killed
    assert [event["event"] for event in events] == ["regime_started", "regime_timed_out"]
    assert events[1]["name"] == "CPU-4workers"
    assert events[1]["timeout_s"] == 300.0
