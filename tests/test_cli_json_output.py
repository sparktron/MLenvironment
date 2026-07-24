"""Tests for the CLI's --json / --json-out automation output plumbing.

These exercise main()'s output handling directly (with the heavy runners
patched out) so the JSON-emitting paths are covered without training a model.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from rl_framework.cli import main as cli_main


@pytest.fixture
def _patched_config(monkeypatch):
    """Stub config loading/validation so any --config-name resolves to a dict."""
    cfg = {
        "experiment_name": "cli_json_test",
        "seed": 0,
        "output": {"base_dir": "outputs"},
        "training": {},
        "environment": {"type": "walker_bullet"},
    }
    import rl_framework.utils.config as config_mod

    monkeypatch.setattr(config_mod, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(config_mod, "to_container", lambda c: dict(c))
    monkeypatch.setattr(config_mod, "validate_experiment_config", lambda c: None)
    return cfg


def _run_cli(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["prog", *argv])
    cli_main.main()


def test_train_json_stdout(monkeypatch, capsys, _patched_config):
    import rl_framework.training.sb3_runner as runner

    monkeypatch.setattr(runner, "train", lambda cfg, resume_from=None: "/out/model.zip")
    _run_cli(monkeypatch, ["train", "--config-name", "x", "--json"])

    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload == {"saved_model": "/out/model.zip"}


def test_train_human_output_when_no_json(monkeypatch, capsys, _patched_config):
    import rl_framework.training.sb3_runner as runner

    monkeypatch.setattr(runner, "train", lambda cfg, resume_from=None: "/out/model.zip")
    _run_cli(monkeypatch, ["train", "--config-name", "x"])

    out = capsys.readouterr().out
    assert "saved_model=/out/model.zip" in out
    with pytest.raises(json.JSONDecodeError):
        json.loads(out.strip())


def test_json_out_writes_file(monkeypatch, tmp_path, _patched_config):
    import rl_framework.training.sb3_runner as runner

    monkeypatch.setattr(runner, "train", lambda cfg, resume_from=None: "/out/model.zip")
    out_file = tmp_path / "nested" / "result.json"
    _run_cli(
        monkeypatch,
        ["train", "--config-name", "x", "--json-out", str(out_file)],
    )

    assert out_file.exists()
    payload = json.loads(out_file.read_text())
    assert payload == {"saved_model": "/out/model.zip"}


def test_registry_cli_inspects_exports_and_prunes(monkeypatch, capsys, tmp_path):
    from rl_framework.utils.run_registry import RunRegistry

    registry = RunRegistry(tmp_path)
    registry.create_analysis_job("stale_job", "replay")
    registry.finish_analysis_job("stale_job", status="interrupted", error="restart")

    _run_cli(
        monkeypatch,
        ["registry", "--base-dir", str(tmp_path), "--json"],
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["analysis_jobs_by_status"] == {"interrupted": 1}

    export_path = tmp_path / "registry-export.json"
    _run_cli(
        monkeypatch,
        [
            "registry",
            "--base-dir",
            str(tmp_path),
            "--registry-action",
            "export",
            "--json-out",
            str(export_path),
        ],
    )
    assert (
        json.loads(export_path.read_text())["analysis_jobs"][0]["job_id"] == "stale_job"
    )

    _run_cli(
        monkeypatch,
        [
            "registry",
            "--base-dir",
            str(tmp_path),
            "--registry-action",
            "prune",
            "--status",
            "interrupted",
            "--older-than-days",
            "0",
            "--dry-run",
            "--json",
        ],
    )
    preview = json.loads(capsys.readouterr().out)
    assert preview["ids"] == ["stale_job"]
    assert registry.get_analysis_job("stale_job") is not None

    _run_cli(
        monkeypatch,
        [
            "registry",
            "--base-dir",
            str(tmp_path),
            "--registry-action",
            "prune",
            "--status",
            "interrupted",
            "--all",
            "--json",
        ],
    )
    assert json.loads(capsys.readouterr().out)["matched"] == 1
    assert registry.get_analysis_job("stale_job") is None


def test_registry_cli_requires_a_prune_filter(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="requires --status"):
        _run_cli(
            monkeypatch,
            [
                "registry",
                "--base-dir",
                str(tmp_path),
                "--registry-action",
                "prune",
            ],
        )


def _registry_argv(tmp_path, *extra):
    return ["registry", "--base-dir", str(tmp_path), *extra]


def test_registry_cli_prunes_runs_and_missing_artifacts(monkeypatch, capsys, tmp_path):
    """Exercise the CLI dispatch to prune_runs and prune_artifacts (missing-only)
    plus the human-readable (non-JSON) prune output branch."""
    from rl_framework.utils.run_registry import RunRegistry

    registry = RunRegistry(tmp_path)
    cfg = {"experiment_name": "exp", "seed": 0, "output": {"base_dir": str(tmp_path)}}
    run_dir = tmp_path / "exp" / "seed_0"
    registry.register_run("run_gone", cfg, run_dir)
    registry.update_run("run_gone", status="failed")

    # One artifact that stays on disk, one recorded then deleted so its index
    # row is "missing" (record_artifact skips paths that don't yet exist).
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    present = checkpoints / "final_model.zip"
    present.write_bytes(b"")
    missing = checkpoints / "old_model.zip"
    missing.write_bytes(b"")
    registry.record_artifact("run_gone", "checkpoint", present)
    registry.record_artifact("run_gone", "checkpoint", missing)
    missing.unlink()

    # Prune only the missing-artifact index row (human-readable output).
    _run_cli(
        monkeypatch,
        _registry_argv(
            tmp_path,
            "--registry-action",
            "prune",
            "--prune-target",
            "artifacts",
            "--missing-only",
        ),
    )
    out = capsys.readouterr().out
    assert "pruned 1 artifacts record(s)" in out
    assert "old_model.zip" in out
    remaining = {a["path"] for a in registry.export()["run_artifacts"]}
    assert str(present) in remaining
    assert str(missing) not in remaining

    # Prune the failed run itself (JSON output), cascading to its rows.
    _run_cli(
        monkeypatch,
        _registry_argv(
            tmp_path,
            "--registry-action",
            "prune",
            "--prune-target",
            "runs",
            "--status",
            "failed",
            "--json",
        ),
    )
    assert json.loads(capsys.readouterr().out)["ids"] == ["run_gone"]
    assert registry.get_run("run_gone") is None
    assert registry.export()["run_artifacts"] == []


def test_registry_cli_human_readable_inspect_and_export(monkeypatch, capsys, tmp_path):
    """Cover the non-JSON inspect and export summary print branches."""
    from rl_framework.utils.run_registry import RunRegistry

    registry = RunRegistry(tmp_path)
    registry.create_analysis_job("j1", "replay")
    registry.finish_analysis_job("j1", status="completed", result={})

    _run_cli(monkeypatch, _registry_argv(tmp_path))  # inspect, no --json
    out = capsys.readouterr().out
    assert "registry=" in out
    assert "analysis_jobs=1" in out
    assert "missing_artifacts=" in out

    _run_cli(
        monkeypatch, _registry_argv(tmp_path, "--registry-action", "export")
    )  # no --json / --json-out
    out = capsys.readouterr().out
    assert "exported rows=" in out
    assert "analysis_jobs" in out


def test_registry_cli_prune_guard_branches(monkeypatch, tmp_path):
    """Cover the remaining prune argument-validation SystemExits."""
    with pytest.raises(SystemExit, match="non-negative"):
        _run_cli(
            monkeypatch,
            _registry_argv(
                tmp_path, "--registry-action", "prune", "--older-than-days", "-1"
            ),
        )
    with pytest.raises(SystemExit, match="missing-only is valid only"):
        _run_cli(
            monkeypatch,
            _registry_argv(tmp_path, "--registry-action", "prune", "--missing-only"),
        )
    with pytest.raises(SystemExit, match="status is not valid"):
        _run_cli(
            monkeypatch,
            _registry_argv(
                tmp_path,
                "--registry-action",
                "prune",
                "--prune-target",
                "artifacts",
                "--status",
                "failed",
            ),
        )


def test_quality_study_cli_dry_run_json(monkeypatch, capsys, tmp_path):
    _run_cli(
        monkeypatch,
        [
            "quality-study",
            "--study",
            "algorithms",
            "--seeds",
            "0,1,2",
            "--study-output-dir",
            str(tmp_path),
            "--dry-run",
            "--json",
        ],
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["planned_runs"] == {
        "walker": 0,
        "arena": 0,
        "algorithms": 18,
    }
    assert list(tmp_path.iterdir()) == []


def test_walker_render_replay_loads_vecnormalize_sidecar(monkeypatch, tmp_path):
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    from rl_framework.cli.main import _render_replay

    class _FakeEnv(gym.Env):
        metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

        def __init__(self) -> None:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(2, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(2, dtype=np.float32), 0.0, True, False, {}

        def render(self):
            return np.zeros((8, 8, 3), dtype=np.uint8)

    class _FakeRecordVideo(gym.Wrapper):
        def __init__(self, env, video_folder, episode_trigger):
            super().__init__(env)
            self.video_folder = video_folder
            self.episode_trigger = episode_trigger

    class _FakeModel:
        def predict(self, obs, deterministic=True):
            return np.zeros((1, 1), dtype=np.float32), None

    import rl_framework.envs.registry as registry_mod

    monkeypatch.setattr(registry_mod, "make_env", lambda _env_type, _cfg: _FakeEnv())
    monkeypatch.setattr("gymnasium.wrappers.RecordVideo", _FakeRecordVideo)
    monkeypatch.setattr(PPO, "load", staticmethod(lambda _path: _FakeModel()))

    loaded = {}

    def _fake_vecnormalize_load(path, venv):
        loaded["path"] = path
        return venv

    monkeypatch.setattr(VecNormalize, "load", staticmethod(_fake_vecnormalize_load))

    model_path = tmp_path / "final_model.zip"
    model_path.write_bytes(b"fake")
    sidecar = tmp_path / "final_model_vecnormalize.pkl"
    sidecar.write_bytes(b"fake")
    cfg = {
        "experiment_name": "replay_test",
        "seed": 0,
        "output": {"base_dir": str(tmp_path / "outputs")},
        "environment": {"type": "walker_bullet"},
    }

    result = _render_replay(cfg, str(model_path))

    assert loaded["path"] == str(sidecar)
    assert result["frames"] == 1
