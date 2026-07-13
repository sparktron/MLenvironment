"""Tests for the Flask GUI endpoints.

These exercise config CRUD, schema, training manager error paths, and outputs
listing through Flask's in-process test client. Real training is not spun up
because only the error/validation branches of the training endpoints are hit.
"""

from __future__ import annotations


import pytest
import yaml


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Fresh Flask test client with isolated CONFIGS_DIR and TrainingManager."""
    from rl_framework.gui import app as gui_app
    from rl_framework.gui.training_manager import TrainingManager

    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    monkeypatch.setattr(gui_app, "CONFIGS_DIR", configs_dir)
    monkeypatch.setattr(gui_app, "manager", TrainingManager())
    monkeypatch.setattr(gui_app, "_DEFAULT_OUTPUTS_DIR", tmp_path / "outputs")

    gui_app.app.config["TESTING"] = True
    with gui_app.app.test_client() as c:
        yield c, configs_dir, tmp_path


def _minimal_cfg(name: str = "demo") -> dict:
    return {
        "experiment_name": name,
        "seed": 0,
        "output": {"base_dir": "outputs"},
        "environment": {"type": "walker_bullet"},
        "training": {"total_timesteps": 1000},
    }


# ----- config CRUD -----


def test_list_configs_empty(client):
    c, _, _ = client
    resp = c.get("/api/configs")
    assert resp.status_code == 200
    assert resp.get_json() == []


def test_list_configs_with_entries(client):
    c, configs_dir, _ = client
    (configs_dir / "foo.yaml").write_text(yaml.dump(_minimal_cfg("foo")))
    resp = c.get("/api/configs")
    assert resp.status_code == 200
    names = [entry["name"] for entry in resp.get_json()]
    assert "foo" in names


def test_get_config_ok(client):
    c, configs_dir, _ = client
    (configs_dir / "bar.yaml").write_text(yaml.dump(_minimal_cfg("bar")))
    resp = c.get("/api/configs/bar")
    assert resp.status_code == 200
    assert resp.get_json()["experiment_name"] == "bar"


def test_get_config_missing_returns_404(client):
    c, _, _ = client
    resp = c.get("/api/configs/nope")
    assert resp.status_code == 404


def test_get_config_rejects_path_traversal(client):
    c, _, _ = client
    # ".." in the name segment should be rejected before any FS access.
    resp = c.get("/api/configs/..")
    assert resp.status_code == 400


def test_save_config_roundtrip(client):
    c, configs_dir, _ = client
    payload = _minimal_cfg("saved")
    resp = c.put("/api/configs/saved", json=payload)
    assert resp.status_code == 200
    assert resp.get_json() == {"saved": "saved"}
    assert (configs_dir / "saved.yaml").exists()


def test_save_config_rejects_traversal(client):
    c, _, _ = client
    resp = c.put("/api/configs/..", json={"x": 1})
    assert resp.status_code == 400


def test_create_config_ok(client):
    c, configs_dir, _ = client
    resp = c.post("/api/configs", json=_minimal_cfg("New Run"))
    assert resp.status_code == 200
    assert (configs_dir / "new_run.yaml").exists()


def test_create_config_missing_name(client):
    c, _, _ = client
    resp = c.post("/api/configs", json={"foo": "bar"})
    assert resp.status_code == 400


def test_create_config_rejects_traversal_name(client):
    c, _, _ = client
    resp = c.post("/api/configs", json={"experiment_name": "../evil"})
    assert resp.status_code == 400


# ----- schema -----


def test_schema_returns_both_envs(client):
    c, _, _ = client
    resp = c.get("/api/schema")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "walker_bullet" in data and "organism_arena_parallel" in data
    assert "training" in data["walker_bullet"]
    assert data["walker_bullet"]["training"]["device"]["value"] == "cpu"
    assert data["walker_bullet"]["training"]["check_nans"]["value"] is False


def test_schema_allows_parallel_self_play_arena_training(client):
    c, _, _ = client
    resp = c.get("/api/schema")

    assert resp.status_code == 200
    training = resp.get_json()["organism_arena_parallel"]["training"]
    assert training["num_envs"]["value"] == 1
    assert training["num_envs"]["max"] > 1
    assert "self-play" in training["num_envs"]["desc"]


def test_schema_no_longer_exposes_ignored_walker_geometry(client):
    c, _, _ = client
    resp = c.get("/api/schema")

    assert resp.status_code == 200
    sim = resp.get_json()["walker_bullet"]["environment"]["sim"]
    assert "body_half_extents" not in sim
    assert sim["timestep"]["value"] > 0


def test_index_walker_card_matches_env_contract(client):
    c, _, _ = client
    resp = c.get("/")

    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Observation: 35-dim" in html
    assert "Action: 10-dim" in html


# ----- training manager error paths -----


def test_train_start_empty_payload(client):
    c, _, _ = client
    # Flask returns 400 on empty JSON; our handler also enforces it
    resp = c.post("/api/train/start", data="{}", content_type="application/json")
    assert resp.status_code == 400


def test_train_start_invalid_config_returns_400_not_500(client):
    """validate_experiment_config exceptions must surface as 400, not 500."""
    c, _, _ = client
    # Missing required keys -> KeyError inside validate_experiment_config
    bad_cfg = {"experiment_name": "test"}
    resp = c.post("/api/train/start", json=bad_cfg)
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_train_start_invalid_seed_type_returns_400(client):
    """seed must be int; sending a string triggers TypeError -> 400."""
    c, _, _ = client
    bad_cfg = {
        "experiment_name": "test",
        "seed": "not_an_int",
        "output": {"base_dir": "outputs"},
        "environment": {"type": "walker_bullet"},
        "training": {"total_timesteps": 1000},
    }
    resp = c.post("/api/train/start", json=bad_cfg)
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_train_start_rejects_stopping_run(client):
    """A draining stop request still owns the single training slot."""
    import threading
    from rl_framework.gui import app as gui_app
    from rl_framework.gui.training_manager import _RunState

    c, _, _ = client
    state = _RunState(
        run_id="run_stopping",
        cfg={"experiment_name": "exp"},
        status="stopping",
        stop_event=threading.Event(),
    )
    gui_app.manager._runs["run_stopping"] = state

    resp = c.post("/api/train/start", json=_minimal_cfg("next_run"))

    assert resp.status_code == 409
    assert "already active" in resp.get_json()["error"]


def test_train_stop_unknown_run(client):
    c, _, _ = client
    resp = c.post("/api/train/stop/does_not_exist")
    assert resp.status_code == 404


def test_train_status_unknown_run(client):
    c, _, _ = client
    resp = c.get("/api/train/status/does_not_exist")
    assert resp.status_code == 404


def test_train_status_known_run_returns_200(client):
    import threading
    from rl_framework.gui import app as gui_app
    from rl_framework.gui.training_manager import _RunState

    c, _, _ = client
    state = _RunState(
        run_id="run_known",
        cfg={"experiment_name": "exp"},
        status="failed",
        error="traceback text",
        stop_event=threading.Event(),
    )
    gui_app.manager._runs["run_known"] = state

    resp = c.get("/api/train/status/run_known")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["run_id"] == "run_known"
    assert data["status"] == "failed"
    assert data["error"] == "traceback text"


def test_train_stop_already_stopped_run_returns_409(client):
    """Stopping a non-running run returns 409."""
    import threading
    from rl_framework.gui import app as gui_app
    from rl_framework.gui.training_manager import _RunState

    c, _, _ = client
    state = _RunState(
        run_id="run_done",
        cfg={"experiment_name": "exp"},
        status="completed",
        stop_event=threading.Event(),
    )
    gui_app.manager._runs["run_done"] = state
    resp = c.post("/api/train/stop/run_done")
    assert resp.status_code == 409


def test_train_tune_unknown_run(client):
    c, _, _ = client
    resp = c.post("/api/train/tune/does_not_exist", json={"learning_rate": 1e-4})
    assert resp.status_code == 400


def test_training_manager_sigterm_chains_previous_handler() -> None:
    import signal
    from rl_framework.gui.training_manager import TrainingManager

    called: list[int] = []
    manager = TrainingManager()
    manager._previous_sigterm_handler = lambda signum, _frame: called.append(signum)

    manager._sigterm_handler(signal.SIGTERM, None)

    assert called == [signal.SIGTERM]


def test_training_manager_sigterm_exits_for_default_handler() -> None:
    import signal
    from rl_framework.gui.training_manager import TrainingManager

    manager = TrainingManager()
    manager._previous_sigterm_handler = signal.SIG_DFL

    with pytest.raises(SystemExit) as exc:
        manager._sigterm_handler(signal.SIGTERM, None)

    assert exc.value.code == 128 + signal.SIGTERM


def test_list_runs_initially_empty(client):
    c, _, _ = client
    resp = c.get("/api/train/runs")
    assert resp.status_code == 200
    assert resp.get_json() == []


# ----- outputs -----


def test_list_outputs_empty_when_missing(client):
    c, _, _ = client
    resp = c.get("/api/outputs")
    assert resp.status_code == 200
    assert resp.get_json() == []


def test_list_outputs_lists_experiments(client, tmp_path):
    c, _, base = client
    outputs = base / "outputs"
    seed_dir = outputs / "exp1" / "seed_0" / "checkpoints"
    seed_dir.mkdir(parents=True)
    (seed_dir / "final_model.zip").write_bytes(b"")
    resp = c.get(f"/api/outputs?base_dir={outputs}")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data) == 1
    assert data[0]["experiment"] == "exp1"
    assert data[0]["run_id"] is None
    assert data[0]["has_final_model"] is True


def test_list_outputs_includes_sweep_morph_run_variants(client, tmp_path):
    c, _, base = client
    outputs = base / "outputs"
    # Plain seed run directly under the experiment.
    (outputs / "exp1" / "seed_0" / "checkpoints").mkdir(parents=True)
    # A sweep/morph variant nested under runs/<run_id>/seed_<seed>/.
    variant = outputs / "exp1" / "runs" / "lr_0.001" / "seed_0" / "checkpoints"
    variant.mkdir(parents=True)
    (variant / "final_model.zip").write_bytes(b"")

    resp = c.get(f"/api/outputs?base_dir={outputs}")
    assert resp.status_code == 200
    data = resp.get_json()

    plain = [d for d in data if d["run_id"] is None]
    nested = [d for d in data if d["run_id"] == "lr_0.001"]
    assert len(plain) == 1 and plain[0]["experiment"] == "exp1"
    assert len(nested) == 1
    assert nested[0]["experiment"] == "exp1"
    assert nested[0]["seed"] == "seed_0"
    assert nested[0]["path"] == "exp1/runs/lr_0.001/seed_0"
    assert nested[0]["has_final_model"] is True


# ----- registry-backed analysis -----


def test_analysis_runs_reads_registry(client):
    c, _, base = client
    from rl_framework.utils.run_registry import RunRegistry

    run_dir = base / "outputs" / "walker" / "seed_0"
    registry = RunRegistry(base / "outputs")
    cfg = {
        "experiment_name": "walker", "seed": 0,
        "output": {"base_dir": str(base / "outputs")},
        "environment": {"type": "walker_bullet"},
        "training": {"algorithm": "PPO"},
    }
    registry.register_run("run_analysis", cfg, run_dir)
    registry.update_run("run_analysis", status="completed", metrics={"score": 4.0})
    artifact = run_dir / "checkpoints" / "best_model.zip"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"")
    registry.record_artifact("run_analysis", "checkpoint", artifact)

    response = c.get("/api/analysis/runs")
    assert response.status_code == 200
    run = response.get_json()[0]
    assert run["run_id"] == "run_analysis"
    assert run["metrics"] == {"score": 4.0}
    assert run["artifacts"][0]["path"].endswith("best_model.zip")


def test_analysis_replay_rejects_invalid_or_unmanifested_path(client):
    c, _, base = client
    assert c.post("/api/analysis/replay", json={"path": "../../etc"}).status_code == 400
    (base / "outputs" / "exp" / "seed_0").mkdir(parents=True)
    response = c.post("/api/analysis/replay", json={"path": "exp/seed_0"})
    assert response.status_code == 400


def test_league_ratings_requires_two_snapshots_and_metadata(client):
    c, _, base = client
    seed_dir = base / "outputs" / "arena" / "seed_0"
    _make_league(seed_dir, [100])
    response = c.post("/api/analysis/league-ratings", json={"path": "arena/seed_0"})
    assert response.status_code == 400


# ----- self-play league dashboard -----


def _make_league(seed_dir, timesteps, with_vecnorm=()):
    league = seed_dir / "checkpoints" / "league"
    league.mkdir(parents=True, exist_ok=True)
    for ts in timesteps:
        (league / f"selfplay_{ts}.zip").write_bytes(b"x")
        if ts in with_vecnorm:
            (league / f"selfplay_{ts}_vecnorm.pkl").write_bytes(b"x")
    return league


def test_list_outputs_reports_league_size(client):
    c, _, base = client
    seed_dir = base / "outputs" / "arena" / "seed_0"
    (seed_dir / "checkpoints").mkdir(parents=True)
    _make_league(seed_dir, [256, 512, 768])
    # A stray non-numeric snapshot must not be counted.
    (seed_dir / "checkpoints" / "league" / "selfplay_best.zip").write_bytes(b"x")

    data = c.get("/api/outputs").get_json()
    arena = next(d for d in data if d["experiment"] == "arena")
    assert arena["league_size"] == 3


def test_list_outputs_league_size_zero_without_league(client):
    c, _, base = client
    seed_dir = base / "outputs" / "walker" / "seed_0"
    (seed_dir / "checkpoints").mkdir(parents=True)
    data = c.get("/api/outputs").get_json()
    assert next(d for d in data if d["experiment"] == "walker")["league_size"] == 0


def test_get_league_returns_sorted_snapshot_detail(client):
    c, _, base = client
    seed_dir = base / "outputs" / "arena" / "seed_0"
    (seed_dir / "checkpoints").mkdir(parents=True)
    _make_league(seed_dir, [768, 256, 512], with_vecnorm=(512,))

    resp = c.get("/api/league?path=arena/seed_0")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["league_size"] == 3
    assert data["newest_timesteps"] == 768
    # Sorted oldest -> newest by timestep regardless of glob order.
    assert [s["timesteps"] for s in data["snapshots"]] == [256, 512, 768]
    by_ts = {s["timesteps"]: s for s in data["snapshots"]}
    assert by_ts[512]["has_vecnorm"] is True
    assert by_ts[256]["has_vecnorm"] is False
    assert all(s["age_seconds"] >= 0 for s in data["snapshots"])


def test_get_league_empty_for_unknown_run(client):
    c, _, _ = client
    resp = c.get("/api/league?path=nope/seed_0")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["league_size"] == 0 and data["snapshots"] == []
    assert data["newest_timesteps"] is None


def test_get_league_rejects_path_traversal(client):
    c, _, _ = client
    assert c.get("/api/league?path=../../etc").status_code == 404
    assert c.get("/api/league?path=").status_code == 404


# ----- frames endpoint -----


def _inject_run_with_callback(manager, run_id: str, callback=None) -> None:
    """Insert a synthetic _RunState directly into manager._runs."""
    import threading
    from rl_framework.gui.training_manager import _RunState

    state = _RunState(
        run_id=run_id,
        cfg={"experiment_name": "test"},
        status="running",
        stop_event=threading.Event(),
        frame_capture_callback=callback,
    )
    manager._runs[run_id] = state


def test_frames_unknown_run_returns_404(client) -> None:
    c, _, _ = client
    resp = c.get("/api/train/frames/no_such_run")
    assert resp.status_code == 404


def test_frames_run_without_callback_returns_empty_list(client) -> None:
    from rl_framework.gui import app as gui_app

    c, _, _ = client
    _inject_run_with_callback(gui_app.manager, "run_no_cb", callback=None)

    resp = c.get("/api/train/frames/run_no_cb")
    assert resp.status_code == 200
    assert resp.get_json() == {"frames": []}


def test_frames_returns_captured_frames(client) -> None:
    from unittest.mock import MagicMock
    from rl_framework.gui import app as gui_app

    c, _, _ = client

    fake_cb = MagicMock()
    fake_cb.get_frames.return_value = [
        {"frame_index": 0, "timestep": 50, "episode_num": 0, "image_base64": "abc"},
        {"frame_index": 1, "timestep": 100, "episode_num": 0, "image_base64": "def"},
    ]
    _inject_run_with_callback(gui_app.manager, "run_with_frames", callback=fake_cb)

    resp = c.get("/api/train/frames/run_with_frames")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["frames"]) == 2
    assert data["frames"][0]["frame_index"] == 0


def test_frames_since_parameter_forwarded_to_callback(client) -> None:
    from unittest.mock import MagicMock
    from rl_framework.gui import app as gui_app

    c, _, _ = client

    fake_cb = MagicMock()
    fake_cb.get_frames.return_value = []
    _inject_run_with_callback(gui_app.manager, "run_since", callback=fake_cb)

    resp = c.get("/api/train/frames/run_since?since=7")
    assert resp.status_code == 200
    fake_cb.get_frames.assert_called_once_with(since=7)


def test_frames_since_defaults_to_zero(client) -> None:
    from unittest.mock import MagicMock
    from rl_framework.gui import app as gui_app

    c, _, _ = client

    fake_cb = MagicMock()
    fake_cb.get_frames.return_value = []
    _inject_run_with_callback(gui_app.manager, "run_default_since", callback=fake_cb)

    resp = c.get("/api/train/frames/run_default_since")
    assert resp.status_code == 200
    fake_cb.get_frames.assert_called_once_with(since=0)


def test_frames_since_invalid_value_falls_back_to_zero(client) -> None:
    from unittest.mock import MagicMock
    from rl_framework.gui import app as gui_app

    c, _, _ = client

    fake_cb = MagicMock()
    fake_cb.get_frames.return_value = []
    _inject_run_with_callback(gui_app.manager, "run_bad_since", callback=fake_cb)

    resp = c.get("/api/train/frames/run_bad_since?since=notanumber")
    assert resp.status_code == 200
    fake_cb.get_frames.assert_called_once_with(since=0)
