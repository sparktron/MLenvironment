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
    assert data["walker_bullet"]["training"]["device"]["value"] == "auto"


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


def test_train_stop_unknown_run(client):
    c, _, _ = client
    resp = c.post("/api/train/stop/does_not_exist")
    assert resp.status_code == 404


def test_train_status_unknown_run(client):
    c, _, _ = client
    resp = c.get("/api/train/status/does_not_exist")
    assert resp.status_code == 404


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
    assert data[0]["has_final_model"] is True


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
