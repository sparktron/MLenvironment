"""Tests for the CLI's --json / --json-out automation output plumbing.

These exercise main()'s output handling directly (with the heavy runners
patched out) so the JSON-emitting paths are covered without training a model.
"""

from __future__ import annotations

import json

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
