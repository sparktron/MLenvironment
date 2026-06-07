from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl_framework.utils import reproducibility
from rl_framework.utils.reproducibility import write_run_metadata


def _cfg() -> dict:
    return {
        "experiment_name": "exp",
        "seed": 7,
        "output": {"base_dir": "outputs"},
        "environment": {"type": "walker_bullet"},
        "training": {"total_timesteps": 1000},
    }


def test_write_run_metadata_writes_manifest(tmp_path: Path) -> None:
    lockfile = tmp_path / "requirements-lock.txt"
    lockfile.write_text("numpy==1.0.0\n", encoding="utf-8")

    out = write_run_metadata(tmp_path / "run", _cfg(), strict=False, lockfile_path=lockfile)
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert out.exists()
    assert payload["experiment_name"] == "exp"
    assert payload["seed"] == 7
    assert payload["config_hash_sha256"]
    assert payload["lockfile"]["sha256"]
    assert payload["strict_reproducibility"] is False


def test_write_run_metadata_strict_requires_lockfile_hash(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Missing lockfile hash"):
        write_run_metadata(
            tmp_path / "run",
            _cfg(),
            strict=True,
            lockfile_path=tmp_path / "does-not-exist.txt",
        )


def test_write_run_metadata_strict_requires_git_commit(tmp_path: Path, monkeypatch) -> None:
    lockfile = tmp_path / "requirements-lock.txt"
    lockfile.write_text("numpy==1.0.0\n", encoding="utf-8")
    monkeypatch.setattr(
        "rl_framework.utils.reproducibility._git_info",
        lambda: {"commit": None, "branch": None, "is_dirty": None},
    )

    with pytest.raises(RuntimeError, match="Missing git commit"):
        write_run_metadata(tmp_path / "run", _cfg(), strict=True, lockfile_path=lockfile)


def test_git_dirty_false_for_empty_status(monkeypatch) -> None:
    monkeypatch.setattr(
        reproducibility.subprocess,
        "check_output",
        lambda *_args, **_kwargs: "",
    )

    assert reproducibility._git_dirty() is False


def test_git_dirty_true_for_staged_changes(monkeypatch) -> None:
    monkeypatch.setattr(
        reproducibility.subprocess,
        "check_output",
        lambda *_args, **_kwargs: "M  src/example.py\n",
    )

    assert reproducibility._git_dirty() is True


def test_git_dirty_true_for_untracked_files(monkeypatch) -> None:
    monkeypatch.setattr(
        reproducibility.subprocess,
        "check_output",
        lambda *_args, **_kwargs: "?? scratch.txt\n",
    )

    assert reproducibility._git_dirty() is True
