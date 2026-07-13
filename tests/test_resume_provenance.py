"""Tests for resume compatibility / provenance checking.

check_resume_provenance compares a resuming run's config against the source
checkpoint's run_metadata.json and flags silent drift that Stable-Baselines3's
own space check would miss (changed environment, flipped normalization, etc.).
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from rl_framework.utils.reproducibility import (
    check_resume_provenance,
    write_run_metadata,
)


def _source_cfg() -> dict:
    return {
        "experiment_name": "exp",
        "seed": 1,
        "output": {"base_dir": "outputs"},
        "environment": {
            "type": "walker_bullet",
            "seed": 1,
            "reward": {"forward_velocity_weight": 1.5},
        },
        "training": {
            "policy": "MlpPolicy",
            "total_timesteps": 1000,
            "learning_rate": 3e-4,
            "normalize_observations": True,
            "gamma": 0.99,
        },
    }


def _make_checkpoint(tmp_path: Path, source_cfg: dict) -> Path:
    """Lay out <run>/checkpoints/final_model.zip + <run>/run_metadata.json."""
    run_dir = tmp_path / "exp" / "seed_1"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    model_path = ckpt_dir / "final_model.zip"
    model_path.write_bytes(b"")
    lockfile = tmp_path / "requirements-lock.txt"
    lockfile.write_text("numpy==1.0.0\n", encoding="utf-8")
    write_run_metadata(run_dir, source_cfg, strict=False, lockfile_path=lockfile)
    return model_path


def test_matching_config_has_no_drift(tmp_path: Path) -> None:
    model_path = _make_checkpoint(tmp_path, _source_cfg())
    # Resuming with more timesteps + a different LR is fine — neither is provenance.
    cfg = copy.deepcopy(_source_cfg())
    cfg["training"]["total_timesteps"] = 5000
    cfg["training"]["learning_rate"] = 1e-4
    assert check_resume_provenance(model_path, cfg, strict=True) == []


def test_seed_changes_are_not_drift(tmp_path: Path) -> None:
    model_path = _make_checkpoint(tmp_path, _source_cfg())
    cfg = copy.deepcopy(_source_cfg())
    cfg["seed"] = 99
    cfg["environment"]["seed"] = 99
    assert check_resume_provenance(model_path, cfg, strict=True) == []


def test_extensionless_resume_path_resolves_to_zip(tmp_path: Path) -> None:
    model_path = _make_checkpoint(tmp_path, _source_cfg())
    stem = model_path.with_suffix("")  # drop .zip
    assert check_resume_provenance(stem, _source_cfg(), strict=True) == []


def test_changed_environment_type_is_strict_failure(tmp_path: Path) -> None:
    model_path = _make_checkpoint(tmp_path, _source_cfg())
    cfg = copy.deepcopy(_source_cfg())
    cfg["environment"]["type"] = "organism_arena_parallel"
    with pytest.raises(RuntimeError, match="environment.type"):
        check_resume_provenance(model_path, cfg, strict=True)


def test_changed_reward_warns_but_does_not_raise_when_not_strict(
    tmp_path: Path,
) -> None:
    model_path = _make_checkpoint(tmp_path, _source_cfg())
    cfg = copy.deepcopy(_source_cfg())
    cfg["environment"]["reward"]["forward_velocity_weight"] = 99.0
    drift = check_resume_provenance(model_path, cfg, strict=False)
    assert any("environment.reward.forward_velocity_weight" in d for d in drift)


def test_flipped_normalization_is_strict_failure(tmp_path: Path) -> None:
    model_path = _make_checkpoint(tmp_path, _source_cfg())
    cfg = copy.deepcopy(_source_cfg())
    cfg["training"]["normalize_observations"] = False
    with pytest.raises(RuntimeError, match="normalize_observations"):
        check_resume_provenance(model_path, cfg, strict=True)


def test_changed_algorithm_is_strict_failure(tmp_path: Path) -> None:
    model_path = _make_checkpoint(tmp_path, _source_cfg())
    cfg = copy.deepcopy(_source_cfg())
    cfg["training"]["algorithm"] = "SAC"
    with pytest.raises(RuntimeError, match="training.algorithm"):
        check_resume_provenance(model_path, cfg, strict=True)


def test_missing_manifest_is_unverifiable(tmp_path: Path) -> None:
    # Checkpoint with no run_metadata.json anywhere near it.
    ckpt_dir = tmp_path / "loose" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    model_path = ckpt_dir / "final_model.zip"
    model_path.write_bytes(b"")

    # Non-strict: returns an unverifiable message, no raise.
    msgs = check_resume_provenance(model_path, _source_cfg(), strict=False)
    assert msgs and "cannot verify resume provenance" in msgs[0]

    # Strict: unverifiable provenance is a hard failure.
    with pytest.raises(RuntimeError, match="cannot verify resume provenance"):
        check_resume_provenance(model_path, _source_cfg(), strict=True)
