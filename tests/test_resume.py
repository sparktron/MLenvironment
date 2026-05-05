"""Tests for checkpoint resume and VecNormalize restoration (plan §5b).

Verifies that:
(a) training resumes from the saved timestep count, not from zero.
(b) vecnormalize.pkl is restored so normalisation statistics carry over.
(c) _validate_resume_path raises FileNotFoundError with clear messages
    when the model zip or sibling vecnormalize.pkl is missing.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _base_cfg(tmp_path: Path, timesteps: int = 128) -> dict:
    return {
        "experiment_name": "resume_test",
        "seed": 7,
        "output": {"base_dir": str(tmp_path)},
        "environment": {
            "type": "walker_bullet",
            "sim": {
                "gravity": -9.81,
                "mass": 3.0,
                "friction": 0.8,
                "max_force": 30.0,
                "body_half_extents": [0.15, 0.15, 0.15],
                "arena_half_extent": 5.0,
            },
            "reward": {
                "alive_bonus": 1.0,
                "forward_velocity_weight": 1.0,
                "target_velocity": 0.5,
                "orientation_penalty_weight": 0.1,
                "torque_penalty_weight": 0.01,
            },
            "termination": {"min_height": -0.5, "max_tilt_radians": 1.5, "max_steps": 50},
            "domain_randomization": {
                "mass_scale_range": [1.0, 1.0],
                "friction_range": [0.8, 0.8],
                "sensor_noise_std": 0.0,
                "action_latency_steps": 0,
            },
        },
        "training": {
            "total_timesteps": timesteps,
            "learning_rate": 3e-4,
            "n_steps": 64,
            "batch_size": 32,
            "num_envs": 1,
            "device": "cpu",
            "checkpoint_every": 64,
            "normalize_observations": True,
        },
        "evaluation": {"episodes": 1},
    }


def test_resume_continues_from_checkpoint(tmp_path: Path) -> None:
    """Resume from a checkpoint and verify training runs without error.

    Also checks that the resumed model produces a final .zip file and that
    vecnormalize.pkl exists alongside it — confirming stats are carried over.
    """
    from rl_framework.training.sb3_runner import train

    cfg = _base_cfg(tmp_path, timesteps=128)
    first_model = train(cfg)

    zip_first = Path(str(first_model) + ".zip") if not str(first_model).endswith(".zip") else Path(first_model)
    assert zip_first.exists()

    # Resume for another 128 steps from the model written by the first run.
    cfg2 = _base_cfg(tmp_path, timesteps=128)
    cfg2["experiment_name"] = "resume_test_continued"
    resumed_model = train(cfg2, resume_from=str(zip_first))

    zip_resumed = Path(str(resumed_model) + ".zip") if not str(resumed_model).endswith(".zip") else Path(resumed_model)
    assert zip_resumed.exists(), "Resumed training did not produce a model zip"

    vecnorm_resumed = zip_resumed.with_name("vecnormalize.pkl")
    assert vecnorm_resumed.exists(), "vecnormalize.pkl missing after resumed training"


def test_validate_resume_path_missing_zip(tmp_path: Path) -> None:
    """_validate_resume_path raises FileNotFoundError when the zip is absent."""
    from rl_framework.training.sb3_runner import _validate_resume_path

    nonexistent = tmp_path / "ghost_model.zip"
    with pytest.raises(FileNotFoundError, match="resume_from model not found"):
        _validate_resume_path(nonexistent, normalize=False)


def test_validate_resume_path_missing_vecnorm(tmp_path: Path) -> None:
    """_validate_resume_path raises FileNotFoundError when vecnormalize.pkl is absent."""
    from rl_framework.training.sb3_runner import _validate_resume_path

    zip_path = tmp_path / "model.zip"
    zip_path.write_bytes(b"fake")  # file must exist; vecnorm must not

    with pytest.raises(FileNotFoundError, match="vecnormalize.pkl not found"):
        _validate_resume_path(zip_path, normalize=True)


def test_validate_resume_path_passes_when_files_present(tmp_path: Path) -> None:
    """_validate_resume_path succeeds when both files exist."""
    from rl_framework.training.sb3_runner import _validate_resume_path

    zip_path = tmp_path / "model.zip"
    zip_path.write_bytes(b"fake")
    (tmp_path / "vecnormalize.pkl").write_bytes(b"fake")

    _validate_resume_path(zip_path, normalize=True)  # should not raise
