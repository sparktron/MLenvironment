import pytest

from rl_framework.utils.config import validate_experiment_config


def _base_cfg() -> dict:
    return {
        "experiment_name": "exp",
        "seed": 0,
        "output": {"base_dir": "outputs"},
        "environment": {"type": "walker_bullet"},
        "training": {"total_timesteps": 1000, "num_envs": 1},
        "evaluation": {"episodes": 3},
    }


def test_validate_experiment_config_accepts_minimal_valid_config() -> None:
    validate_experiment_config(_base_cfg())


def test_validate_experiment_config_rejects_missing_required_key() -> None:
    cfg = _base_cfg()
    del cfg["training"]["total_timesteps"]
    with pytest.raises(KeyError, match="training.total_timesteps|total_timesteps"):
        validate_experiment_config(cfg)


def test_validate_experiment_config_rejects_invalid_ranges() -> None:
    cfg = _base_cfg()
    cfg["training"]["num_envs"] = 0
    with pytest.raises(ValueError, match="training.num_envs"):
        validate_experiment_config(cfg)
