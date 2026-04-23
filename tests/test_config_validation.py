import pytest

from rl_framework.utils.config import validate_experiment_config


def _base_cfg() -> dict:
    return {
        "experiment_name": "exp",
        "seed": 0,
        "output": {"base_dir": "outputs"},
        "environment": {"type": "walker_bullet"},
        "training": {
            "total_timesteps": 1000,
            "num_envs": 1,
            "learning_rate": 3e-4,
            "n_steps": 128,
            "batch_size": 64,
            "checkpoint_every": 100,
        },
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


def test_validate_experiment_config_rejects_non_int_seed() -> None:
    cfg = _base_cfg()
    cfg["seed"] = "0"
    with pytest.raises(TypeError, match="seed"):
        validate_experiment_config(cfg)


@pytest.mark.parametrize("bad_name", [
    "../escape",
    "foo/../bar",
    "/absolute/path",
    "sub/dir",
    "back\\slash",
    "",
])
def test_validate_experiment_config_rejects_unsafe_experiment_name(bad_name: str) -> None:
    cfg = _base_cfg()
    cfg["experiment_name"] = bad_name
    with pytest.raises(ValueError, match="experiment_name"):
        validate_experiment_config(cfg)


def test_validate_experiment_config_rejects_non_string_experiment_name() -> None:
    cfg = _base_cfg()
    cfg["experiment_name"] = 123
    with pytest.raises(ValueError, match="experiment_name"):
        validate_experiment_config(cfg)


def test_validate_experiment_config_rejects_invalid_self_play_values() -> None:
    cfg = _base_cfg()
    cfg["self_play"] = {"enabled": True, "snapshot_freq": 0, "max_league_size": 2}
    with pytest.raises(ValueError, match="self_play.snapshot_freq"):
        validate_experiment_config(cfg)


def test_validate_experiment_config_rejects_non_positive_learning_rate() -> None:
    cfg = _base_cfg()
    cfg["training"]["learning_rate"] = 0
    with pytest.raises(ValueError, match="training.learning_rate"):
        validate_experiment_config(cfg)


def test_validate_experiment_config_rejects_batch_larger_than_rollout() -> None:
    cfg = _base_cfg()
    cfg["training"]["n_steps"] = 32
    cfg["training"]["num_envs"] = 1
    cfg["training"]["batch_size"] = 128
    with pytest.raises(ValueError, match="training.batch_size"):
        validate_experiment_config(cfg)


def test_validate_experiment_config_rejects_non_bool_repro_strict() -> None:
    cfg = _base_cfg()
    cfg["reproducibility"] = {"strict": "yes"}
    with pytest.raises(TypeError, match="reproducibility.strict"):
        validate_experiment_config(cfg)
