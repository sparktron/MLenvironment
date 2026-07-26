import pytest

from rl_framework.utils.config import (
    load_config,
    to_container,
    validate_experiment_config,
)


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


def test_load_config_rejects_non_mapping_yaml(tmp_path) -> None:
    (tmp_path / "list_config.yaml").write_text("- not\n- a\n- mapping\n")

    with pytest.raises(TypeError, match="must be a mapping"):
        load_config("list_config", tmp_path)


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


@pytest.mark.parametrize(
    "bad_name",
    [
        "../escape",
        "foo/../bar",
        "/absolute/path",
        "sub/dir",
        "back\\slash",
        "",
        "   ",
    ],
)
def test_validate_experiment_config_rejects_unsafe_experiment_name(
    bad_name: str,
) -> None:
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


def test_validate_experiment_config_rejects_invalid_annealing_gate() -> None:
    cfg = _base_cfg()
    cfg["reward_annealing"] = {
        "enabled": True,
        "anneal_steps": 100,
        "min_eliminations": -1,
    }
    with pytest.raises(ValueError, match="min_eliminations"):
        validate_experiment_config(cfg)


@pytest.mark.parametrize("value", [0, -0.1, 1.1, True, "0.5"])
def test_validate_walker_rejects_invalid_action_scale(value) -> None:
    cfg = _base_cfg()
    cfg["environment"]["sim"] = {"action_scale": value}

    with pytest.raises(ValueError, match="action_scale"):
        validate_experiment_config(cfg)


def test_validate_curriculum_rejects_invalid_behavior_condition() -> None:
    cfg = _base_cfg()
    cfg["curriculum"] = {
        "enabled": True,
        "level_params": {1: {"sim.action_scale": 0.5}},
        "level_up_conditions": {
            0: {"walker/fall_rate": {"min": 0.5, "max": 0.1}}
        },
    }

    with pytest.raises(ValueError, match="min must not exceed max"):
        validate_experiment_config(cfg)


def test_validate_experiment_config_rejects_non_positive_learning_rate() -> None:
    cfg = _base_cfg()
    cfg["training"]["learning_rate"] = 0
    with pytest.raises(ValueError, match="training.learning_rate"):
        validate_experiment_config(cfg)


@pytest.mark.parametrize("value", [-21.0, 2.1, True, "low"])
def test_validate_experiment_config_rejects_invalid_log_std_init(value) -> None:
    cfg = _base_cfg()
    cfg["training"]["log_std_init"] = value

    with pytest.raises(ValueError, match="training.log_std_init"):
        validate_experiment_config(cfg)


def test_validate_experiment_config_rejects_log_std_init_for_sac() -> None:
    cfg = _base_cfg()
    cfg["training"]["algorithm"] = "SAC"
    cfg["training"]["log_std_init"] = -1.0

    with pytest.raises(ValueError, match="supported only for PPO"):
        validate_experiment_config(cfg)


@pytest.mark.parametrize("value", [-0.01, True, "zero"])
def test_validate_experiment_config_rejects_invalid_entropy_coefficient(value) -> None:
    cfg = _base_cfg()
    cfg["training"]["ent_coef"] = value

    with pytest.raises(ValueError, match="training.ent_coef"):
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


def test_validate_experiment_config_rejects_invalid_deterministic_settings() -> None:
    cfg = _base_cfg()
    cfg["reproducibility"] = {"deterministic": True}
    cfg["training"]["torch_num_threads"] = 2
    with pytest.raises(ValueError, match="deterministic.*torch_num_threads"):
        validate_experiment_config(cfg)

    cfg["training"]["torch_num_threads"] = 1
    cfg["training"]["worker_start_method"] = "fork"
    with pytest.raises(ValueError, match="deterministic.*worker_start_method"):
        validate_experiment_config(cfg)


def test_validate_experiment_config_rejects_best_model_for_arena() -> None:
    cfg = _base_cfg()
    cfg["environment"] = {"type": "organism_arena_parallel"}
    cfg["evaluation"]["best_model"] = {"enabled": True}
    with pytest.raises(ValueError, match="best_model.*walker_bullet"):
        validate_experiment_config(cfg)


def test_validate_experiment_config_rejects_non_bool_check_nans() -> None:
    cfg = _base_cfg()
    cfg["training"]["check_nans"] = "true"
    with pytest.raises(TypeError, match="training.check_nans"):
        validate_experiment_config(cfg)


@pytest.mark.parametrize("value", [0, -1, True, "2"])
def test_validate_experiment_config_rejects_invalid_torch_thread_count(value) -> None:
    cfg = _base_cfg()
    cfg["training"]["torch_num_threads"] = value
    with pytest.raises((TypeError, ValueError), match="torch_num_threads"):
        validate_experiment_config(cfg)


def test_validate_experiment_config_rejects_invalid_worker_start_method() -> None:
    cfg = _base_cfg()
    cfg["training"]["worker_start_method"] = "invalid"
    with pytest.raises(ValueError, match="worker_start_method"):
        validate_experiment_config(cfg)


@pytest.mark.parametrize(
    "name", ["walker_ppo_baseline", "walker_sac_baseline", "walker_td3_baseline"]
)
def test_off_policy_walker_baselines_validate(name: str) -> None:
    cfg = to_container(load_config(name, "src/rl_framework/configs/experiments"))
    validate_experiment_config(cfg)


@pytest.mark.parametrize(
    ("section", "field", "value", "match"),
    [
        ("battle_rules", "collision_damage", -0.1, "collision_damage"),
        ("resources", "food_placement", "corners", "food_placement"),
    ],
)
def test_arena_rejects_invalid_strategic_depth_settings(
    section: str, field: str, value, match: str
) -> None:
    cfg = _base_cfg()
    cfg["environment"] = {
        "type": "organism_arena_parallel",
        "battle_rules": {"damage": 0.05, "attack_range": 0.2},
        "resources": {},
    }
    cfg["environment"][section][field] = value

    with pytest.raises(ValueError, match=match):
        validate_experiment_config(cfg)


def test_arena_rejects_negative_approach_reward_weight() -> None:
    cfg = _base_cfg()
    cfg["environment"] = {
        "type": "organism_arena_parallel",
        "reward": {"approach_weight": -0.1},
    }

    with pytest.raises(ValueError, match="approach_weight"):
        validate_experiment_config(cfg)


def test_off_policy_algorithms_reject_arena() -> None:
    cfg = _base_cfg()
    cfg["environment"] = {"type": "organism_arena_parallel"}
    cfg["training"]["algorithm"] = "SAC"
    with pytest.raises(ValueError, match="walker_bullet"):
        validate_experiment_config(cfg)


def test_validate_experiment_config_rejects_multi_env_shared_policy_arena() -> None:
    cfg = _base_cfg()
    cfg["environment"] = {"type": "organism_arena_parallel"}
    cfg["training"]["num_envs"] = 8
    with pytest.raises(ValueError, match="Shared-policy.*num_envs == 1"):
        validate_experiment_config(cfg)


def test_validate_experiment_config_accepts_multi_env_self_play_arena() -> None:
    cfg = _base_cfg()
    cfg["environment"] = {"type": "organism_arena_parallel"}
    cfg["training"]["num_envs"] = 8
    cfg["self_play"] = {"enabled": True, "snapshot_freq": 128, "max_league_size": 5}

    validate_experiment_config(cfg)


def test_shipped_parallel_self_play_arena_config_validates() -> None:
    cfg = to_container(
        load_config("organisms_fight_arena", "src/rl_framework/configs/experiments")
    )

    validate_experiment_config(cfg)


def test_robot_push_recovery_config_uses_current_atlas_body() -> None:
    """robot_push_recovery.yaml pre-dated the Atlas-class overhaul and still
    shipped a 3.2 kg torso (lighter than a single 7 kg thigh) with a 45.0
    max_force reinterpreted as a 1.29x global torque scale over the current
    per-joint caps. Regression-pins the migrated values so the config cannot
    silently drift back to the stale pre-overhaul body."""
    cfg = to_container(
        load_config("robot_push_recovery", "src/rl_framework/configs/experiments")
    )

    validate_experiment_config(cfg)
    sim = cfg["environment"]["sim"]
    assert sim["mass"] == 28.0, "torso mass must match the current Atlas-class default"
    assert sim["max_force"] == 35.0, (
        "max_force must match the current 1.0x per-joint torque baseline"
    )
    assert cfg["environment"]["terrain"]["preset"] == "push_recovery"


@pytest.mark.parametrize(
    "name",
    [
        "walker_balance_curriculum",
        "walker_low_velocity_candidate",
        "walker_stochastic_balance_candidate",
        "walker_curriculum_flat",
        "walker_curriculum_uneven",
        "walker_curriculum_obstacles",
    ],
)
def test_walker_curriculum_presets_validate(name: str) -> None:
    cfg = to_container(load_config(name, "src/rl_framework/configs/experiments"))
    validate_experiment_config(cfg)
