from pathlib import Path

import pytest
import yaml

from rl_framework.core.registry import get_task, make_env


def test_shipped_live_parameter_keys_are_declared() -> None:
    config_dir = Path("src/rl_framework/configs/experiments")
    for path in sorted(config_dir.glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        spec = get_task(cfg["environment"]["type"])
        keys = {key for params in (cfg.get("curriculum") or {}).get("level_params", {}).values()
                for key in params}
        if (cfg.get("training") or {}).get("velocity_target_ramp"):
            keys.update({"reward.target_velocity", "reward.velocity_sigma"})
        if (cfg.get("reward_annealing") or {}).get("enabled"):
            keys.add("reward.damage_scale")
        assert keys <= spec.live_params, f"{path}: undeclared {keys - spec.live_params}"


@pytest.mark.parametrize("task", ["walker_bullet", "organism_arena_parallel"])
def test_live_updates_reject_unknown_and_invalid_values(task: str) -> None:
    env = make_env(task, {"type": task})
    try:
        with pytest.raises(ValueError, match="valid"):
            env.update_live_params({"reward.misspelled": 1})
        key = "reward.alive_bonus" if task == "walker_bullet" else "battle_rules.damage"
        with pytest.raises((TypeError, ValueError)):
            env.update_live_params({key: "not-a-number"})
    finally:
        env.close()


def test_callbacks_propagate_worker_errors() -> None:
    from types import SimpleNamespace

    from rl_framework.core.training.callbacks.curriculum_callback import CurriculumCallback
    from rl_framework.core.training.callbacks.live_tuning_callback import LiveTuningCallback

    def fail(*args):
        raise ValueError("worker rejected update")

    env = SimpleNamespace(env_method=fail)
    model = SimpleNamespace(get_env=lambda: env)
    curriculum = CurriculumCallback({"level_params": {1: {"reward.alive_bonus": 1}}}, {})
    curriculum.model = model
    with pytest.raises(ValueError, match="worker rejected"):
        curriculum._apply_level_params(1)
    tuning = LiveTuningCallback({"type": "walker_bullet"})
    tuning.model = model
    with pytest.raises(ValueError, match="worker rejected"):
        tuning._apply_params({"reward.alive_bonus": 1})
