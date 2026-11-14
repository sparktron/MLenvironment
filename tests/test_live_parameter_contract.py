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


def _terrain_worker():
    return make_env("walker_bullet", {"type": "walker_bullet", "terrain": {"preset": "flat"}})


def test_terrain_live_update_applies_on_next_reset_in_both_workers() -> None:
    import numpy as np
    from stable_baselines3.common.vec_env import SubprocVecEnv

    vec = SubprocVecEnv([_terrain_worker, _terrain_worker], start_method="spawn")
    try:
        vec.reset()
        vec.env_method("update_live_params", {"terrain.mix": [{"preset": "uneven", "height": 0.02}]})
        _, _, _, infos = vec.step(np.zeros((2, 10)))
        assert [info["terrain_preset"] for info in infos] == ["flat", "flat"]
        vec.reset()
        _, _, _, infos = vec.step(np.zeros((2, 10)))
        assert [info["terrain_preset"] for info in infos] == ["uneven", "uneven"]
    finally:
        vec.close()


def test_terrain_live_update_rejects_empty_mix() -> None:
    env = _terrain_worker()
    try:
        with pytest.raises(ValueError, match="at least one"):
            env.update_live_params({"terrain.mix": []})
    finally:
        env.close()


def test_live_frame_skip_updates_reported_gait_timing() -> None:
    import numpy as np

    env = _terrain_worker()
    try:
        env.reset(seed=1)
        tracker = env._gait_tracker
        tracker.touchdown_debounce_steps = 1

        def swing_duration():
            tracker.reset(initial_contacts=(True, True), action_size=10)
            for step, contact in [(1, False), (2, False), (3, True)]:
                tracker.update(
                    step=step, contacts=(contact, True), pelvis_x=0.0,
                    foot_slip_speeds=(0.0, 0.0),
                    foot_positions=((step * 0.1, 0.1 if not contact else 0.0), (0.0, 0.0)),
                    applied_action=np.zeros(10),
                )
            return tracker.episode_metrics()["gait_swing_duration"]

        baseline = swing_duration()
        assert baseline > 0
        env.update_live_params({"sim.frame_skip": 8})
        assert swing_duration() == pytest.approx(2 * baseline)
        env.update_live_params({"sim.timestep": 1.0 / 120.0})
        assert swing_duration() == pytest.approx(4 * baseline)
    finally:
        env.close()


def test_swing_spanning_live_frame_skip_change_uses_both_timesteps() -> None:
    import numpy as np

    env = _terrain_worker()
    try:
        env.reset(seed=1)
        tracker = env._gait_tracker
        tracker.touchdown_debounce_steps = 1
        tracker.reset(initial_contacts=(True, True), action_size=10)
        for step, contact in [(1, False), (2, False), (3, True)]:
            if step == 3:
                env.update_live_params({"sim.frame_skip": 8})
            tracker.update(
                step=step, contacts=(contact, True), pelvis_x=0.0,
                foot_slip_speeds=(0.0, 0.0),
                foot_positions=((step * 0.1, 0.1 if not contact else 0.0), (0.0, 0.0)),
                applied_action=np.zeros(10),
            )
        assert tracker.episode_metrics()["gait_swing_duration"] == pytest.approx(3 / 60)
        assert tracker.episode_metrics()["gait_cadence_steps_per_min"] == pytest.approx(900)
    finally:
        env.close()
