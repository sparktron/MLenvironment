from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from rl_framework.envs.registry import make_env
from rl_framework.training.curriculum_callback import CurriculumCallback
from rl_framework.training.self_play_callback import SelfPlayCallback
from rl_framework.utils.logging_utils import create_experiment_paths


class StopOnEvent(BaseCallback):
    """SB3 callback that halts training when a :class:`threading.Event` is set.

    Used by the GUI's ``stop_run()`` to interrupt ``model.learn()`` at the end
    of the current rollout without killing the process.
    """

    def __init__(self, stop_event: threading.Event) -> None:
        super().__init__(verbose=0)
        self._stop_event = stop_event

    def _on_step(self) -> bool:
        return not self._stop_event.is_set()


def _build_single_env(env_cfg: dict[str, Any]):
    return lambda: make_env(env_cfg["type"], env_cfg)


def train(
    cfg: dict[str, Any],
    extra_callbacks: list[BaseCallback] | None = None,
    stop_event: threading.Event | None = None,
    resume_from: str | Path | None = None,
) -> Path:
    """Train a PPO agent from *cfg* and return the path to the saved model.

    Parameters
    ----------
    cfg:
        Experiment config dict.
    extra_callbacks:
        Additional SB3 callbacks inserted after the checkpoint callback and
        before the built-in curriculum / self-play callbacks.  The GUI uses
        this to inject :class:`~rl_framework.training.live_tuning_callback.LiveTuningCallback`.
    stop_event:
        When set, a :class:`StopOnEvent` callback is prepended so training
        halts at the end of the current rollout.
    resume_from:
        Path to a saved PPO model (``.zip`` or the path without extension).
        When provided, the model weights and optimizer state are restored and
        training continues from the saved timestep counter.  If a sibling
        ``vecnormalize.pkl`` exists, its running statistics are also restored.
    """
    paths = create_experiment_paths(cfg["output"]["base_dir"], cfg["experiment_name"], cfg["seed"])
    env_cfg = cfg["environment"]

    num_envs = int(cfg["training"].get("num_envs", 1))

    if env_cfg["type"] == "organism_arena_parallel":
        par_env = make_env(env_cfg["type"], env_cfg)
        vec_env = ss.pettingzoo_env_to_vec_env_v1(par_env)
        vec_env = ss.concat_vec_envs_v1(
            vec_env, max(num_envs, 1), num_cpus=max(num_envs, 1), base_class="stable_baselines3",
        )
    else:
        env_fns = [_build_single_env(env_cfg) for _ in range(max(num_envs, 1))]
        if num_envs > 1:
            vec_env = SubprocVecEnv(env_fns)
        else:
            vec_env = DummyVecEnv(env_fns)

    try:
        normalize = cfg["training"].get("normalize_observations", True)
        if normalize:
            vecnorm_path = None
            if resume_from is not None:
                candidate = Path(resume_from).with_name("vecnormalize.pkl")
                if candidate.exists():
                    vecnorm_path = candidate
            if vecnorm_path is not None:
                vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
                # Keep updating stats and continue training.
                vec_env.training = True
                vec_env.norm_reward = False
            else:
                vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

        if resume_from is not None:
            model = PPO.load(
                str(resume_from),
                env=vec_env,
                tensorboard_log=str(paths.logs_dir),
                device=cfg["training"].get("device", "cuda"),
            )
        else:
            model = PPO(
                policy=cfg["training"].get("policy", "MlpPolicy"),
                env=vec_env,
                learning_rate=cfg["training"].get("learning_rate", 3e-4),
                n_steps=cfg["training"].get("n_steps", 1024),
                batch_size=cfg["training"].get("batch_size", 256),
                tensorboard_log=str(paths.logs_dir),
                seed=cfg["seed"],
                device=cfg["training"].get("device", "cuda"),
                verbose=1,
            )

        checkpoint_cb = CheckpointCallback(
            save_freq=cfg["training"].get("checkpoint_every", 10000),
            save_path=str(paths.checkpoints_dir),
            name_prefix="ppo_model",
        )
        callbacks: list[BaseCallback] = [checkpoint_cb]

        # Optional stop-on-request (e.g. from the GUI stop button).
        if stop_event is not None:
            callbacks.append(StopOnEvent(stop_event))

        # Caller-supplied callbacks (e.g. LiveTuningCallback from the GUI).
        if extra_callbacks:
            callbacks.extend(extra_callbacks)

        # Curriculum learning: bump env difficulty when performance exceeds threshold.
        curriculum_cfg = cfg.get("curriculum", {})
        if curriculum_cfg.get("enabled", False):
            callbacks.append(CurriculumCallback(curriculum_cfg, env_cfg, verbose=1))

        # Self-play league: periodically freeze policy snapshots as past opponents.
        self_play_cfg = cfg.get("self_play", {})
        if self_play_cfg.get("enabled", False) and env_cfg["type"] == "organism_arena_parallel":
            callbacks.append(SelfPlayCallback(
                snapshot_dir=paths.checkpoints_dir / "league",
                snapshot_freq=int(self_play_cfg.get("snapshot_freq", 5000)),
                max_league_size=int(self_play_cfg.get("max_league_size", 10)),
                sampling_mode=str(self_play_cfg.get("sampling_mode", "uniform")),
                recent_bias_alpha=float(self_play_cfg.get("recent_bias_alpha", 1.0)),
                seed=cfg["seed"],
                verbose=1,
            ))

        model.learn(
            total_timesteps=cfg["training"]["total_timesteps"],
            callback=callbacks,
            reset_num_timesteps=resume_from is None,
        )
        final_path = paths.checkpoints_dir / "final_model"
        model.save(str(final_path))
        if isinstance(vec_env, VecNormalize):
            vec_env.save(str(paths.checkpoints_dir / "vecnormalize.pkl"))
        return final_path
    finally:
        vec_env.close()
