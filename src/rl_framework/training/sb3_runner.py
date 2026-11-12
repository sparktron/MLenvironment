from __future__ import annotations

import copy
import os
import threading
from pathlib import Path
from typing import Any

import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecCheckNan,
    VecNormalize,
)

from rl_framework.core.registry import get_task, make_env, registered_tasks
from rl_framework.training.curriculum_callback import CurriculumCallback
from rl_framework.training.evaluation_workers import resolve_evaluation_workers
from rl_framework.utils.checkpoint import (
    find_vecnormalize_path_for_model as _find_vecnormalize_path_for_model,
    validate_resume_path as _validate_resume_path,
    vecnormalize_path_for_model as _vecnormalize_path_for_model,
)
from rl_framework.utils.logging_utils import create_experiment_paths
from rl_framework.utils.run_registry import new_run_id, registry_for_config
from rl_framework.utils.reproducibility import (
    check_resume_provenance,
    configure_deterministic_mode,
    write_run_metadata,
)


def _resolve_registry_run_id(explicit_id: str | None = None) -> str:
    """Return the execution identity, independent of the output variant id."""
    return str(explicit_id) if explicit_id else new_run_id()


def _configure_torch_num_threads(training_cfg: dict[str, Any]) -> None:
    """Apply an explicit PyTorch CPU thread limit when configured."""
    value = training_cfg.get("torch_num_threads")
    if value is None:
        return
    import torch

    torch.set_num_threads(int(value))


def _make_subproc_vec_env(env_fns: list, training_cfg: dict[str, Any]):
    """Build a subprocess vec env with the configured process start method."""
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLAS_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        os.environ.setdefault(name, "1")
    start_method = training_cfg.get("worker_start_method")
    if start_method is None:
        return SubprocVecEnv(env_fns)
    return SubprocVecEnv(env_fns, start_method=start_method)


def _algorithm_class(name: str):
    algorithms = {"PPO": PPO, "SAC": SAC, "TD3": TD3}
    try:
        return algorithms[name.upper()]
    except KeyError as exc:
        raise ValueError(f"Unsupported training.algorithm: {name!r}") from exc


class VecNormalizeBestModelCallback(BaseCallback):
    """Save the training normalizer beside an EvalCallback best model."""

    def __init__(self, model_path: Path) -> None:
        super().__init__(verbose=0)
        self._model_path = model_path

    def _on_step(self) -> bool:
        vecnorm = self.model.get_vec_normalize_env()
        if vecnorm is not None:
            vecnorm.save(str(_vecnormalize_path_for_model(self._model_path)))
        return True


def _build_best_model_eval_env(
    env_cfg: dict[str, Any],
    normalize_observations: bool,
    workers: int,
    training_cfg: dict[str, Any],
):
    """Build a fresh walker eval env whose running stats are synced by SB3."""
    env_fns = [_build_single_env(env_cfg, rank=worker) for worker in range(workers)]
    eval_env: VecEnv = (
        DummyVecEnv(env_fns)
        if workers == 1
        else _make_subproc_vec_env(env_fns, training_cfg)
    )
    if normalize_observations:
        eval_env = VecNormalize(eval_env, training=False, norm_reward=False)
    return eval_env


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


class _AdvertisedRenderModeWrapper(gym.Wrapper):
    """Expose vector-compatible render metadata without enabling rendering.

    SB3 requires every worker in a ``VecEnv`` to report the same
    ``render_mode``. GUI capture only enables RGB rendering in worker 0, so
    headless workers use this wrapper to advertise the shared mode while their
    underlying environments retain ``render_mode=None`` and do no rendering.
    """

    render_mode: str = ""

    def __init__(self, env: gym.Env, render_mode: str) -> None:
        super().__init__(env)
        self.render_mode = render_mode


class VecNormalizeCheckpointCallback(CheckpointCallback):
    """Checkpoint callback that writes model-specific VecNormalize sidecars."""

    def __init__(self, save_freq: int, save_path: str, name_prefix: str) -> None:
        super().__init__(
            save_freq=save_freq,
            save_path=save_path,
            name_prefix=name_prefix,
        )

    def _on_step(self) -> bool:
        should_save = self.n_calls % self.save_freq == 0
        result = super()._on_step()
        if should_save:
            vecnorm = self.model.get_vec_normalize_env()
            if vecnorm is not None:
                model_path = Path(self._checkpoint_path(extension="zip"))
                vecnorm.save(str(_vecnormalize_path_for_model(model_path)))
        return result


def _build_single_env(
    env_cfg: dict[str, Any],
    monitor_dir: Path | None = None,
    rank: int = 0,
    render_env_index: int | None = None,
):
    """Return a factory that creates one gym env wrapped in SB3's ``Monitor``.

    The Monitor wrapper is what makes ``rollout/ep_rew_mean`` and
    ``rollout/ep_len_mean`` appear in the TensorBoard logger — SB3 pulls
    episode stats from each env's Monitor info dict via its ``ep_info_buffer``.
    Without it, the rollout/* tags are silently empty and the dashboard
    counters stay at ``--``.
    """

    def _make():
        rank_cfg = _worker_env_config(env_cfg, rank, render_env_index)
        env = make_env(rank_cfg["type"], rank_cfg)
        env = _advertise_selected_render_mode(env, env_cfg, rank, render_env_index)
        filename = (
            str(monitor_dir / f"monitor_env{rank}.csv")
            if monitor_dir is not None
            else None
        )
        return Monitor(env, filename=filename)

    return _make


def _worker_env_config(
    env_cfg: dict[str, Any], rank: int, render_env_index: int | None
) -> dict[str, Any]:
    """Return an isolated config for one rollout worker.

    When a caller selects one render worker (the GUI frame-capture path), all
    other workers have ``render_mode`` removed. A ``None`` selection preserves
    the existing behavior for non-GUI callers that deliberately configure a
    render mode for every environment.
    """
    rank_cfg = copy.deepcopy(env_cfg)
    if render_env_index is not None and rank != render_env_index:
        rank_cfg.pop("render_mode", None)
    return rank_cfg


def _advertise_selected_render_mode(
    env: gym.Env,
    env_cfg: dict[str, Any],
    rank: int,
    render_env_index: int | None,
) -> gym.Env:
    """Keep VecEnv render metadata uniform while non-selected workers stay headless."""
    configured_mode = env_cfg.get("render_mode")
    if (
        render_env_index is not None
        and rank != render_env_index
        and isinstance(configured_mode, str)
    ):
        return _AdvertisedRenderModeWrapper(env, configured_mode)
    return env


def _make_lr_schedule(start: float, end: float | None):
    """Return either a constant LR or a linear-decay callable accepting SB3's
    remaining-progress value (1.0 at start of training, 0.0 at end)."""
    if end is None:
        return float(start)
    start_f, end_f = float(start), float(end)

    def schedule(progress_remaining: float) -> float:
        # SB3 passes progress_remaining ∈ [1.0, 0.0]
        return end_f + (start_f - end_f) * float(progress_remaining)

    return schedule


def _callback_freq_from_timesteps(timesteps: int, num_envs: int) -> int:
    """Convert an env-timestep cadence to SB3 callback calls.

    SB3 calls callbacks once per vector-env step, where each call advances
    ``num_envs`` environment timesteps. Config values stay expressed in real
    environment timesteps because that is what operators see in logs/checkpoint
    names.
    """
    return max(int(timesteps) // max(int(num_envs), 1), 1)


def build_train_vec_env(cfg, paths, render_env_index, spec):
    if spec.build_train_vec_env is not None:
        return spec.build_train_vec_env(cfg, paths, render_env_index)
    env_cfg = cfg["environment"]
    train_cfg = cfg["training"]
    num_envs = int(train_cfg.get("num_envs", 1))
    env_fns = [
        _build_single_env(env_cfg, monitor_dir=paths.logs_dir, rank=i,
                          render_env_index=render_env_index)
        for i in range(max(num_envs, 1))
    ]
    return _make_subproc_vec_env(env_fns, train_cfg) if num_envs > 1 else DummyVecEnv(env_fns)


def build_model(cfg, train_cfg, algorithm_name, algorithm_cls, vec_env, paths, resume_from):
    if resume_from is not None:
        model = algorithm_cls.load(
            str(resume_from),
            env=vec_env,
            tensorboard_log=str(paths.logs_dir),
            device=cfg["training"].get("device", "auto"),
        )
    else:
        lr = _make_lr_schedule(
            train_cfg.get("learning_rate", 3e-4),
            train_cfg.get("learning_rate_end"),  # None → constant LR
        )
        common_kwargs = {
            "policy": train_cfg.get("policy", "MlpPolicy"),
            "env": vec_env,
            "learning_rate": lr,
            "batch_size": train_cfg.get("batch_size", 256),
            "gamma": train_cfg.get("gamma", 0.99),
            "tensorboard_log": str(paths.logs_dir),
            "seed": cfg["seed"],
            "device": train_cfg.get("device", "auto"),
            "verbose": 1,
        }
        if algorithm_name == "PPO" and "log_std_init" in train_cfg:
            common_kwargs["policy_kwargs"] = {
                "log_std_init": float(train_cfg["log_std_init"])
            }
        if algorithm_name == "PPO":
            model = PPO(
                **common_kwargs,
                n_steps=train_cfg.get("n_steps", 1024),
                n_epochs=train_cfg.get("n_epochs", 10),
                gae_lambda=train_cfg.get("gae_lambda", 0.95),
                clip_range=train_cfg.get("clip_range", 0.2),
                ent_coef=train_cfg.get("ent_coef", 0.005),
                vf_coef=train_cfg.get("vf_coef", 0.5),
                max_grad_norm=train_cfg.get("max_grad_norm", 0.5),
            )
        else:
            off_policy_kwargs = {
                **common_kwargs,
                "buffer_size": train_cfg.get("buffer_size", 1_000_000),
                "learning_starts": train_cfg.get("learning_starts", 1_000),
                "train_freq": train_cfg.get("train_freq", 1),
                "gradient_steps": train_cfg.get("gradient_steps", 1),
                "tau": train_cfg.get("tau", 0.005),
            }
            if algorithm_name == "SAC":
                model = SAC(
                    **off_policy_kwargs,
                    ent_coef=train_cfg.get("ent_coef", "auto"),
                )
            else:
                model = TD3(
                    **off_policy_kwargs,
                    policy_delay=train_cfg.get("policy_delay", 2),
                    target_policy_noise=train_cfg.get("target_policy_noise", 0.2),
                    target_noise_clip=train_cfg.get("target_noise_clip", 0.5),
                )
    return model


def build_callbacks(cfg, paths, spec, num_envs, algorithm_name, normalize,
                    render_env_index, stop_event, extra_callbacks):
    train_cfg = cfg["training"]
    env_cfg = cfg["environment"]
    best_eval_env = None
    checkpoint_cb = VecNormalizeCheckpointCallback(
        save_freq=_callback_freq_from_timesteps(
            cfg["training"].get("checkpoint_every", 10000),
            num_envs,
        ),
        save_path=str(paths.checkpoints_dir),
        name_prefix=f"{algorithm_name.lower()}_model",
    )
    callbacks: list[BaseCallback] = [checkpoint_cb]

    if train_cfg.get("velocity_target_ramp") is not None and "velocity_target_ramp" not in spec.capabilities:
        supported = next(name for name in registered_tasks() if "velocity_target_ramp" in get_task(name).capabilities)
        raise ValueError(f"training.velocity_target_ramp requires {supported}")
    if spec.train_callbacks:
        callbacks.extend(spec.train_callbacks(cfg, paths, "after_checkpoint"))

    best_model_cfg = cfg.get("evaluation", {}).get("best_model", {})
    if best_model_cfg.get("enabled", False):
        best_model_path = paths.checkpoints_dir / "best_model"
        eval_env_cfg = _worker_env_config(env_cfg, -1, render_env_index)
        best_eval_episodes = int(best_model_cfg.get("episodes", 5))
        best_eval_workers = resolve_evaluation_workers(
            {
                **cfg,
                "evaluation": {
                    **cfg.get("evaluation", {}),
                    "workers": best_model_cfg.get(
                        "workers",
                        cfg.get("evaluation", {}).get("workers"),
                    ),
                },
            },
            best_eval_episodes,
        )
        best_eval_env = _build_best_model_eval_env(
            eval_env_cfg,
            normalize,
            best_eval_workers,
            cfg.get("training", {}),
        )
        callbacks.append(
            EvalCallback(
                best_eval_env,
                callback_on_new_best=VecNormalizeBestModelCallback(best_model_path),
                n_eval_episodes=best_eval_episodes,
                eval_freq=_callback_freq_from_timesteps(
                    int(best_model_cfg.get("eval_every", 50_000)), num_envs
                ),
                best_model_save_path=str(paths.checkpoints_dir),
                log_path=str(paths.logs_dir),
                deterministic=True,
                warn=False,
            )
        )

    # Optional stop-on-request (e.g. from the GUI stop button).
    if stop_event is not None:
        callbacks.append(StopOnEvent(stop_event))

    # Caller-supplied callbacks (e.g. LiveTuningCallback from the GUI).
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    if spec.train_callbacks:
        callbacks.extend(spec.train_callbacks(cfg, paths, "before_curriculum"))

    # Curriculum learning: bump env difficulty when performance exceeds threshold.
    curriculum_cfg = cfg.get("curriculum", {})
    if curriculum_cfg.get("enabled", False):
        callbacks.append(CurriculumCallback(curriculum_cfg, env_cfg, verbose=1))

    if spec.train_callbacks:
        callbacks.extend(spec.train_callbacks(cfg, paths, "after_curriculum"))
    return callbacks, best_eval_env


def train(
    cfg: dict[str, Any],
    extra_callbacks: list[BaseCallback] | None = None,
    stop_event: threading.Event | None = None,
    resume_from: str | Path | None = None,
    render_env_index: int | None = None,
    registry_run_id: str | None = None,
) -> Path:
    """Train an SB3 agent from *cfg* and return the path to the saved model.

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
    render_env_index:
        Optional rollout-worker index that alone retains the configured
        ``environment.render_mode``. The GUI uses index 0 for live frame
        capture without enabling RGB rendering on every parallel worker.
    registry_run_id:
        Opaque execution identity supplied by an orchestrator that has already
        registered the run. Direct calls generate a fresh identity.
    """
    train_cfg = cfg["training"]
    num_envs = int(train_cfg.get("num_envs", 1))
    if render_env_index is not None and not 0 <= render_env_index < max(num_envs, 1):
        raise ValueError(
            f"render_env_index must be in [0, {max(num_envs, 1) - 1}], "
            f"got {render_env_index}"
        )
    repro_cfg = cfg.get("reproducibility", {})
    if repro_cfg.get("deterministic", False):
        train_cfg.setdefault("torch_num_threads", 1)
        train_cfg.setdefault("worker_start_method", "spawn")
        configure_deterministic_mode(int(cfg["seed"]))
    _configure_torch_num_threads(train_cfg)
    if resume_from is not None:
        _validate_resume_path(
            Path(resume_from),
            cfg["training"].get("normalize_observations", True),
        )
        check_resume_provenance(
            resume_from, cfg, strict=bool(repro_cfg.get("strict", False))
        )

    paths = create_experiment_paths(
        cfg["output"]["base_dir"],
        cfg["experiment_name"],
        cfg["seed"],
        run_id=cfg["output"].get("run_id"),
    )
    write_run_metadata(
        paths.run_dir,
        cfg,
        strict=bool(repro_cfg.get("strict", False)),
        resume_from=resume_from,
    )
    registry = registry_for_config(cfg)
    run_identity = _resolve_registry_run_id(registry_run_id)
    registry.register_run(
        run_identity,
        cfg,
        paths.run_dir,
        resume_from=resume_from,
        allow_existing=registry_run_id is not None,
    )
    env_cfg = cfg["environment"]
    algorithm_name = str(train_cfg.get("algorithm", "PPO")).upper()
    algorithm_cls = _algorithm_class(algorithm_name)

    spec = get_task(env_cfg["type"])
    league_dir = paths.checkpoints_dir / "league"
    vec_env = build_train_vec_env(cfg, paths, render_env_index, spec)

    try:
        normalize = cfg["training"].get("normalize_observations", True)
        vecnormalize_env = None
        best_eval_env = None
        if normalize:
            vecnorm_path = None
            if resume_from is not None:
                vecnorm_path = _find_vecnormalize_path_for_model(resume_from)
            if vecnorm_path is not None:
                vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
                # Keep updating stats and continue training.
                vec_env.training = True
                vec_env.norm_reward = False
            else:
                vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
            vecnormalize_env = vec_env

        if cfg["training"].get("check_nans", False):
            vec_env = VecCheckNan(vec_env, raise_exception=True)

        model = build_model(
            cfg, train_cfg, algorithm_name, algorithm_cls, vec_env, paths, resume_from
        )

        callbacks, best_eval_env = build_callbacks(
            cfg, paths, spec, num_envs, algorithm_name, normalize,
            render_env_index, stop_event, extra_callbacks,
        )

        model.learn(
            total_timesteps=cfg["training"]["total_timesteps"],
            callback=callbacks,
            reset_num_timesteps=resume_from is None,
        )
        final_path = paths.checkpoints_dir / "final_model"
        model.save(str(final_path))
        if vecnormalize_env is not None:
            vecnormalize_env.save(str(_vecnormalize_path_for_model(final_path)))
            vecnormalize_env.save(str(paths.checkpoints_dir / "vecnormalize.pkl"))
        registry.update_run(run_identity, status="completed", model_path=final_path)
        registry.record_artifact(
            run_identity, "final_model", final_path.with_suffix(".zip")
        )
        registry.record_artifact(
            run_identity, "run_metadata", paths.run_dir / "run_metadata.json"
        )
        registry.record_artifact(
            run_identity, "vecnormalize", paths.checkpoints_dir / "vecnormalize.pkl"
        )
        for artifact in paths.checkpoints_dir.iterdir():
            if artifact.is_file():
                registry.record_artifact(run_identity, "checkpoint", artifact)
        return final_path
    except Exception as exc:
        registry.update_run(run_identity, status="failed", error=str(exc))
        registry.record_event(run_identity, "run_failed", {"error": str(exc)})
        raise
    finally:
        if "best_eval_env" in locals() and best_eval_env is not None:
            best_eval_env.close()
        if "vec_env" in locals():
            vec_env.close()
