from __future__ import annotations

import copy
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecCheckNan,
    VecNormalize,
)
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper

from rl_framework.envs.registry import make_env
from rl_framework.training.curriculum_callback import CurriculumCallback
from rl_framework.training.reward_annealing_callback import RewardAnnealingCallback
from rl_framework.training.self_play_callback import SelfPlayCallback
from rl_framework.training.self_play_env_wrapper import (
    LeagueSampler,
    SelfPlayEnvWrapper,
    SingleAgentArenaEnv,
)
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


def _configure_torch_num_threads(training_cfg: dict[str, Any]) -> None:
    """Apply an explicit PyTorch CPU thread limit when configured."""
    value = training_cfg.get("torch_num_threads")
    if value is None:
        return
    import torch

    torch.set_num_threads(int(value))


def _make_subproc_vec_env(env_fns: list, training_cfg: dict[str, Any]):
    """Build a subprocess vec env with the configured process start method."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLAS_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    start_method = training_cfg.get("worker_start_method")
    if start_method is None:
        return SubprocVecEnv(env_fns)
    return SubprocVecEnv(env_fns, start_method=start_method)


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
    env_cfg: dict[str, Any], normalize_observations: bool
):
    """Build a fresh walker eval env whose running stats are synced by SB3."""
    eval_env = DummyVecEnv([_build_single_env(env_cfg)])
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


class _ArenaVecEnvAdapter(VecEnvWrapper):
    """Adapt SuperSuit's SB3 arena vec env to SB3's expectations.

    Two SuperSuit 3.10 ↔ Stable-Baselines3 incompatibilities are patched here:

    1. **Missing ``seed()``.** SuperSuit's ``ConcatVecEnv`` only supports seeding
       through ``reset(seed=)`` and exposes no ``seed()`` method, yet SB3's
       ``set_random_seed`` calls ``env.seed(seed)`` during ``PPO`` construction —
       otherwise raising ``AttributeError`` before training can start. Routing
       through SuperSuit's ``reset(seed=)`` is not viable: its
       ``SB3VecEnvWrapper.reset`` re-enters the same broken ``seed()`` and
       recurses. The arena env's RNG is already seeded from ``cfg["seed"]`` at
       construction (and preserved across ``reset(seed=None)``), so ``seed()``
       only needs to seed the action/observation spaces for reproducible
       sampling.

    2. **``uint8`` dones.** SuperSuit returns the ``dones`` array as ``uint8``.
       ``VecNormalize.step_wait`` does ``self.returns[dones] = 0``, which numpy
       interprets as *integer fancy-indexing* rather than a boolean mask — at
       ``num_envs == 1`` (single-agent self-play) ``returns[[1]]`` is out of
       bounds and crashes; at higher counts it silently resets the wrong slots.
       Casting ``dones`` to ``bool`` restores mask semantics.
    """

    def reset(self):  # type: ignore[override]
        return self.venv.reset()

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return obs, rewards, np.asarray(dones, dtype=bool), infos

    def seed(self, seed: int | None = None):
        if seed is not None:
            self.action_space.seed(seed)
            self.observation_space.seed(seed)
        return [seed] * self.num_envs

    def _arena_envs(self) -> list:
        """Reach the live arena ``ParallelEnv`` instances inside the chain.

        SuperSuit's ``ConcatVecEnv`` is a gymnasium vector env and does not
        implement SB3's ``env_method``, so curriculum / reward-annealing updates
        cannot propagate natively. We walk the structure
        (``SB3VecEnvWrapper.venv`` → ``ConcatVecEnv.vec_envs`` →
        ``MarkovVectorEnv.par_env`` → ``.unwrapped``) to the underlying arena
        envs, unwrapping any SelfPlayEnvWrapper.
        """
        envs: list = []
        concat = getattr(self.venv, "venv", None)  # SB3VecEnvWrapper -> ConcatVecEnv
        for markov in getattr(concat, "vec_envs", []):
            par_env = getattr(markov, "par_env", None)
            if par_env is not None:
                envs.append(par_env.unwrapped)
        return envs

    def env_method(self, method_name, *args, indices=None, **kwargs):
        """Call a method on each underlying arena env (SB3 ``env_method`` shim)."""
        return [
            getattr(env, method_name)(*args, **kwargs) for env in self._arena_envs()
        ]


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


class ArenaMetricsCallback(BaseCallback):
    """Log per-episode arena outcomes (win rates, timeouts) to TensorBoard.

    The arena env is wrapped through SuperSuit rather than SB3's ``Monitor``,
    so ``rollout/ep_rew_mean`` is unavailable. This callback instead reads the
    ``episode_outcome`` annotation that the arena attaches to terminal/truncated
    steps (or that ``SelfPlayEnvWrapper`` attaches when a live N-agent learner is
    eliminated) and records aggregate win rates at the end of each rollout.

    Note: SuperSuit's vec-env conversion surfaces one ``info`` per agent slot,
    so a finished episode contributes its outcome once per *active* agent
    (2 for a duel, N for an N-agent free-for-all — spectators stay active
    until the episode ends). Win rates are computed as a fraction of *outcome
    observations*, which keeps the ratios correct regardless of N since both
    the numerator and denominator scale by the same active-agent count.

    Win totals are keyed by whatever agent names are actually observed as
    winners, so this scales to N-agent free-for-alls (``environment.
    num_agents`` up to 8 in the GUI schema) rather than only the 2-agent
    default; ``agent_0``/``agent_1`` are seeded up front so their rate is
    always logged (as 0.0) even in rollouts where they didn't win, matching
    prior behavior for the common duel case.
    """

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self._wins: dict[str, int] = {"agent_0": 0, "agent_1": 0}
        self._timeouts = 0
        self._draws = 0
        self._eliminations = 0
        self._outcomes = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            outcome = info.get("episode_outcome")
            if not outcome:
                continue
            self._outcomes += 1
            kind = outcome.get("outcome")
            if kind == "timeout":
                self._timeouts += 1
            elif kind == "draw":
                self._draws += 1
            elif kind == "eliminated":
                self._eliminations += 1
            else:
                winner = outcome.get("winner")
                if winner is not None:
                    self._wins[winner] = self._wins.get(winner, 0) + 1
        return True

    def _on_rollout_end(self) -> None:
        if self._outcomes == 0:
            return
        total = self._outcomes
        for agent, count in self._wins.items():
            self.logger.record(f"arena/{agent}_win_rate", count / total)
        self.logger.record("arena/timeout_rate", self._timeouts / total)
        self.logger.record("arena/draw_rate", self._draws / total)
        self.logger.record("arena/elimination_rate", self._eliminations / total)
        self.logger.record("arena/episode_outcomes", total)
        # Keep any agent keys discovered this run (so their metric keeps
        # being logged, at 0.0, in rollouts where they don't win).
        self._wins = dict.fromkeys(self._wins, 0)
        self._timeouts = 0
        self._draws = 0
        self._eliminations = 0
        self._outcomes = 0


def _build_single_env(
    env_cfg: dict[str, Any], monitor_dir: Path | None = None, rank: int = 0
):
    """Return a factory that creates one gym env wrapped in SB3's ``Monitor``.

    The Monitor wrapper is what makes ``rollout/ep_rew_mean`` and
    ``rollout/ep_len_mean`` appear in the TensorBoard logger — SB3 pulls
    episode stats from each env's Monitor info dict via its ``ep_info_buffer``.
    Without it, the rollout/* tags are silently empty and the dashboard
    counters stay at ``--``.
    """

    def _make():
        env = make_env(env_cfg["type"], env_cfg)
        filename = (
            str(monitor_dir / f"monitor_env{rank}.csv")
            if monitor_dir is not None
            else None
        )
        return Monitor(env, filename=filename)

    return _make


def _build_arena_selfplay_env(
    env_cfg: dict[str, Any],
    self_play_cfg: dict[str, Any],
    league_dir: Path,
    base_seed: int,
    monitor_dir: Path | None = None,
    rank: int = 0,
):
    """Factory for one self-play arena env on SB3's native vec-env path.

    The arena ``ParallelEnv`` is wrapped in a :class:`SelfPlayEnvWrapper`
    (frozen opponent in the second slot) and then a :class:`SingleAgentArenaEnv`
    Gymnasium adapter, so it behaves like any single-agent env under
    ``DummyVecEnv``/``SubprocVecEnv`` — no SuperSuit, working ``env_method``,
    and a ``Monitor`` for ``rollout/ep_rew_mean``.

    Each rank is seeded independently (env construction + league sampler) so
    parallel workers explore distinct spawns and sample distinct opponent
    sequences rather than running identical episodes.
    """

    def _make():
        rank_cfg = copy.deepcopy(env_cfg)
        # Spread per-worker seeds so spawn jitter and the RNG differ by rank.
        rank_cfg["seed"] = base_seed + 1000 * rank
        par_env = make_env(rank_cfg["type"], rank_cfg)
        sampler = LeagueSampler(
            league_dir,
            sampling_mode=str(self_play_cfg.get("sampling_mode", "uniform")),
            recent_bias_alpha=float(self_play_cfg.get("recent_bias_alpha", 1.0)),
            seed=base_seed + 1000 * rank,
        )
        env = SingleAgentArenaEnv(SelfPlayEnvWrapper(par_env, sampler))
        filename = (
            str(monitor_dir / f"monitor_env{rank}.csv")
            if monitor_dir is not None
            else None
        )
        # info_keywords persists the per-episode arena outcome into the Monitor
        # CSV; the live agent's info carries it on terminal/timeout steps.
        return Monitor(env, filename=filename, info_keywords=("episode_outcome",))

    return _make


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
    train_cfg = cfg["training"]
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
    run_identity = str(cfg["output"].get("run_id") or new_run_id())
    registry.register_run(run_identity, cfg, paths.run_dir, resume_from=resume_from)
    env_cfg = cfg["environment"]

    num_envs = int(cfg["training"].get("num_envs", 1))

    # Self-play league config — used both to wrap the env (opponent routing) and
    # to register the snapshot-saving callback later. The snapshot directory is
    # the shared channel between them (survives cloudpickle cloning into workers).
    self_play_cfg = cfg.get("self_play", {})
    self_play_enabled = bool(self_play_cfg.get("enabled", False)) and (
        env_cfg["type"] == "organism_arena_parallel"
    )
    league_dir = paths.checkpoints_dir / "league"

    # The *shared-policy* arena path (no self-play) must stay single-process.
    # It needs SuperSuit's multi-agent vec conversion, and SuperSuit's
    # concat_vec_envs_v1(num_cpus=num_envs) forks the envs at num_envs > 1, which
    #   * empties the in-process chain _ArenaVecEnvAdapter.env_method() walks, so
    #     reward annealing / curriculum updates become silent no-ops, and
    #   * is unstable in SuperSuit 3.10 at num_cpus >= 2.
    # The self-play path does NOT use SuperSuit (single-agent view + native SB3
    # vec env), so it parallelizes safely — only guard the shared-policy path.
    if (
        env_cfg["type"] == "organism_arena_parallel"
        and num_envs > 1
        and not self_play_enabled
    ):
        raise ValueError(
            "Shared-policy organism_arena_parallel training (self_play.enabled: "
            f"false) requires training.num_envs == 1 (got {num_envs}). It runs "
            "single-process via SuperSuit; num_envs > 1 forks into subprocesses, "
            "silently disabling live env_method updates (reward annealing, "
            "curriculum) and is unstable in SuperSuit 3.10. Either set "
            "training.num_envs: 1, or enable self_play to use the parallel path."
        )

    if env_cfg["type"] == "organism_arena_parallel" and self_play_enabled:
        # Self-play exposes a single live agent, so skip SuperSuit entirely and
        # use SB3's native vec-env path (parallel-safe, working env_method,
        # Monitor metrics). The opponent is sampled per-episode from the on-disk
        # league inside each worker.
        env_fns = [
            _build_arena_selfplay_env(
                env_cfg,
                self_play_cfg,
                league_dir,
                base_seed=int(cfg["seed"]),
                monitor_dir=paths.logs_dir,
                rank=i,
            )
            for i in range(max(num_envs, 1))
        ]
        if num_envs > 1:
            vec_env = _make_subproc_vec_env(env_fns, train_cfg)
        else:
            vec_env = DummyVecEnv(env_fns)
    elif env_cfg["type"] == "organism_arena_parallel":
        # Shared-policy arena: SuperSuit multi-agent conversion, single process.
        par_env = make_env(env_cfg["type"], env_cfg)
        vec_env = ss.pettingzoo_env_to_vec_env_v1(par_env)
        vec_env = ss.concat_vec_envs_v1(
            vec_env,
            1,
            num_cpus=1,
            base_class="stable_baselines3",
        )
        # Patch SuperSuit↔SB3 incompatibilities (missing seed(), uint8 dones).
        vec_env = _ArenaVecEnvAdapter(vec_env)
    else:
        env_fns = [
            _build_single_env(env_cfg, monitor_dir=paths.logs_dir, rank=i)
            for i in range(max(num_envs, 1))
        ]
        if num_envs > 1:
            vec_env = _make_subproc_vec_env(env_fns, train_cfg)
        else:
            vec_env = DummyVecEnv(env_fns)

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

        if resume_from is not None:
            model = PPO.load(
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
            model = PPO(
                policy=train_cfg.get("policy", "MlpPolicy"),
                env=vec_env,
                learning_rate=lr,
                n_steps=train_cfg.get("n_steps", 1024),
                batch_size=train_cfg.get("batch_size", 256),
                n_epochs=train_cfg.get("n_epochs", 10),
                gamma=train_cfg.get("gamma", 0.99),
                gae_lambda=train_cfg.get("gae_lambda", 0.95),
                clip_range=train_cfg.get("clip_range", 0.2),
                ent_coef=train_cfg.get("ent_coef", 0.005),
                vf_coef=train_cfg.get("vf_coef", 0.5),
                max_grad_norm=train_cfg.get("max_grad_norm", 0.5),
                tensorboard_log=str(paths.logs_dir),
                seed=cfg["seed"],
                device=train_cfg.get("device", "auto"),
                verbose=1,
            )

        checkpoint_cb = VecNormalizeCheckpointCallback(
            save_freq=_callback_freq_from_timesteps(
                cfg["training"].get("checkpoint_every", 10000),
                num_envs,
            ),
            save_path=str(paths.checkpoints_dir),
            name_prefix="ppo_model",
        )
        callbacks: list[BaseCallback] = [checkpoint_cb]

        best_model_cfg = cfg.get("evaluation", {}).get("best_model", {})
        if best_model_cfg.get("enabled", False):
            best_model_path = paths.checkpoints_dir / "best_model"
            best_eval_env = _build_best_model_eval_env(env_cfg, normalize)
            callbacks.append(
                EvalCallback(
                    best_eval_env,
                    callback_on_new_best=VecNormalizeBestModelCallback(best_model_path),
                    n_eval_episodes=int(best_model_cfg.get("episodes", 5)),
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

        # Arena training has no Monitor wrapper; surface win/loss metrics instead.
        # (Recorded before CurriculumCallback so a win-rate gate can read them.)
        if env_cfg["type"] == "organism_arena_parallel":
            callbacks.append(ArenaMetricsCallback())

        # Dense-to-sparse reward annealing for the arena: ramp the per-hit reward
        # down so the terminal win/loss signal eventually dominates.
        anneal_cfg = cfg.get("reward_annealing", {})
        if (
            anneal_cfg.get("enabled", False)
            and env_cfg["type"] == "organism_arena_parallel"
        ):
            callbacks.append(
                RewardAnnealingCallback(
                    anneal_steps=int(anneal_cfg.get("anneal_steps", 500_000)),
                    verbose=1,
                )
            )

        # Curriculum learning: bump env difficulty when performance exceeds threshold.
        curriculum_cfg = cfg.get("curriculum", {})
        if curriculum_cfg.get("enabled", False):
            callbacks.append(CurriculumCallback(curriculum_cfg, env_cfg, verbose=1))

        # Self-play league: periodically freeze policy snapshots as past
        # opponents. Writes to the same league_dir the env's LeagueSampler reads.
        if self_play_enabled:
            callbacks.append(
                SelfPlayCallback(
                    snapshot_dir=league_dir,
                    snapshot_freq=int(self_play_cfg.get("snapshot_freq", 5000)),
                    max_league_size=int(self_play_cfg.get("max_league_size", 10)),
                    sampling_mode=str(self_play_cfg.get("sampling_mode", "uniform")),
                    recent_bias_alpha=float(
                        self_play_cfg.get("recent_bias_alpha", 1.0)
                    ),
                    seed=cfg["seed"],
                    verbose=1,
                )
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
        registry.record_artifact(run_identity, "final_model", final_path.with_suffix(".zip"))
        registry.record_artifact(run_identity, "run_metadata", paths.run_dir / "run_metadata.json")
        registry.record_artifact(run_identity, "vecnormalize", paths.checkpoints_dir / "vecnormalize.pkl")
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
