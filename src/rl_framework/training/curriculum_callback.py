from __future__ import annotations

from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from rl_framework.training.rollout_metrics import rollout_metric
from rl_framework.utils.config_merge import set_nested


class CurriculumCallback(BaseCallback):
    """SB3 callback that adjusts environment difficulty based on training progress.

    At the end of each rollout, reads the mean episode reward from the logger.
    When the metric exceeds *level_up_threshold* for the current level, bumps
    ``curriculum.level`` and applies the matching parameter overrides defined
    in the *level_params* mapping.

    Parameters
    ----------
    curriculum_cfg:
        The ``curriculum`` section of the experiment YAML, expected to contain::

            enabled: true
            level_up_threshold: 150.0   # default threshold applied to all levels
            # Optional per-level thresholds (take priority over the default):
            level_up_thresholds:
              0: 100.0   # threshold to leave level 0
              1: 150.0
              2: 200.0
            max_level: 3
            level_params:
              1:
                reward.target_velocity: 1.0
                termination.min_height: 0.35
              2:
                reward.target_velocity: 1.5
                termination.min_height: 0.30
              3:
                reward.target_velocity: 2.0
                termination.min_height: 0.25

    env_cfg:
        A *mutable* reference to the environment config dict so that changes
        are reflected on subsequent ``env.reset()`` calls.  Live env objects
        are also updated immediately via ``env_method("update_live_params")``.
    verbose:
        Verbosity level (0 = silent, 1 = level-up messages).
    """

    def __init__(
        self, curriculum_cfg: dict[str, Any], env_cfg: dict[str, Any], verbose: int = 0
    ):
        super().__init__(verbose)
        self._cur_cfg = curriculum_cfg
        self._env_cfg = env_cfg
        self._level = int(curriculum_cfg.get("level", 0))
        self._default_threshold = float(curriculum_cfg.get("level_up_threshold", 150.0))
        self._per_level_thresholds: dict[int, float] = {
            int(k): float(v)
            for k, v in curriculum_cfg.get("level_up_thresholds", {}).items()
        }
        self._max_level = int(curriculum_cfg.get("max_level", 3))
        # Metric driving level-ups. Defaults to the walker's episode-reward mean;
        # the arena gates on a win-rate metric logged by ArenaMetricsCallback
        # (e.g. "arena/agent_0_win_rate"), which has no Monitor ep_rew_mean.
        self._metric = str(curriculum_cfg.get("metric", "rollout/ep_rew_mean"))
        # Suppress level-ups until this many timesteps have elapsed. For the
        # arena self-play setup the league is empty (opponent = random actions)
        # until the first snapshot lands, so an early win rate reflects beating
        # noise, not skill. Gating on warmup_steps >= snapshot_freq stops the
        # curriculum from ramping difficulty against a random opponent.
        self._warmup_steps = int(curriculum_cfg.get("warmup_steps", 0))
        self._level_params: dict[int, dict[str, Any]] = {
            int(k): v for k, v in curriculum_cfg.get("level_params", {}).items()
        }

    # ------------------------------------------------------------------
    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        """Check after every rollout whether performance warrants a level bump."""
        if self._level >= self._max_level:
            return

        # Hold off until the warmup window has elapsed (e.g. until the self-play
        # league has real opponents rather than a random fallback).
        if self.num_timesteps < self._warmup_steps:
            return

        # Resolve the configured metric (ep_rew_mean for the walker via the
        # model's ep_info_buffer, arena/agent_0_win_rate via the logger — see
        # rollout_metrics for why the logger cannot serve rollout/* keys here).
        # None while no data exists yet.
        metric_value = rollout_metric(self.model, self.logger, self._metric)
        if metric_value is None:
            return

        threshold = self._per_level_thresholds.get(self._level, self._default_threshold)
        if metric_value >= threshold:
            self._level += 1
            self._cur_cfg["level"] = self._level
            self._apply_level_params(self._level)
            if self.verbose >= 1:
                print(
                    f"[CurriculumCallback] Level up -> {self._level}  "
                    f"({self._metric}={metric_value:.3f})"
                )

    # ------------------------------------------------------------------
    def _apply_level_params(self, level: int) -> None:
        """Write the parameter overrides for *level* into the live env config and
        propagate them directly to each env's reward/termination objects."""
        overrides = self._level_params.get(level, {})
        if not overrides:
            return
        for dotted_key, value in overrides.items():
            set_nested(self._env_cfg, dotted_key, value, strict=False)
        # Push changes to live env instances so they take effect immediately,
        # not just on the next reset.
        if self.training_env is not None:
            try:
                self.training_env.env_method("update_live_params", overrides)
            except Exception:
                pass  # env_method unavailable (e.g. non-walker env); cfg update suffices

    @property
    def current_level(self) -> int:
        return self._level
