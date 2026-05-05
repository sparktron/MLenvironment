from __future__ import annotations

from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

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
                termination.max_tilt_radians: 0.9
              2:
                reward.target_velocity: 1.5
                termination.max_tilt_radians: 0.7
              3:
                reward.target_velocity: 2.0
                termination.max_tilt_radians: 0.5

    env_cfg:
        A *mutable* reference to the environment config dict so that changes
        take effect on the next ``env.reset()``.
    verbose:
        Verbosity level (0 = silent, 1 = level-up messages).
    """

    def __init__(self, curriculum_cfg: dict[str, Any], env_cfg: dict[str, Any], verbose: int = 0):
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

        # SB3 logs ep_rew_mean after enough episodes have been collected.
        mean_reward = _safe_logger_value(self.logger, "rollout/ep_rew_mean")
        if mean_reward is None:
            return

        threshold = self._per_level_thresholds.get(self._level, self._default_threshold)
        if mean_reward >= threshold:
            self._level += 1
            self._cur_cfg["level"] = self._level
            self._apply_level_params(self._level)
            if self.verbose >= 1:
                print(f"[CurriculumCallback] Level up -> {self._level}  (mean_reward={mean_reward:.2f})")

    # ------------------------------------------------------------------
    def _apply_level_params(self, level: int) -> None:
        """Write the parameter overrides for *level* into the live env config."""
        overrides = self._level_params.get(level, {})
        for dotted_key, value in overrides.items():
            set_nested(self._env_cfg, dotted_key, value, strict=False)

    @property
    def current_level(self) -> int:
        return self._level


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _safe_logger_value(logger: Any, key: str) -> float | None:
    """Extract a value from the SB3 logger's name-to-value map, if present."""
    try:
        return float(logger.name_to_value[key])
    except (KeyError, AttributeError, TypeError):
        return None


