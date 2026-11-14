from __future__ import annotations

import logging
from typing import Any
from typing import Callable

from stable_baselines3.common.callbacks import BaseCallback

from rl_framework.core.training.rollout_metrics import rollout_metric
from rl_framework.core.config.config_merge import get_section
from rl_framework.core.registry import get_task

_log = logging.getLogger(__name__)


class LiveTuningCallback(BaseCallback):
    """SB3 callback that applies real-time parameter changes from an event source.

    The GUI pushes tuning requests into an in-memory queue managed by
    :class:`TrainingManager`. This callback consumes one merged event payload at
    the end of each rollout and publishes status snapshots back to the manager.

    Supported live-tunable parameters
    ---------------------------------
    - ``learning_rate``          (float)  – PPO optimizer LR
    - ``reward.*``               (float)  – any key under env_cfg["reward"]
    - ``termination.*``          (float)  – any key under env_cfg["termination"]
    - ``domain_randomization.*`` (float)  – any key under env_cfg["domain_randomization"]
    - ``battle_rules.*``         (float)  – any key under env_cfg["battle_rules"]

    Accepted env-section changes are propagated through
    ``env_method("update_live_params")``. Each env applies the subset it can
    safely change mid-run and stores the rest in its config for future resets.

    """

    ENV_SECTIONS = (
        "reward",
        "termination",
        "domain_randomization",
        "battle_rules",
        "morphology",
        "resources",
        "sim",
    )

    def __init__(
        self,
        env_cfg: dict[str, Any],
        pop_tuning_event: Callable[[], dict[str, Any] | None] | None = None,
        publish_status: Callable[[dict[str, Any]], None] | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._env_cfg = env_cfg
        self._pop_tuning_event = pop_tuning_event
        self._publish_status = publish_status
        self._applied: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        params = (
            self._pop_tuning_event() if self._pop_tuning_event is not None else None
        )
        if isinstance(params, dict) and params:
            self._apply_params(params)
        # Publish a status snapshot every rollout — the GUI dashboard polls
        # these metrics, so they must stream even when no tuning is pending.
        self._write_status()

    def _apply_params(self, params: dict[str, Any]) -> None:
        applied = {}
        env_params = {}
        for key, value in params.items():
            if key == "learning_rate":
                lr = float(value)
                self.model.lr_schedule = lambda _progress: lr
                applied[key] = lr
            else:
                parts = key.split(".", 1)
                if len(parts) != 2:
                    raise ValueError(f"Unknown live parameter {key!r}")
                task_type = self._env_cfg.get("type")
                if task_type and key not in get_task(task_type).live_params:
                    valid = sorted(get_task(task_type).live_params)
                    raise ValueError(f"Unknown live parameter {key!r}; valid keys: {valid}")
                section, param = parts
                section_cfg = get_section(self._env_cfg, section)
                current = section_cfg.get(param)
                env_params[key] = type(current)(value) if current is not None else value

        if env_params and self.training_env is not None and callable(getattr(self.training_env, "env_method", None)):
            self.training_env.env_method("update_live_params", env_params)
        for key, value in env_params.items():
            section, param = key.split(".", 1)
            get_section(self._env_cfg, section)[param] = value
            applied[key] = value

        if applied:
            self._applied.append(applied)

    def _write_status(self) -> None:
        if self._publish_status is None:
            return
        try:
            status: dict[str, Any] = {
                "timesteps": self.num_timesteps,
                "applied_count": len(self._applied),
            }
            # Episode stats come from the model's ep_info_buffer, train/* from
            # the logger; absent metrics are omitted rather than reported as
            # 0.0 (see rollout_metrics for the SB3 logger-clearing pitfall).
            for key in (
                "rollout/ep_rew_mean",
                "rollout/ep_len_mean",
                "train/loss",
                "train/policy_gradient_loss",
                "train/value_loss",
                "train/entropy_loss",
                "train/learning_rate",
            ):
                value = rollout_metric(self.model, self.logger, key)
                if value is not None:
                    status[key] = value
            self._publish_status(status)
        except Exception:
            pass
