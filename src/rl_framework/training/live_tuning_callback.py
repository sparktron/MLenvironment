from __future__ import annotations

import logging
from typing import Any
from typing import Callable

from stable_baselines3.common.callbacks import BaseCallback

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

    """

    ENV_SECTIONS = ("reward", "termination", "domain_randomization", "battle_rules", "morphology", "sim")

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
        if self._pop_tuning_event is None:
            return
        params = self._pop_tuning_event()
        if not isinstance(params, dict) or not params:
            return

        applied = {}
        for key, value in params.items():
            if key == "learning_rate":
                try:
                    lr = float(value)
                    self.model.lr_schedule = lambda _progress: lr
                    applied[key] = lr
                    if self.verbose >= 1:
                        print(f"[LiveTuning] learning_rate -> {lr}")
                except (TypeError, ValueError) as exc:
                    _log.warning("LiveTuning: could not apply 'learning_rate'=%r: %s", value, exc)
            else:
                parts = key.split(".", 1)
                if len(parts) == 2 and parts[0] in self.ENV_SECTIONS:
                    section, param = parts
                    if section in self._env_cfg and param in self._env_cfg[section]:
                        try:
                            cast_val = type(self._env_cfg[section][param])(value)
                            self._env_cfg[section][param] = cast_val
                            applied[key] = cast_val
                            if self.verbose >= 1:
                                print(f"[LiveTuning] {key} -> {cast_val}")
                        except (TypeError, ValueError) as exc:
                            _log.warning("LiveTuning: could not apply %r=%r: %s", key, value, exc)

        if applied:
            self._applied.append(applied)

        # Emit status snapshot for GUI polling.
        self._write_status()

    def _write_status(self) -> None:
        if self._publish_status is None:
            return
        try:
            status = {
                "timesteps": self.num_timesteps,
                "applied_count": len(self._applied),
            }
            # Pull metrics from SB3 logger.
            if self.logger is not None:
                for key in ("rollout/ep_rew_mean", "rollout/ep_len_mean",
                            "train/loss", "train/policy_gradient_loss",
                            "train/value_loss", "train/entropy_loss",
                            "train/learning_rate"):
                    try:
                        status[key] = float(self.logger.name_to_value[key])
                    except (KeyError, AttributeError, TypeError):
                        pass
            self._publish_status(status)
        except Exception:
            pass
