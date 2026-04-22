from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback


class LiveTuningCallback(BaseCallback):
    """SB3 callback that reads a JSON file each rollout to apply real-time parameter changes.

    The GUI writes tuning requests to a JSON file.  This callback picks them
    up at the end of every rollout, applies supported changes, and clears the
    file so each command is consumed exactly once.

    Supported live-tunable parameters
    ---------------------------------
    - ``learning_rate``          (float)  – PPO optimizer LR
    - ``reward.*``               (float)  – any key under env_cfg["reward"]
    - ``termination.*``          (float)  – any key under env_cfg["termination"]
    - ``domain_randomization.*`` (float)  – any key under env_cfg["domain_randomization"]
    - ``battle_rules.*``         (float)  – any key under env_cfg["battle_rules"]

    File format (``live_tuning.json``)::

        {
            "learning_rate": 0.001,
            "reward.target_velocity": 2.0,
            "termination.max_tilt_radians": 0.6
        }
    """

    ENV_SECTIONS = ("reward", "termination", "domain_randomization", "battle_rules", "morphology", "sim")

    def __init__(
        self,
        tuning_file: str | Path,
        env_cfg: dict[str, Any],
        status_file: str | Path | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._tuning_file = Path(tuning_file)
        self._env_cfg = env_cfg
        self._status_file = Path(status_file) if status_file else None
        self._applied: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if not self._tuning_file.exists():
            return
        try:
            raw = self._tuning_file.read_text(encoding="utf-8").strip()
            if not raw:
                return
            params = json.loads(raw)
            if not isinstance(params, dict) or not params:
                return
        except (json.JSONDecodeError, OSError):
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
                except (TypeError, ValueError):
                    pass
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
                        except (TypeError, ValueError):
                            pass

        if applied:
            self._applied.append(applied)

        # Clear the file so we don't re-apply.
        try:
            self._tuning_file.write_text("", encoding="utf-8")
        except OSError:
            pass

        # Write status for the GUI to read.
        self._write_status()

    def _write_status(self) -> None:
        if self._status_file is None:
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
            self._status_file.write_text(json.dumps(status), encoding="utf-8")
        except OSError:
            pass
