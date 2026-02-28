from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import Any

from rl_framework.training.sb3_runner import train


def _set_nested(d: dict[str, Any], key: str, value: Any) -> None:
    keys = key.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur[k]
    cur[keys[-1]] = value


def run_sweep(cfg: dict[str, Any]) -> None:
    sweep_cfg = cfg.get("sweep", {})
    params = sweep_cfg.get("parameters", {})
    keys = list(params.keys())
    values = [params[k] for k in keys]

    for combo in product(*values):
        run_cfg = deepcopy(cfg)
        name_suffix = []
        for k, v in zip(keys, combo):
            _set_nested(run_cfg, k, v)
            name_suffix.append(f"{k.split('.')[-1]}_{v}")
        run_cfg["experiment_name"] = f"{cfg['experiment_name']}__{'__'.join(name_suffix)}"
        train(run_cfg)
