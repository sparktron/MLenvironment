from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(config_name: str, config_dir: str | Path) -> DictConfig:
    path = Path(config_dir) / f"{config_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return OmegaConf.load(path)


def to_container(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def validate_experiment_config(cfg: dict[str, Any]) -> None:
    """Validate required experiment config structure and safe value ranges."""
    _require_keys(cfg, ["experiment_name", "seed", "output", "environment", "training"])
    _validate_experiment_name(cfg["experiment_name"])
    _require_keys(cfg["output"], ["base_dir"])
    _require_keys(cfg["environment"], ["type"])
    _require_keys(cfg["training"], ["total_timesteps"])

    _ensure_int(cfg["seed"], "seed")
    _ensure_int(cfg["training"]["total_timesteps"], "training.total_timesteps", min_value=1)

    num_envs = cfg["training"].get("num_envs", 1)
    _ensure_int(num_envs, "training.num_envs", min_value=1)

    device = cfg["training"].get("device", "auto")
    _validate_device(device)

    eval_cfg = cfg.get("evaluation", {})
    if "episodes" in eval_cfg:
        _ensure_int(eval_cfg["episodes"], "evaluation.episodes", min_value=1)

    self_play_cfg = cfg.get("self_play", {})
    if self_play_cfg.get("enabled", False):
        _ensure_int(self_play_cfg.get("snapshot_freq", 5000), "self_play.snapshot_freq", min_value=1)
        _ensure_int(self_play_cfg.get("max_league_size", 10), "self_play.max_league_size", min_value=1)


def _validate_experiment_name(name: Any) -> None:
    """Reject experiment names that could escape the output directory."""
    if not isinstance(name, str) or not name:
        raise ValueError("experiment_name must be a non-empty string")
    if ".." in name or name.startswith("/") or name.startswith("\\"):
        raise ValueError(
            f"experiment_name contains unsafe path components: {name!r}"
        )
    # Block any OS path separator to keep names as simple directory segments.
    if re.search(r'[/\\]', name):
        raise ValueError(
            f"experiment_name must not contain path separators: {name!r}"
        )


def _validate_device(value: Any) -> None:
    """Accept 'auto', 'cpu', 'cuda', or 'cuda:<int>'.

    'auto' (the default) selects CUDA when an NVIDIA GPU is available and
    falls back to CPU otherwise.  Pass 'cpu' to force CPU explicitly.
    """
    if not isinstance(value, str):
        raise TypeError(f"training.device must be a string, got {type(value).__name__}")
    if value not in ("auto", "cpu", "cuda") and not re.fullmatch(r"cuda:\d+", value):
        raise ValueError(
            f"training.device must be 'auto', 'cpu', 'cuda', or 'cuda:<N>' (e.g. 'cuda:0'), got {value!r}"
        )


def _require_keys(d: dict[str, Any], keys: list[str]) -> None:
    for key in keys:
        if key not in d:
            raise KeyError(f"Missing required config key: {key}")


def _ensure_int(value: Any, key: str, min_value: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Config key '{key}' must be int, got {type(value).__name__}")
    if min_value is not None and value < min_value:
        raise ValueError(f"Config key '{key}' must be >= {min_value}, got {value}")
