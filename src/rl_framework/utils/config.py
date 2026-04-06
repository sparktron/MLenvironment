from __future__ import annotations

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
    _require_keys(cfg["output"], ["base_dir"])
    _require_keys(cfg["environment"], ["type"])
    _require_keys(cfg["training"], ["total_timesteps"])

    _ensure_int(cfg["seed"], "seed")
    _ensure_int(cfg["training"]["total_timesteps"], "training.total_timesteps", min_value=1)

    num_envs = cfg["training"].get("num_envs", 1)
    _ensure_int(num_envs, "training.num_envs", min_value=1)

    eval_cfg = cfg.get("evaluation", {})
    if "episodes" in eval_cfg:
        _ensure_int(eval_cfg["episodes"], "evaluation.episodes", min_value=1)

    self_play_cfg = cfg.get("self_play", {})
    if self_play_cfg.get("enabled", False):
        _ensure_int(self_play_cfg.get("snapshot_freq", 5000), "self_play.snapshot_freq", min_value=1)
        _ensure_int(self_play_cfg.get("max_league_size", 10), "self_play.max_league_size", min_value=1)


def _require_keys(d: dict[str, Any], keys: list[str]) -> None:
    for key in keys:
        if key not in d:
            raise KeyError(f"Missing required config key: {key}")


def _ensure_int(value: Any, key: str, min_value: int | None = None) -> None:
    if not isinstance(value, int):
        raise TypeError(f"Config key '{key}' must be int, got {type(value).__name__}")
    if min_value is not None and value < min_value:
        raise ValueError(f"Config key '{key}' must be >= {min_value}, got {value}")
