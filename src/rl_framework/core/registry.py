"""Task registration and lazy plugin loading."""
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any, Callable

from rl_framework.plugins import PLUGIN_MODULES


@dataclass(frozen=True)
class TaskSpec:
    name: str
    make_env: Callable[..., Any]
    validate_config: Callable[..., Any] | None = None
    gui_schema: Callable[..., Any] | None = None
    build_train_vec_env: Callable[..., Any] | None = None
    build_eval_vec_env: Callable[..., Any] | None = None
    evaluate_rollouts: Callable[..., Any] | None = None
    train_callbacks: Callable[..., Any] | None = None
    capabilities: frozenset[str] = frozenset()


_tasks: dict[str, TaskSpec] = {}


def register_task(spec: TaskSpec) -> None:
    existing = _tasks.get(spec.name)
    if existing is not None and existing != spec:
        raise ValueError(f"Task {spec.name!r} is already registered")
    _tasks[spec.name] = spec


def load_plugins() -> None:
    extra = os.environ.get("RL_FRAMEWORK_PLUGINS", "")
    for module in (*PLUGIN_MODULES, *(name.strip() for name in extra.split(",") if name.strip())):
        importlib.import_module(module)


def registered_tasks() -> tuple[str, ...]:
    load_plugins()
    return tuple(sorted(_tasks))


def get_task(name: str) -> TaskSpec:
    load_plugins()
    try:
        return _tasks[name]
    except KeyError:
        raise ValueError(f"Unknown env_type={name}; registered tasks: {', '.join(sorted(_tasks))}") from None


def make_env(env_type: str, cfg: dict[str, Any]) -> Any:
    return get_task(env_type).make_env(cfg)
