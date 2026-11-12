"""Definitions contributed by tasks to the quality-study runner."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from rl_framework.core.registry import get_task, registered_tasks


@dataclass(frozen=True)
class StudySpec:
    name: str
    result_key: str
    run: Callable[..., dict[str, Any]]
    variants: int
    default_steps: int
    order: int
    per_seed: bool = True
    include_in_all: bool = False
    uses_source: bool = False
    uses_approval: bool = False
    uses_wall_clock: bool = False


def registered_studies() -> tuple[StudySpec, ...]:
    studies = []
    for name in registered_tasks():
        provider = get_task(name).studies
        if provider is not None:
            studies.extend(provider())
    names = [study.name for study in studies]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate quality study name")
    return tuple(sorted(studies, key=lambda study: study.order))
