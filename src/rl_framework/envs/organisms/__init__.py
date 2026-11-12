from rl_framework.core.registry import TaskSpec, register_task
from rl_framework.envs.organisms.gui_schema import gui_schema
from rl_framework.envs.organisms.arena_parallel import OrganismArenaParallelEnv
from rl_framework.envs.organisms.validation import validate_config


def _make_arena(cfg):
    return OrganismArenaParallelEnv(cfg, render_mode=cfg.get("render_mode"))


register_task(TaskSpec(
    name="organism_arena_parallel", make_env=_make_arena, validate_config=validate_config, gui_schema=gui_schema,
))

__all__ = ["OrganismArenaParallelEnv"]
