from rl_framework.core.registry import TaskSpec, register_task
from rl_framework.envs.organisms.arena_parallel import OrganismArenaParallelEnv


def _make_arena(cfg):
    return OrganismArenaParallelEnv(cfg, render_mode=cfg.get("render_mode"))


register_task(TaskSpec(name="organism_arena_parallel", make_env=_make_arena))

__all__ = ["OrganismArenaParallelEnv"]
