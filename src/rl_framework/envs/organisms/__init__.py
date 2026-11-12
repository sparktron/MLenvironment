from rl_framework.core.registry import TaskSpec, register_task
from rl_framework.envs.organisms.evaluation import build_eval_vec_env, evaluate_rollouts
from rl_framework.envs.organisms.gui_schema import gui_schema
from rl_framework.envs.organisms.arena_parallel import OrganismArenaParallelEnv
from rl_framework.envs.organisms.validation import validate_config


def _make_arena(cfg):
    return OrganismArenaParallelEnv(cfg, render_mode=cfg.get("render_mode"))


def _train_callbacks(cfg, paths, slot):
    from .training import train_callbacks
    return train_callbacks(cfg, paths, slot)


def _build_train_vec_env(cfg, paths, render_env_index):
    from .training import build_train_vec_env
    return build_train_vec_env(cfg, paths, render_env_index)


def _studies():
    from .quality_study import study_specs
    return study_specs()


register_task(TaskSpec(
    studies=_studies, name="organism_arena_parallel", make_env=_make_arena, validate_config=validate_config, gui_schema=gui_schema,
    build_train_vec_env=_build_train_vec_env, train_callbacks=_train_callbacks,
    build_eval_vec_env=build_eval_vec_env, evaluate_rollouts=evaluate_rollouts,
))

__all__ = ["OrganismArenaParallelEnv"]
