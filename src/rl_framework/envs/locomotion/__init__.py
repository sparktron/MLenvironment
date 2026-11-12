from rl_framework.core.registry import TaskSpec, register_task
from rl_framework.envs.locomotion.gui_schema import gui_schema
from rl_framework.envs.locomotion.validation import validate_config
from rl_framework.envs.locomotion.walker_bullet import WalkerBulletEnv

def _train_callbacks(cfg, paths, slot):
    from .training import train_callbacks
    return train_callbacks(cfg, paths, slot)


def _studies():
    from .quality_study import study_specs
    return study_specs()


register_task(TaskSpec(
    studies=_studies, name="walker_bullet", make_env=WalkerBulletEnv, validate_config=validate_config, gui_schema=gui_schema,
    train_callbacks=_train_callbacks,
    capabilities=frozenset({"off_policy", "best_model_eval", "velocity_target_ramp"}),
))

__all__ = ["WalkerBulletEnv"]
