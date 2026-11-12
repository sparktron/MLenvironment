from rl_framework.core.registry import TaskSpec, register_task
from rl_framework.envs.locomotion.walker_bullet import WalkerBulletEnv

register_task(TaskSpec(name="walker_bullet", make_env=WalkerBulletEnv))

__all__ = ["WalkerBulletEnv"]
