import numpy as np

from rl_framework.envs.registry import make_env


def test_walker_seed_reproducibility() -> None:
    cfg = {
        "type": "walker_bullet",
        "seed": 123,
        "sim": {"gravity": -9.81, "mass": 3.0, "friction": 0.8, "max_force": 30.0},
        "reset_randomization": {"position_xy_noise": 0.02, "yaw_noise": 0.1},
    }
    env_a = make_env("walker_bullet", cfg)
    env_b = make_env("walker_bullet", cfg)
    obs_a, _ = env_a.reset(seed=123)
    obs_b, _ = env_b.reset(seed=123)
    np.testing.assert_allclose(obs_a, obs_b, atol=1e-6)
    env_a.close()
    env_b.close()
