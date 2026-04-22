import numpy as np

from rl_framework.envs.registry import make_env


def test_walker_env_api() -> None:
    cfg = {
        "type": "walker_bullet",
        "seed": 1,
        "sim": {"gravity": -9.81, "mass": 3.0, "friction": 0.8, "max_force": 30.0},
    }
    env = make_env("walker_bullet", cfg)
    obs, info = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape
    assert isinstance(info, dict)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    env.close()


def test_organism_env_api() -> None:
    cfg = {"type": "organism_arena_parallel", "seed": 1, "sim": {"arena_half_extent": 1.0}}
    env = make_env("organism_arena_parallel", cfg)
    observations, infos = env.reset(seed=1)
    assert "agent_0" in observations and "agent_1" in observations
    actions = {agent: env.action_space(agent).sample() for agent in observations.keys()}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    assert set(rewards.keys()) == {"agent_0", "agent_1"}
    assert set(terminations.keys()) == {"agent_0", "agent_1"}
    assert set(truncations.keys()) == {"agent_0", "agent_1"}


def test_organism_env_render_rgb_array() -> None:
    """Smoke test: rgb_array render returns HxWx3 uint8 frames."""
    cfg = {"type": "organism_arena_parallel", "seed": 1, "render_mode": "rgb_array"}
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=1)
    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.dtype == np.uint8
    env.close()


def test_organism_env_obs_shape() -> None:
    cfg = {"type": "organism_arena_parallel", "seed": 1}
    env = make_env("organism_arena_parallel", cfg)
    observations, _ = env.reset(seed=1)
    for agent, obs in observations.items():
        assert obs.shape == env.observation_space(agent).shape, f"{agent} obs shape mismatch"


def test_organism_env_terminates_when_health_depleted() -> None:
    """An agent whose health reaches 0 should be marked terminated."""
    cfg = {
        "type": "organism_arena_parallel",
        "seed": 0,
        "battle_rules": {"damage": 10.0, "attack_range": 5.0, "cooldown_steps": 0, "max_steps": 400},
    }
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=0)
    # Force agent_0's health to near zero; only agent_1 attacks.
    env.state["agent_0"]["health"] = 0.001
    env.state["agent_0"]["pos"] = np.array([0.0, 0.0], dtype=np.float32)
    env.state["agent_1"]["pos"] = np.array([0.0, 0.0], dtype=np.float32)
    no_attack = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    attack = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    _, rewards, terminations, _, _ = env.step({"agent_0": no_attack, "agent_1": attack})
    assert terminations["agent_0"], "agent_0 should be terminated after health reaches 0"
    assert rewards["agent_1"] > 0, "winner (agent_1) should receive a positive reward"
    assert env.agents == [], "agents list should be cleared after termination"


def test_organism_env_truncates_at_max_steps() -> None:
    """Episode should truncate (not terminate) when max_steps is reached."""
    cfg = {
        "type": "organism_arena_parallel",
        "seed": 0,
        "battle_rules": {"damage": 0.0, "max_steps": 3},
    }
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=0)
    noop = np.zeros(3, dtype=np.float32)
    for step in range(3):
        _, _, terminations, truncations, _ = env.step({"agent_0": noop, "agent_1": noop})
    # On the final step both agents should be truncated, not terminated.
    assert all(truncations.values()), "all agents should be truncated at max_steps"
    assert not any(terminations.values()), "no agent should be terminated on truncation"
    assert env.agents == [], "agents list should be cleared after truncation"


def test_organism_env_agents_cleared_after_game_over() -> None:
    """self.agents must be empty after any terminal/truncated step."""
    cfg = {"type": "organism_arena_parallel", "seed": 0, "battle_rules": {"max_steps": 1}}
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=0)
    noop = np.zeros(3, dtype=np.float32)
    env.step({"agent_0": noop, "agent_1": noop})
    assert env.agents == []
