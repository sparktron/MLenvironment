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
    cfg = {
        "type": "organism_arena_parallel",
        "seed": 1,
        "sim": {"arena_half_extent": 1.0},
    }
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
        assert obs.shape == env.observation_space(agent).shape, (
            f"{agent} obs shape mismatch"
        )


def test_organism_env_terminates_when_health_depleted() -> None:
    """An agent whose health reaches 0 should be marked terminated."""
    cfg = {
        "type": "organism_arena_parallel",
        "seed": 0,
        "battle_rules": {
            "damage": 10.0,
            "attack_range": 5.0,
            "cooldown_steps": 0,
            "max_steps": 400,
        },
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
    assert terminations["agent_0"], (
        "agent_0 should be terminated after health reaches 0"
    )
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
        _, _, terminations, truncations, _ = env.step(
            {"agent_0": noop, "agent_1": noop}
        )
    # On the final step both agents should be truncated, not terminated.
    assert all(truncations.values()), "all agents should be truncated at max_steps"
    assert not any(terminations.values()), "no agent should be terminated on truncation"
    assert env.agents == [], "agents list should be cleared after truncation"


def test_organism_env_agents_cleared_after_game_over() -> None:
    """self.agents must be empty after any terminal/truncated step."""
    cfg = {
        "type": "organism_arena_parallel",
        "seed": 0,
        "battle_rules": {"max_steps": 1},
    }
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=0)
    noop = np.zeros(3, dtype=np.float32)
    env.step({"agent_0": noop, "agent_1": noop})
    assert env.agents == []


def test_organism_env_obs_reports_self_velocity() -> None:
    """Feature 4: the first two obs components are displacement since last step."""
    cfg = {"type": "organism_arena_parallel", "seed": 0}
    env = make_env("organism_arena_parallel", cfg)
    observations, _ = env.reset(seed=0)
    # Velocity reads zero on the first observation after reset.
    assert observations["agent_0"][0] == 0.0 and observations["agent_0"][1] == 0.0
    # Move agent_0 by a known amount: action[:2] is scaled by 0.05 internally.
    move = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    noop = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    observations, _, _, _, _ = env.step({"agent_0": move, "agent_1": noop})
    assert abs(observations["agent_0"][0] - 0.05) < 1e-5, "vel_x should be +0.05"
    assert abs(observations["agent_0"][1]) < 1e-6, "vel_y should be 0"


def test_organism_env_obs_hides_opponent_beyond_sensing_radius() -> None:
    """Feature 4: opponent relative pos/health zeroed past sensing_radius."""
    cfg = {
        "type": "organism_arena_parallel",
        "seed": 0,
        "battle_rules": {"sensing_radius": 0.5},
    }
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=0)
    env.state["agent_0"]["pos"] = np.array([0.0, 0.0], dtype=np.float32)
    env.state["agent_1"]["pos"] = np.array([5.0, 0.0], dtype=np.float32)
    obs = env._obs("agent_0")
    # rel_opp_x, rel_opp_y, opp_health are obs[3:6].
    assert obs[3] == 0.0 and obs[4] == 0.0 and obs[5] == 0.0


def test_organism_env_attack_damage_scales_with_distance() -> None:
    """Feature 6: linear falloff deals ~50% damage at half attack_range."""
    cfg = {
        "type": "organism_arena_parallel",
        "seed": 0,
        "battle_rules": {"damage": 0.1, "attack_range": 0.4, "cooldown_steps": 0},
    }
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=0)
    env.state["agent_0"]["pos"] = np.array([0.0, 0.0], dtype=np.float32)
    env.state["agent_1"]["pos"] = np.array([0.2, 0.0], dtype=np.float32)  # half range
    env.state["agent_0"]["cooldown"] = 0
    attack = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    noop = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    _, rewards, _, _, _ = env.step({"agent_0": attack, "agent_1": noop})
    expected = 0.1 * 0.5  # linear falloff at half range, size == 1.0
    assert abs(rewards["agent_0"] - expected) < 0.01


def test_organism_env_attack_zero_damage_beyond_range() -> None:
    """Feature 6: no damage when the defender sits past attack_range."""
    cfg = {
        "type": "organism_arena_parallel",
        "seed": 0,
        "battle_rules": {"damage": 0.1, "attack_range": 0.2, "cooldown_steps": 0},
    }
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=0)
    env.state["agent_0"]["pos"] = np.array([0.0, 0.0], dtype=np.float32)
    env.state["agent_1"]["pos"] = np.array([0.3, 0.0], dtype=np.float32)
    env.state["agent_0"]["cooldown"] = 0
    attack = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    noop = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    _, rewards, _, _, _ = env.step({"agent_0": attack, "agent_1": noop})
    assert rewards["agent_0"] == 0.0


def test_organism_env_attack_binary_mode_is_a_cliff() -> None:
    """Feature 6: binary mode deals full damage inside range, none outside."""
    cfg = {
        "type": "organism_arena_parallel",
        "seed": 0,
        "battle_rules": {
            "damage": 0.1,
            "attack_range": 0.2,
            "cooldown_steps": 0,
            "attack_falloff": "binary",
        },
    }
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=0)
    env.state["agent_0"]["pos"] = np.array([0.0, 0.0], dtype=np.float32)
    env.state["agent_1"]["pos"] = np.array([0.19, 0.0], dtype=np.float32)  # inside
    env.state["agent_0"]["cooldown"] = 0
    attack = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    noop = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    _, rewards, _, _, _ = env.step({"agent_0": attack, "agent_1": noop})
    assert abs(rewards["agent_0"] - 0.1) < 1e-6, "full damage inside range"


def test_organism_env_infos_episode_outcome_on_ko() -> None:
    """Feature 2: terminal step annotates infos with a 'ko' episode_outcome."""
    cfg = {
        "type": "organism_arena_parallel",
        "seed": 0,
        "battle_rules": {"damage": 10.0, "attack_range": 5.0, "cooldown_steps": 0},
    }
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=0)
    env.state["agent_1"]["health"] = 0.001
    env.state["agent_0"]["pos"] = np.array([0.0, 0.0], dtype=np.float32)
    env.state["agent_1"]["pos"] = np.array([0.0, 0.0], dtype=np.float32)
    attack = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    noop = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    _, _, terminations, _, infos = env.step({"agent_0": attack, "agent_1": noop})
    assert any(terminations.values()), "episode should terminate on KO"
    for info in infos.values():
        assert "episode_outcome" in info
        assert info["episode_outcome"]["outcome"] == "ko"
        assert info["episode_outcome"]["winner"] == "agent_0"
        assert info["episode_outcome"]["loser"] == "agent_1"


def test_organism_env_infos_episode_outcome_on_timeout() -> None:
    """Feature 2: truncation annotates infos with a 'timeout' episode_outcome."""
    cfg = {
        "type": "organism_arena_parallel",
        "seed": 0,
        "battle_rules": {"damage": 0.0, "max_steps": 1},
    }
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=0)
    noop = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    _, _, _, truncations, infos = env.step({"agent_0": noop, "agent_1": noop})
    assert all(truncations.values())
    for info in infos.values():
        assert info["episode_outcome"]["outcome"] == "timeout"
        assert info["episode_outcome"]["winner"] is None


def test_arena_metrics_callback_computes_win_rates() -> None:
    """ArenaMetricsCallback aggregates outcomes and records win rates."""
    from rl_framework.training.sb3_runner import ArenaMetricsCallback

    class _StubLogger:
        def __init__(self) -> None:
            self.records: dict[str, float] = {}

        def record(self, key: str, value: float) -> None:
            self.records[key] = value

    from types import SimpleNamespace

    cb = ArenaMetricsCallback()
    cb.model = SimpleNamespace(logger=_StubLogger())
    # Three episodes: agent_0 KO win, agent_1 KO win, a timeout.
    cb.locals = {
        "infos": [
            {"episode_outcome": {"outcome": "ko", "winner": "agent_0"}},
            {"episode_outcome": {"outcome": "ko", "winner": "agent_1"}},
            {"episode_outcome": {"outcome": "timeout", "winner": None}},
            {"step": 5},  # non-terminal info — ignored
        ]
    }
    cb._on_step()
    cb._on_rollout_end()
    recs = cb.logger.records
    assert abs(recs["arena/agent_0_win_rate"] - 1 / 3) < 1e-9
    assert abs(recs["arena/agent_1_win_rate"] - 1 / 3) < 1e-9
    assert abs(recs["arena/timeout_rate"] - 1 / 3) < 1e-9
    assert recs["arena/episode_outcomes"] == 3
