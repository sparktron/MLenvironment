import numpy as np

from rl_framework.envs.organisms.arena_parallel import OrganismArenaParallelEnv
from rl_framework.envs.registry import make_env


def _make_arena(**battle_rules):
    cfg = {"battle_rules": battle_rules} if battle_rules else {}
    return OrganismArenaParallelEnv(cfg)


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


# ---------------------------------------------------------------------------
# Feature 4: Egocentric observation
# ---------------------------------------------------------------------------


def test_obs_shape_is_7() -> None:
    env = _make_arena()
    obs, _ = env.reset()
    for agent_id, o in obs.items():
        assert o.shape == (7,), f"{agent_id} obs shape mismatch"


def test_obs_is_symmetric_across_slots() -> None:
    env = _make_arena()
    obs, _ = env.reset()
    # agent_0 spawns at (-0.6, 0), agent_1 at (0.6, 0)
    assert obs["agent_0"][3] > 0, "agent_0 rel_opp_x should be positive"
    assert obs["agent_1"][3] < 0, "agent_1 rel_opp_x should be negative"
    assert obs["agent_0"][2] == obs["agent_1"][2], "health should be equal at reset"


def test_velocity_is_zero_on_reset() -> None:
    env = _make_arena()
    obs, _ = env.reset()
    for agent_id, o in obs.items():
        assert o[0] == 0.0 and o[1] == 0.0, (
            f"{agent_id} velocity should be zero on reset"
        )


# ---------------------------------------------------------------------------
# Feature 6: Continuous attack falloff
# ---------------------------------------------------------------------------


def test_attack_damage_scales_with_distance() -> None:
    env = _make_arena(damage=0.1, attack_range=0.2, cooldown_steps=1)
    env.reset()
    env.state["agent_0"]["pos"] = np.array([0.0, 0.0], dtype=np.float32)
    env.state["agent_1"]["pos"] = np.array(
        [env.rules.attack_range / 2, 0.0], dtype=np.float32
    )
    env.state["agent_0"]["cooldown"] = 0
    # Size is updated during step; with no growth it stays at base_size=1.0
    _, rewards, _, _, _ = env.step(
        {
            "agent_0": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "agent_1": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        }
    )
    expected = env.rules.damage * 0.5  # linear falloff at half range, size=1, scale=1
    assert abs(rewards["agent_0"] - expected) < 0.01


def test_attack_zero_damage_beyond_range() -> None:
    env = _make_arena(damage=0.1, attack_range=0.2, cooldown_steps=1)
    env.reset()
    env.state["agent_0"]["pos"] = np.array([0.0, 0.0], dtype=np.float32)
    env.state["agent_1"]["pos"] = np.array(
        [env.rules.attack_range + 0.1, 0.0], dtype=np.float32
    )
    env.state["agent_0"]["cooldown"] = 0
    _, rewards, _, _, _ = env.step(
        {
            "agent_0": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "agent_1": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        }
    )
    assert rewards["agent_0"] == 0.0


# ---------------------------------------------------------------------------
# Feature 2: Episode outcome in infos
# ---------------------------------------------------------------------------


def test_infos_contains_episode_outcome_on_ko() -> None:
    env = _make_arena(damage=10.0, attack_range=5.0, cooldown_steps=0)
    env.reset()
    env.state["agent_1"]["health"] = 0.001
    env.state["agent_0"]["pos"] = env.state["agent_1"]["pos"].copy()
    env.state["agent_0"]["cooldown"] = 0
    _, _, terminations, _, infos = env.step(
        {
            "agent_0": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "agent_1": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        }
    )
    assert any(terminations.values()), "episode should terminate"
    for info in infos.values():
        assert "episode_outcome" in info
        assert info["episode_outcome"]["outcome"] == "ko"


def test_infos_contains_timeout_on_truncation() -> None:
    env = _make_arena(max_steps=1)
    env.reset()
    _, _, _, truncations, infos = env.step(
        {
            "agent_0": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "agent_1": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        }
    )
    assert all(truncations.values())
    for info in infos.values():
        assert info["episode_outcome"]["outcome"] == "timeout"


# ---------------------------------------------------------------------------
# Feature 5: update_live_params
# ---------------------------------------------------------------------------


def test_reward_annealing_scales_damage_to_zero() -> None:
    env = _make_arena(damage=0.1, attack_range=5.0, cooldown_steps=0)
    env.reset()
    env.update_live_params({"reward.damage_scale": 0.0})
    env.state["agent_0"]["pos"] = env.state["agent_1"]["pos"].copy()
    env.state["agent_0"]["cooldown"] = 0
    _, rewards, _, _, _ = env.step(
        {
            "agent_0": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "agent_1": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        }
    )
    assert rewards["agent_0"] == 0.0, "damage scale=0 should produce zero reward"


def test_curriculum_battle_rules_update_via_live_params() -> None:
    env = _make_arena(cooldown_steps=3)
    original_cooldown = env.rules.cooldown_steps
    env.update_live_params({"battle_rules.cooldown_steps": original_cooldown + 2})
    assert env.rules.cooldown_steps == original_cooldown + 2
