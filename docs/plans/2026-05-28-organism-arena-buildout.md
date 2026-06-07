# Organism Arena Buildout — Implementation Plan

**Date:** 2026-05-28
**Status:** Ready for implementation
**Estimated scope:** 6 features, sequenced by dependency

---

## Context

The organism arena (`src/rl_framework/envs/organisms/arena_parallel.py`) is a 2-agent PettingZoo ParallelEnv with a mostly-complete skeleton. The self-play infrastructure exists but has one missing connector. Training is currently blind (no TensorBoard metrics), the observation design has a correctness flaw, and the core combat mechanic produces a gradient cliff. This plan addresses all six issues in dependency order.

**Key files:**
- `src/rl_framework/envs/organisms/arena_parallel.py` — the environment
- `src/rl_framework/training/self_play_callback.py` — league snapshot machinery (complete)
- `src/rl_framework/training/sb3_runner.py` — training runner (wires callbacks + SuperSuit)
- `src/rl_framework/training/eval_runner.py` — evaluation runner
- `src/rl_framework/training/curriculum_callback.py` — curriculum hook (used by walker, absent for arena)
- `src/rl_framework/cli/main.py` — CLI entry point
- `tests/test_env_api.py` — arena unit tests (currently ~5 tests, unit-level only)
- `src/rl_framework/configs/experiments/organisms_fight_arena.yaml`
- `src/rl_framework/configs/experiments/organisms_growth_competition.yaml`

**Dependency order:**
```
Feature 4 (Egocentric obs)   →  Feature 1 (SelfPlayEnvWrapper)
                              →  Feature 2 (Instrumentation)  →  Feature 5 (Curriculum)
                                                               →  Feature 3 (Eval harness)
Feature 6 (Attack falloff)   (independent — any time)
```

Start with Feature 4 (egocentric obs) and Feature 6 (attack falloff) since they are independent and foundational. Then Feature 2 (instrumentation), Feature 1 (self-play wrapper), Feature 3 (eval harness), Feature 5 (curriculum) in that order.

---

## Feature 4 — Egocentric Observation Reframing

**File:** `src/rl_framework/envs/organisms/arena_parallel.py`
**Complexity:** Low
**Blocks:** Features 1, 2, 3, 5 (do this first to avoid breaking their tests)

### Problem

`_obs()` (lines 90–105) returns each agent's absolute `(x, y)` position combined with the opponent's *relative* displacement — a mixed frame. Because `agent_0` always spawns at `(-0.6, 0)` and `agent_1` at `(+0.6, 0)`, the global self-position is structurally asymmetric: the same physical situation produces different observation values depending on which agent slot the shared policy is filling. The shared PPO policy therefore learns two different input distributions glued together.

### Change

Replace absolute self-position with self-velocity. New 7D obs:

```
[vel_x, vel_y, health, rel_opp_x, rel_opp_y, opp_health, cooldown]
```

Where:
- `vel_x`, `vel_y` = displacement from previous position (computed as `pos - prev_pos`). On the first step after reset, use `[0.0, 0.0]`.
- `rel_opp_x`, `rel_opp_y` = opponent position minus self position (already correct in existing code)
- `health`, `opp_health`, `cooldown` = unchanged

### Implementation steps

1. Add `self._prev_positions: dict[str, np.ndarray]` to `__init__`, initialized to `{}`.
2. In `reset()`, after spawning agents, populate `_prev_positions[agent_id] = state["pos"].copy()` for each agent.
3. In `_obs(agent_id)`, compute:
   ```python
   prev = self._prev_positions.get(agent_id, self._agents[agent_id]["pos"])
   vel = self._agents[agent_id]["pos"] - prev
   self._prev_positions[agent_id] = self._agents[agent_id]["pos"].copy()
   ```
   Return `np.array([vel[0], vel[1], health, rel_x, rel_y, opp_health, cooldown], dtype=np.float32)`.
4. Update `observation_space` in `__init__` — bounds stay `[-inf, inf]` for vel; `[0, 1]` for health; `[-arena*2, arena*2]` for relative positions; `[0, 1]` for cooldown fraction.
5. Update the docstring comment for obs shape.

### Tests to add (`tests/test_env_api.py`)

```python
def test_obs_shape_is_7():
    env = make_arena_env()
    obs, _ = env.reset()
    for agent_id, o in obs.items():
        assert o.shape == (7,), f"{agent_id} obs shape mismatch"

def test_obs_is_symmetric_across_slots():
    """Same relative geometry should produce same obs for both agents when mirrored."""
    env = make_arena_env()
    obs, _ = env.reset()
    # After reset, both agents have zero velocity, symmetric health, mirrored positions
    # rel_opp_x for agent_0 should be positive; for agent_1 should be negative
    assert obs["agent_0"][3] > 0  # rel_opp_x > 0 (opponent is to the right)
    assert obs["agent_1"][3] < 0  # rel_opp_x < 0 (opponent is to the left)
    # health and cooldown should be equal
    assert obs["agent_0"][2] == obs["agent_1"][2]

def test_velocity_is_zero_on_reset():
    env = make_arena_env()
    obs, _ = env.reset()
    for o in obs.values():
        assert o[0] == 0.0 and o[1] == 0.0, "velocity should be zero on reset"
```

---

## Feature 6 — Continuous Attack Falloff

**File:** `src/rl_framework/envs/organisms/arena_parallel.py`
**Complexity:** Low
**Blocks:** nothing (independent)

### Problem

Attack resolves as a binary hit/miss at `arena_parallel.py:136–147`: full damage at `dist <= attack_range`, zero damage at `dist > attack_range`. This is a hard gradient cliff — the policy near the boundary gets identical gradient signal to one far away. The `attack_range` hyperparameter swept in `organisms_fight_arena.yaml` is meaningless as a cliff.

### Change

Replace binary threshold with linear falloff:

```python
if trigger > 0.5 and state["cooldown"] == 0:
    dist = np.linalg.norm(opp_state["pos"] - state["pos"])
    falloff = max(0.0, 1.0 - dist / self.rules.attack_range)
    damage_dealt = self.rules.damage * falloff * self._current_size(agent_id)
    if damage_dealt > 0:
        opp_state["health"] -= damage_dealt
        rewards[agent_id] += damage_dealt
        rewards[opponent] -= damage_dealt
        state["cooldown"] = self.rules.cooldown_steps
```

The `falloff` produces a gradient that teaches the policy to close distance before attacking. Full damage only at `dist == 0`; half damage at `dist == attack_range / 2`; zero at `dist >= attack_range`.

### Config change

Add optional `attack_falloff: linear` key to `BattleRules` (default: `linear`; future option: `binary` for backwards compat):

```python
@dataclass
class BattleRules:
    ...
    attack_falloff: str = "linear"  # "linear" or "binary"
```

No YAML changes needed unless users want to opt into `binary` for comparison.

### Tests to add

```python
def test_attack_damage_scales_with_distance():
    env = make_arena_env()
    env.reset()
    # Place agents at half attack_range → should deal ~50% damage
    env._agents["agent_0"]["pos"] = np.array([0.0, 0.0])
    env._agents["agent_1"]["pos"] = np.array([env.rules.attack_range / 2, 0.0])
    env._agents["agent_0"]["cooldown"] = 0
    obs, rewards, _, _, _ = env.step({"agent_0": np.array([0.0, 0.0, 1.0]),
                                       "agent_1": np.array([0.0, 0.0, 0.0])})
    expected = env.rules.damage * 0.5  # linear falloff at half range
    assert abs(rewards["agent_0"] - expected) < 0.01

def test_attack_zero_damage_beyond_range():
    env = make_arena_env()
    env.reset()
    env._agents["agent_0"]["pos"] = np.array([0.0, 0.0])
    env._agents["agent_1"]["pos"] = np.array([env.rules.attack_range + 0.1, 0.0])
    env._agents["agent_0"]["cooldown"] = 0
    _, rewards, _, _, _ = env.step({"agent_0": np.array([0.0, 0.0, 1.0]),
                                     "agent_1": np.array([0.0, 0.0, 0.0])})
    assert rewards["agent_0"] == 0.0
```

---

## Feature 2 — Arena Training Instrumentation

**Files:** `src/rl_framework/training/sb3_runner.py`, `src/rl_framework/envs/organisms/arena_parallel.py`
**Complexity:** Low–Medium
**Blocks:** Features 3, 5

### Problem A: No TensorBoard metrics for arena

`sb3_runner.py:147–155` wraps the arena env via SuperSuit without any `Monitor` wrapper. Walker training gets `rollout/ep_rew_mean` in TensorBoard; arena training shows nothing.

### Fix A: Add per-agent metrics callback

Rather than injecting `Monitor` into the SuperSuit chain (fragile), add a `ArenaMetricsCallback(BaseCallback)` that reads per-episode outcome from `infos` and logs to TensorBoard:

```python
class ArenaMetricsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self._ep_rewards = {"agent_0": [], "agent_1": []}
        self._ep_wins = {"agent_0": 0, "agent_1": 0}
        self._ep_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode_outcome" in info:
                winner = info["episode_outcome"].get("winner")
                if winner:
                    self._ep_wins[winner] += 1
                self._ep_count += 1
        return True

    def _on_rollout_end(self) -> None:
        if self._ep_count > 0:
            self.logger.record("arena/agent_0_win_rate",
                               self._ep_wins["agent_0"] / self._ep_count)
            self.logger.record("arena/agent_1_win_rate",
                               self._ep_wins["agent_1"] / self._ep_count)
            self.logger.record("arena/ep_count", self._ep_count)
            self._ep_wins = {"agent_0": 0, "agent_1": 0}
            self._ep_count = 0
```

Wire it in `sb3_runner.py` after the arena env check (alongside `SelfPlayCallback`).

### Problem B: No win/loss annotation in `infos`

`arena_parallel.py:149–163` terminates episodes and applies `±1.0` win bonuses, but `infos` only returns `{"step": self.step_count}`.

### Fix B: Annotate `infos` on terminal steps

In `step()`, when an agent's health drops to or below `win_health_threshold`:

```python
winner_id = agent_id  # the attacker
loser_id = opponent
infos[winner_id]["episode_outcome"] = {
    "winner": winner_id,
    "loser": loser_id,
    "outcome": "ko",
    "step": self.step_count,
}
infos[loser_id]["episode_outcome"] = {
    "winner": winner_id,
    "loser": loser_id,
    "outcome": "ko",
    "step": self.step_count,
}
```

On truncation (max_steps reached):

```python
for agent_id in active_agents:
    infos[agent_id]["episode_outcome"] = {
        "winner": None,
        "outcome": "timeout",
        "step": self.step_count,
    }
```

### Tests to add

```python
def test_infos_contains_episode_outcome_on_ko():
    env = make_arena_env()
    env.reset()
    # Force agent_1 health to near-zero and agent_0 in attack range
    env._agents["agent_1"]["health"] = 0.001
    env._agents["agent_0"]["pos"] = env._agents["agent_1"]["pos"].copy()
    env._agents["agent_0"]["cooldown"] = 0
    _, _, terminations, _, infos = env.step(
        {"agent_0": np.array([0.0, 0.0, 1.0]),
         "agent_1": np.array([0.0, 0.0, 0.0])}
    )
    assert any(terminations.values()), "episode should terminate"
    for info in infos.values():
        assert "episode_outcome" in info
        assert info["episode_outcome"]["outcome"] == "ko"

def test_infos_contains_timeout_on_truncation():
    env = make_arena_env(max_steps=1)
    env.reset()
    _, _, _, truncations, infos = env.step(
        {"agent_0": np.array([0.0, 0.0, 0.0]),
         "agent_1": np.array([0.0, 0.0, 0.0])}
    )
    assert all(truncations.values())
    for info in infos.values():
        assert info["episode_outcome"]["outcome"] == "timeout"
```

---

## Feature 1 — Complete the SelfPlayEnvWrapper

**New file:** `src/rl_framework/training/self_play_env_wrapper.py`
**Modified:** `src/rl_framework/training/sb3_runner.py`
**Complexity:** Low (~80 lines)
**Blocks:** Feature 5

### Problem

`SelfPlayCallback` (self_play_callback.py) saves policy snapshots to disk and exposes `sample_opponent()` to load a frozen past policy. But nothing routes `agent_1`'s observations through that frozen policy during rollout collection. Both agents still share the live policy. Enabling `self_play: enabled: true` is a silent no-op.

### Implementation

Create `SelfPlayEnvWrapper` as a `pettingzoo.utils.BaseWrapper`:

```python
# src/rl_framework/training/self_play_env_wrapper.py

import numpy as np
from pettingzoo.utils import BaseWrapper

class SelfPlayEnvWrapper(BaseWrapper):
    """Routes agent_1's actions through a frozen opponent policy.

    The live training policy controls agent_0. agent_1's observations
    are passed through SelfPlayCallback.sample_opponent() at each step.
    The frozen policy is re-sampled from the league at the start of
    each episode (on reset()).
    """

    FROZEN_AGENT = "agent_1"
    LIVE_AGENT = "agent_0"

    def __init__(self, env, self_play_callback):
        super().__init__(env)
        self._callback = self_play_callback
        self._frozen_policy = None

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # Sample a new frozen opponent at the start of each episode.
        # Falls back to random actions if league is empty (early training).
        self._frozen_policy = self._callback.sample_opponent()
        return obs, info

    def step(self, actions):
        # actions dict comes from the SB3 VecEnv and contains only agent_0's action.
        # We compute agent_1's action from the frozen policy.
        if self._frozen_policy is not None and self.FROZEN_AGENT in self.agents:
            frozen_obs = self._get_obs_for_agent(self.FROZEN_AGENT)
            frozen_action, _ = self._frozen_policy.predict(
                frozen_obs[np.newaxis], deterministic=True
            )
            actions[self.FROZEN_AGENT] = frozen_action[0]
        else:
            # No frozen policy yet (empty league) — use random action
            if self.FROZEN_AGENT in self.agents:
                actions[self.FROZEN_AGENT] = self.action_space(
                    self.FROZEN_AGENT
                ).sample()
        return super().step(actions)

    def _get_obs_for_agent(self, agent_id):
        """Return the current observation for a specific agent."""
        # Access the underlying env's observation for the given agent.
        return self.env._obs(agent_id)
```

### Wiring in `sb3_runner.py`

In the arena branch (after `SelfPlayCallback` is instantiated, around line 254):

```python
if self_play_cb is not None:
    # Wrap the raw PettingZoo env before SuperSuit conversion
    par_env = SelfPlayEnvWrapper(par_env, self_play_cb)
```

This must happen **before** `ss.pettingzoo_env_to_vec_env_v1(par_env)`.

### PettingZoo API constraint

When `agent_1` is eliminated mid-episode (health → 0), it is removed from `self.agents`. The wrapper must check `if self.FROZEN_AGENT in self.agents` before computing the frozen action — already handled in the step above.

### Tests to add (`tests/test_self_play_wrapper.py`)

```python
def test_wrapper_uses_frozen_policy_for_agent_1():
    """agent_1 actions should come from the frozen policy, not the live one."""
    env = make_arena_env()
    callback = make_mock_callback_with_league()  # returns a mock frozen policy
    wrapped = SelfPlayEnvWrapper(env, callback)
    obs, _ = wrapped.reset()
    # Spy on the frozen policy's predict call
    with mock.patch.object(callback._current_frozen, "predict",
                           wraps=callback._current_frozen.predict) as spy:
        wrapped.step({"agent_0": env.action_space("agent_0").sample()})
        spy.assert_called_once()

def test_wrapper_uses_random_action_when_league_empty():
    """No crash when the league has no snapshots yet."""
    env = make_arena_env()
    callback = make_mock_callback_empty_league()
    wrapped = SelfPlayEnvWrapper(env, callback)
    obs, _ = wrapped.reset()
    # Should not raise
    wrapped.step({"agent_0": env.action_space("agent_0").sample()})

def test_wrapper_resamples_opponent_on_each_reset():
    callback = make_mock_callback_with_league()
    env = make_arena_env()
    wrapped = SelfPlayEnvWrapper(env, callback)
    wrapped.reset()
    first_policy = wrapped._frozen_policy
    wrapped.reset()
    # With a league of >1 snapshot, policies may differ across resets
    assert callback.sample_opponent.call_count == 2
```

---

## Feature 3 — Head-to-Head Snapshot Eval Harness

**New file:** `src/rl_framework/training/arena_eval.py`
**Modified:** `src/rl_framework/cli/main.py`
**Complexity:** Medium
**Blocks:** Feature 5 (provides win-rate signal for curriculum)

### Problem

`eval_runner.py` runs both agent slots with the same shared policy — win rate against a frozen baseline is uncomputable. There is no CLI path to evaluate "does checkpoint A beat checkpoint B?"

### New function: `run_arena_eval()`

```python
# src/rl_framework/training/arena_eval.py

def run_arena_eval(
    policy_path: str,
    opponent_path: str,            # or "random" for random-action baseline
    cfg: dict,
    n_episodes: int = 100,
    swap_roles: bool = True,       # controls for positional bias
    output_path: str | None = None,
) -> dict:
    """
    Run N episodes of policy vs opponent. Returns:
      {
        "policy_win_rate": float,
        "opponent_win_rate": float,
        "timeout_rate": float,
        "policy_mean_return": float,
        "opponent_mean_return": float,
        "n_episodes": int,
      }

    When swap_roles=True, runs n_episodes/2 with policy as agent_0 and
    n_episodes/2 with policy as agent_1 (roles swapped) to control for
    the positional spawn bias.
    """
```

**Implementation sketch:**
1. Load `policy = PPO.load(policy_path)` and `opponent = PPO.load(opponent_path)` (or a random-action stub).
2. Build arena env via `make_env(cfg)`.
3. For each episode:
   - Wrap with `SelfPlayEnvWrapper`-style routing but with `policy` for the live slot and `opponent` for the frozen slot.
   - Run until terminal, collect per-agent returns and `infos["episode_outcome"]`.
4. If `swap_roles=True`, repeat with roles inverted and aggregate.
5. Write `output_path` as JSON if provided.

### CLI subcommand

In `cli/main.py`, add `arena-eval` subcommand:

```
python -m rl_framework.cli.main arena-eval \
    --config-name organisms_fight_arena \
    --policy outputs/fight/seed_0/checkpoints/model_100000.zip \
    --opponent outputs/fight/seed_0/checkpoints/model_50000.zip \
    --n-episodes 100 \
    --output outputs/fight/seed_0/evals/100k_vs_50k.json
```

Also support `--opponent random` to benchmark against a random-action baseline.

### Tests to add

```python
def test_arena_eval_returns_correct_keys():
    result = run_arena_eval("mock_policy", "random", cfg, n_episodes=10)
    assert {"policy_win_rate", "opponent_win_rate", "timeout_rate",
            "policy_mean_return", "n_episodes"}.issubset(result.keys())

def test_arena_eval_win_rates_sum_to_lte_one():
    result = run_arena_eval("mock_policy", "random", cfg, n_episodes=20)
    assert result["policy_win_rate"] + result["opponent_win_rate"] <= 1.0 + 1e-6

def test_arena_eval_role_swap_doubles_episode_count():
    result = run_arena_eval("mock_policy", "random", cfg,
                            n_episodes=10, swap_roles=True)
    assert result["n_episodes"] == 20
```

---

## Feature 5 — Win-Rate-Gated Curriculum + Dense-to-Sparse Reward Annealing

**New file:** `src/rl_framework/training/reward_annealing_callback.py`
**Modified:** `src/rl_framework/envs/organisms/arena_parallel.py` (add `update_live_params`), `src/rl_framework/training/sb3_runner.py`, YAML configs
**Complexity:** Medium
**Requires:** Features 2 (win/loss metrics) and 1 (self-play wrapper active)

### Part A: Dense-to-Sparse Reward Annealing

The damage-per-step reward trains agents to spam attacks for dense reward rather than to win matches. Anneal it to zero over the first `anneal_steps` timesteps, leaving only the terminal `±1.0` win/loss signal.

**New callback:**

```python
# src/rl_framework/training/reward_annealing_callback.py

class RewardAnnealingCallback(BaseCallback):
    """Linearly anneals the arena damage reward scale from 1.0 to 0.0.

    After anneal_steps total env steps, damage rewards are zeroed and only
    the terminal win/loss signal (+1 / -1) drives learning.
    """

    def __init__(self, anneal_steps: int = 500_000):
        super().__init__()
        self.anneal_steps = anneal_steps

    def _on_step(self) -> bool:
        scale = max(0.0, 1.0 - self.num_timesteps / self.anneal_steps)
        self.training_env.env_method("update_live_params",
                                      {"reward.damage_scale": scale})
        return True
```

**Add `update_live_params` to arena env:**

```python
def update_live_params(self, params: dict) -> None:
    """Apply live parameter overrides (used by callbacks)."""
    for key, value in params.items():
        if key == "reward.damage_scale":
            self._damage_scale = float(value)
        # Add more keys here as needed

# In __init__: self._damage_scale = 1.0
# In step(), scale damage: damage_dealt = base_damage * self._damage_scale
```

### Part B: Win-Rate-Gated Curriculum

Use the existing `CurriculumCallback` pattern. Add a `curriculum` section to arena YAML configs:

```yaml
curriculum:
  enabled: true
  metric: "arena/agent_0_win_rate"   # from ArenaMetricsCallback (Feature 2)
  levels:
    - threshold: 0.60               # advance when agent_0 wins 60%+
      level_params:
        battle_rules.cooldown_steps: 4   # opponent attacks slower
        battle_rules.damage: 0.03        # opponent hits lighter
    - threshold: 0.65
      level_params:
        battle_rules.cooldown_steps: 3   # back to default
        battle_rules.damage: 0.05
    - threshold: 0.70
      level_params:
        battle_rules.cooldown_steps: 2   # opponent attacks faster
        battle_rules.damage: 0.07
```

`CurriculumCallback` already reads `metric` from rollout info and calls `env_method("update_live_params", overrides)`. The arena env's new `update_live_params` method handles the `battle_rules.*` namespace:

```python
elif key.startswith("battle_rules."):
    field = key.removeprefix("battle_rules.")
    if hasattr(self.rules, field):
        setattr(self.rules, field, type(getattr(self.rules, field))(value))
```

**Wire both callbacks in `sb3_runner.py`** for arena runs (alongside existing SelfPlayCallback):

```python
if env_type == "organism_arena_parallel":
    callbacks.append(ArenaMetricsCallback())
    if cfg.get("reward_annealing", {}).get("enabled", False):
        anneal_steps = cfg["reward_annealing"].get("anneal_steps", 500_000)
        callbacks.append(RewardAnnealingCallback(anneal_steps))
    if cfg.get("curriculum", {}).get("enabled", False):
        callbacks.append(CurriculumCallback(cfg["curriculum"], ...))
```

### Tests to add

```python
def test_reward_annealing_scales_damage_to_zero():
    env = make_arena_env()
    env.reset()
    env.update_live_params({"reward.damage_scale": 0.0})
    # Force a hit
    env._agents["agent_0"]["pos"] = env._agents["agent_1"]["pos"].copy()
    env._agents["agent_0"]["cooldown"] = 0
    _, rewards, _, _, _ = env.step(
        {"agent_0": np.array([0.0, 0.0, 1.0]),
         "agent_1": np.array([0.0, 0.0, 0.0])}
    )
    assert rewards["agent_0"] == 0.0, "damage scale=0 should produce zero reward"

def test_curriculum_battle_rules_update_via_live_params():
    env = make_arena_env()
    original_cooldown = env.rules.cooldown_steps
    env.update_live_params({"battle_rules.cooldown_steps": original_cooldown + 2})
    assert env.rules.cooldown_steps == original_cooldown + 2
```

---

## YAML Config Fixes (Do alongside Feature 2)

Both shipped YAML configs have a dead key that fails validation:

**`src/rl_framework/configs/experiments/organisms_fight_arena.yaml`** and
**`src/rl_framework/configs/experiments/organisms_growth_competition.yaml`**:

Remove the `energy: 1.0` line from the `morphology:` section. The `config.py` validator rejects it with `ValueError` (it's not in `valid_morph_keys`).

Also: add `sensing_radius` to `BattleRules` config exposure in YAML so users can actually tune it (it's declared in `BattleRules` but absent from configs). Wire it into `_obs()` as a range gate: when `dist_to_opponent > sensing_radius`, zero out the opponent components of the observation:

```python
dist_to_opp = np.linalg.norm(opp_pos - self_pos)
if dist_to_opp > self.rules.sensing_radius:
    rel_x, rel_y, opp_health = 0.0, 0.0, 0.0  # opponent not visible
```

---

## Summary Table

| Feature | Files changed | Complexity | Depends on |
|---------|--------------|-----------|------------|
| 4. Egocentric obs | `arena_parallel.py` | Low | — |
| 6. Attack falloff | `arena_parallel.py` | Low | — |
| YAML fixes | 2 YAML files | Trivial | — |
| 2. Instrumentation | `arena_parallel.py`, `sb3_runner.py`, new callback | Low–Med | Feature 4 |
| 1. SelfPlayEnvWrapper | new file, `sb3_runner.py` | Low | Feature 2 |
| 3. Eval harness | new file, `cli/main.py` | Medium | Feature 1 |
| 5. Curriculum | new callback, `arena_parallel.py`, `sb3_runner.py`, YAMLs | Medium | Features 1, 2 |

## Acceptance Criteria

- `pytest -q --cov=src/rl_framework --cov-fail-under=60` passes (all new code has tests)
- `ruff check src tests scripts` passes
- `python -m rl_framework.cli.main train --config-name organisms_fight_arena` completes 5k steps without error and produces `rollout/` metrics in TensorBoard
- `python -m rl_framework.cli.main arena-eval --policy <path> --opponent random` completes and writes a JSON result
- `python -m rl_framework.cli.main train --config-name organisms_fight_arena` with `self_play.enabled: true` creates snapshot files in `outputs/*/checkpoints/league/`
- Both arena YAML configs pass `validate_experiment_config()` without error
