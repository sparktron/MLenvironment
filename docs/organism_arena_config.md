# Organism Arena — Configuration Reference

Config reference for `type: organism_arena_parallel` experiments. Defaults below
are the values the env falls back to when a key is omitted; see
`src/rl_framework/envs/organisms/arena_parallel.py` for the source of truth and
`configs/experiments/organisms_fight_arena.yaml` for a worked example.

The arena is an N-agent PettingZoo `ParallelEnv` (`agent_0` … `agent_{N-1}`,
default N=2). Spaces are slot-symmetric, so one shared policy can fill any slot.

## `environment.num_agents`

| Key | Default | Meaning |
|-----|--------:|---------|
| `num_agents` | `2` | Number of organisms. `2` is a duel; `>2` is a free-for-all. Must be ≥ 2. Agents spawn evenly on a circle (the 2-agent case reduces exactly to the legacy `±0.6` layout). |

**N-agent mechanics.** The observation stays a fixed 8-D and describes the
*nearest living opponent*, so it is identical at N=2 and old checkpoints remain
valid. An attack hits the attacker's nearest living opponent within
`attack_range` (single target). All attacks in a step resolve against the
agents alive at the start of that step, so mutual knockouts are draws, not
order-dependent wins. A knocked-out agent takes a one-time `−1` and becomes an
**inert spectator** (cannot move, attack, or be targeted; earns 0 reward) — the
agent set stays a constant size so SuperSuit/VecNormalize see a fixed
population. The episode ends (last-organism-standing) when ≤ 1 agent remains
(the survivor gets `+1`) or at `max_steps`.

**Tooling note.** `arena-eval` and `arena-tournament` are head-to-head (pairwise)
and require `num_agents: 2`. Self-play (`SelfPlayEnvWrapper`) drives every
non-live slot from one sampled frozen past-self, so N-agent self-play works; the
shared-policy SuperSuit path also supports N > 2. In N-agent self-play, the live
policy's SB3 episode ends as soon as it is knocked out, even while the remaining
arena agents continue their underlying match.

## Observation (8-D, `float32`, pre-scaled to ~[-1, 1])

| Index | Component | Meaning |
|------:|-----------|---------|
| 0–1 | `self_vel_x/y` | Displacement since last step, in units of `move_speed`. |
| 2 | `health_frac` | Own health ÷ own max health. |
| 3–4 | `rel_opp_x/y` | Opponent position relative to self, in units of the arena diameter (`2·arena_half_extent`). Zeroed when out of `sensing_radius`. |
| 5 | `opp_health_frac` | Opponent health ÷ its max health. Zeroed when out of range. |
| 6 | `cooldown_frac` | Own attack cooldown ÷ `cooldown_steps`. |
| 7 | `opp_visible` | `1.0` if the opponent is within `sensing_radius`, else `0.0`. Disambiguates "out of range" from "adjacent, near-zero health". |

## Action (3-D, `float32`, each in [-1, 1])

`[move_x, move_y, attack_trigger]`. The move vector is **norm-clamped** to
`move_speed` (so diagonal movement is not faster than axis-aligned). `attack`
fires when `attack_trigger > 0.5` and the attacker is off cooldown.

## `environment.sim`

| Key | Default | Meaning |
|-----|--------:|---------|
| `arena_half_extent` | `1.0` | Half-width of the square arena; positions are clipped to `[-extent, +extent]`. |
| `move_speed` | `0.05` | Max distance (arena units) an agent moves per step. |
| `spawn_jitter` | `0.1` | Half-width of uniform spawn-position jitter applied to the base spawn (`±0.6, 0`). **Set `0` for deterministic fixed spawns** — but note that with two deterministic policies a jitter of 0 makes every episode an identical replay, so keep it non-zero for meaningful multi-episode eval. |

## `environment.morphology`

| Key | Default | Meaning |
|-----|--------:|---------|
| `base_size` | `1.0` | Starting organism size, clipped to `[0.5, 2.0]`. Size scales rendered radius, **damage dealt**, and **max health**. |
| `health` | `1.0` | Base health pool. Actual max health = `health · size`. |
| `episode_growth_scale` | `0.0` | Per-step size growth: `size = clip(base_size + scale·step, 0.5, 2.0)`. Growth raises **both** damage output and max health in lockstep; current health is rescaled to preserve the health *fraction*, so a growing organism gets tankier and harder-hitting together (it does not heal in fractional terms). `0.0` disables growth. |

## `environment.battle_rules`

Unknown keys here raise a `UserWarning` and are ignored (typo guard).

| Key | Default | Meaning |
|-----|--------:|---------|
| `damage` | `0.05` | Base health removed per landed hit (before size and falloff scaling). |
| `attack_range` | `0.2` | Distance at/under which an attack can connect. |
| `cooldown_steps` | `3` | Steps an agent must wait between attacks. |
| `sensing_radius` | `2.0` | Range within which the opponent is visible in the observation. Note the default arena's max separation is `2√2·extent ≈ 2.83`, so a radius of `2.0` leaves only the far corners blind — lower it for partial observability. |
| `max_steps` | `400` | Episode truncates (timeout, no winner) at this step count. |
| `win_health_threshold` | `0.0` | An agent is knocked out when its health drops to/below this. |
| `attack_falloff` | `"linear"` | `"linear"`: full damage at point-blank, scaling to zero at `attack_range`. `"binary"`: full damage inside range, none beyond (hard cliff). |

Effective damage per hit = `damage · falloff(distance) · attacker_size`. Health
always takes full damage so combat resolves; only the *dense per-hit reward* is
scaled by reward annealing (below).

## Reward

- **Dense:** the attacker gains `+damage_dealt`, the defender loses the same,
  multiplied by the live `damage_scale` (annealed toward 0 — see below).
- **Terminal:** the winner gains `+1.0`, the loser `−1.0` on a knockout.

## Arena-relevant training sections

These live under the top-level config (siblings of `environment`), not inside
`environment`.

### `self_play`
| Key | Default | Meaning |
|-----|--------:|---------|
| `enabled` | `false` | Route `agent_1` through a frozen past-self sampled from an on-disk league. **Enables the parallel native-vec-env training path** (so `training.num_envs > 1` is allowed). With it `false`, the arena trains shared-policy via SuperSuit and is capped at `num_envs == 1`. |
| `snapshot_freq` | `5000` | Save a frozen league snapshot every N timesteps. |
| `max_league_size` | `10` | Cap on retained snapshots (oldest pruned). |
| `sampling_mode` | `"uniform"` | `"uniform"` or `"recent_bias"` (weights newer snapshots by `recent_bias_alpha`). |
| `recent_bias_alpha` | `1.0` | Exponent on the recency weighting when `sampling_mode: recent_bias`. |

### `reward_annealing`
| Key | Default | Meaning |
|-----|--------:|---------|
| `enabled` | `false` | Linearly anneal the dense per-hit reward scale from 1.0 to 0.0 so the terminal win/loss signal eventually dominates. |
| `anneal_steps` | `500000` | Timesteps over which the dense reward decays to zero. |

### `curriculum`
Win-rate-gated difficulty ramp. For the arena, gate on `arena/agent_0_win_rate`
(logged by `ArenaMetricsCallback`) and set `warmup_steps >= self_play.snapshot_freq`
so level-ups don't trigger against the random-action opponent before the league
fills. `level_params` apply `battle_rules.<field>` overrides per level. See
`organisms_fight_arena.yaml` for a complete example.

## Evaluating and comparing checkpoints

- `arena-eval --policy A --opponent B` — head-to-head win/draw/timeout rates.
- `arena-tournament --checkpoints DIR --include-random` — round-robin Elo
  ranking over a pool (e.g. a league directory).
- `render-replay --model-path A --replay-opponent B` — GIF of a matchup
  (`agent_0 = A`, `agent_1 = B`; omit `--replay-opponent` for a shared-policy
  replay).
