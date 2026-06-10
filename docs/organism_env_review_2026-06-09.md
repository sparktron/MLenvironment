# Organism Arena Environment — Code Review & Development Roadmap

**Date:** 2026-06-09
**Status update (2026-06-09):** B2, B5, B7, B8, B9, and B11 are fixed (with
regression tests).
**Status update (2026-06-10):** B1, B3, B4, and B10 are fixed as one batched
obs/dynamics contract change: spawn jitter (`sim.spawn_jitter`, default 0.1),
norm-clamped movement (`sim.move_speed`, default 0.05), and a normalized 8D
observation with a visibility flag. **Breaking:** arena checkpoints and league
snapshots trained before this change are incompatible (obs 7D → 8D) — retrain.
**Status update (2026-06-10, cont.):** B6 fixed — episode growth now scales
`max_health` in lockstep with `size`, and current health is rescaled by the
same factor so the health *fraction* is preserved (growth makes an organism
tankier as well as harder-hitting). **All bugs from this review are now
closed**; remaining items are the Part 2/3/4 efficiency and feature roadmap.
**Scope reviewed:**
- `src/rl_framework/envs/organisms/arena_parallel.py` (full)
- `src/rl_framework/training/self_play_env_wrapper.py` (full)
- `src/rl_framework/training/arena_eval.py` (full)
- `src/rl_framework/training/sb3_runner.py` (arena adapter + arena training path)
- `src/rl_framework/training/self_play_callback.py` (outline)

---

## Executive summary

The arena env is a clean, well-commented 2-agent PettingZoo implementation, and the
self-play plumbing (disk-backed league, vecnorm sidecars, SuperSuit patches) shows
hard-won correctness work. However, **two high-severity bugs make head-to-head
evaluation results untrustworthy today**:

1. The environment is **fully deterministic** — the seeded RNG is never used, spawn
   positions are fixed. With two deterministic policies, all `n_episodes` of an
   `arena-eval` run replay the *identical* episode, so reported win rates are only
   ever 0%, 50%, or 100% and the per-episode statistics are meaningless.
2. `arena_eval` **counts draws (simultaneous KO) as losses** for the policy under
   test, biasing win-rate comparisons.

The biggest efficiency problem is structural: arena training is hard-capped at
`num_envs == 1` (single process) on a 24-core machine. There is a clean escape
hatch — in self-play mode the wrapped env is effectively single-agent, so it can
bypass SuperSuit entirely and use the standard `SubprocVecEnv` path.

---

## Part 1 — Bugs

### P0 — break correctness of results

**B1. Env has zero stochasticity; eval episodes are identical replays**
`arena_parallel.py:36,96` — `self._rng` is created in `__init__` and re-created in
`reset(seed=...)` but **never used anywhere**. Spawn positions are fixed at
`(±0.6, 0)` (`_spawn_agent`, line 79). Consequences:
- `run_arena_eval` seeds each episode (`base_seed + i`) to no effect. Two
  deterministic checkpoints produce N identical episodes per slot — the win-rate
  denominators are fiction and the compute is wasted.
- During training, episode diversity comes only from PPO's stochastic action
  sampling; eval-time behavior cliffs are invisible.

*Fix:* randomize spawn (position jitter and/or angle) from `self._rng` in
`_spawn_agent`, and/or add small action/observation noise. Optionally add a
`deterministic: bool` flag to `run_arena_eval` to allow stochastic policy sampling.
Until then, `n_episodes > 1` against a fixed opponent is misleading.

**B2. Draws counted as losses in head-to-head eval**
`arena_eval.py:48-53` — `_play_episode` classifies `outcome == "draw"`
(simultaneous KO, `winner: None`) as a **loss** because the `elif winner ==
policy_slot` check falls through to `"loss"`. Draws inflate `opponent_win_rate`.
*Fix:* add an explicit draw branch and a `draw_rate` field in the result dict.

### P1 — wrong behavior, currently survivable

**B3. Diagonal movement is 41% faster than axis-aligned**
`arena_parallel.py:194` — `move = action[:2] * 0.05` clips per-component, so
`(1,1)` moves at `0.05·√2 ≈ 0.0707` while `(1,0)` moves at `0.05`. Policies will
exploit this (diagonal kiting). *Fix:* clamp the move vector's norm to the speed
limit instead of per-component scaling.

**B4. "Out of sensing range" is encoded identically to "opponent at zero health, on top of me"**
`arena_parallel.py:126-128` — when the opponent is beyond `sensing_radius`, the obs
zeroes `rel_x, rel_y, opp_health`. The policy cannot distinguish this from a real
adjacent near-dead opponent. *Fix:* add an 8th obs dimension `opponent_visible ∈
{0,1}`. (Note also: default `sensing_radius=2.0` covers almost the whole default
arena — max possible separation is `2√2·bounds ≈ 2.83` — so the gate almost never
fires with default config; tune defaults when fixing.)

**B5. Unknown `battle_rules` keys in YAML are silently dropped**
`arena_parallel.py:38-44` — the dict-comprehension filter
(`if k in BattleRules.__annotations__`) means a typo like `dammage: 0.1` silently
runs with defaults. *Fix:* warn (or raise) on unrecognized keys.

**B6. Episode growth scales damage but not max health** — ✅ FIXED (2026-06-10)
`_current_size` grows `size` each step (up to 2.0); damage scales with attacker
`size` but `max_health` was frozen at spawn — an undocumented asymmetry.
*Resolution:* `max_health` now tracks `size` each step via the shared
`_max_health_for_size` helper, and current health is rescaled by the same
factor so the health fraction is preserved across growth. Growth now buffs
defense and offense together. `episode_growth_scale`/`health` schema docs
updated to say so.

### P2 — robustness / hygiene

**B7. `LeagueSampler._league_files` crashes on non-numeric snapshot names**
`self_play_env_wrapper.py:136-139` — `int(p.stem.rsplit("_", 1)[-1])` raises
`ValueError` if anything else matching `selfplay_*.zip` (e.g. a hand-copied
`selfplay_best.zip`) lands in the league dir, killing training at the next reset.
*Fix:* filter to numeric stems, skip others with a warning.

**B8. Step-after-done returns a spurious timeout outcome**
`arena_parallel.py` `step()` — if called after `agents == []`, `truncations == {}`
and `all({}) is True`, so the code takes the timeout branch and rebuilds an
"episode over" state. Harmless under SuperSuit's auto-reset, but a latent trap for
any direct user of the env. *Fix:* guard `step()` with `if not self.agents: raise`
(or no-op return), and use `active_agents and all(...)`.

**B9. `matplotlib.use("TkAgg")` on every human-mode `render()` call**
`arena_parallel.py:286` — calling `use()` after pyplot is loaded is ignored or
warns; it also breaks headless boxes if `render_mode="human"` is requested without
a display. *Fix:* select backend once at first render, wrap in try/except with a
clear error, never force a backend for `rgb_array` (Agg works headless).

**B10. Obs feature scales are wildly mismatched**
Velocity components max ±0.07/step; health ~1.0; rel-pos up to ±2·bounds; cooldown
raw int 0–3. `VecNormalize` rescues training (it's on by default), but any run with
`normalize_observations: false`, and the raw-obs path in `FrozenPolicy` fallback
("if no sidecar, use raw"), will see badly conditioned inputs. *Fix:* normalize in
the env (health/max_health, cooldown/cooldown_steps, rel/bounds).

**B11. `SelfPlayEnvWrapper` reaches into env privates**
`self_play_env_wrapper.py:240` — `self.env._obs(self.FROZEN_AGENT)` breaks if any
wrapper is inserted between it and the arena env (see the existing memory note on
chain-wrapper rules). *Fix:* expose a public `observe(agent)` on the arena env.

---

## Part 2 — Efficiency issues

**E1. Arena training is capped at 1 env / 1 core (the dominant cost).**
`sb3_runner.py:288` deliberately raises on `num_envs > 1` because SuperSuit's
subprocess mode breaks `env_method` (annealing/curriculum become silent no-ops) and
is unstable in SuperSuit 3.10. The machine has 24 physical cores; arena throughput
is therefore ~1/20th of what the walker path achieves. See roadmap R2 for the fix —
this is the highest-value engineering item in the file set.

**E2. League sampling does disk I/O + possible `PPO.load` every episode reset.**
`LeagueSampler.sample()` globs the directory per reset and pays a full model load
(~50–200 ms) on each cache miss. Fine at current scale; will matter at higher
episode rates after E1 is fixed. Consider caching the file list with an mtime check
and pre-warming the newest snapshot.

**E3. Matplotlib rendering is slow for video generation** (~tens of ms/frame,
full patch rebuild via `ax.clear()` each frame). Adequate for occasional replays;
a tiny pygame/PIL rasterizer would render 100–1000× faster if arena videos become
routine.

---

## Part 3 — Development roadmap

### Phase 1 — Trustworthy evaluation (small, do first)
| ID | Item | Effort |
|----|------|--------|
| R1a | Fix draw-as-loss in `arena_eval` (+ `draw_rate` output) — bug B2 | ~½ hr |
| R1b | Spawn randomization from `self._rng` (jitter radius in config); makes seeds meaningful — bug B1 | ~1–2 hr |
| R1c | `deterministic` flag + per-episode result list in `run_arena_eval` (enables confidence intervals) | ~1 hr |
| R1d | Movement-norm clamp (B3), visibility flag in obs (B4), config-key validation (B5) | ~2 hr |

Note: R1b and R1d change the obs/dynamics contract — old checkpoints will not be
comparable. Batch them into one breaking change and retrain once.

### Phase 2 — Throughput (the big win)
| ID | Item | Effort |
|----|------|--------|
| R2 | **Parallel self-play training without SuperSuit.** When `self_play.enabled`, `SelfPlayEnvWrapper` already exposes exactly one agent — wrap it in a thin `ParallelEnv → gymnasium.Env` adapter and feed the standard `SubprocVecEnv` path (`num_envs: 24`, `Monitor`, native `env_method` for annealing/curriculum). Removes `_ArenaVecEnvAdapter`, the num_envs guard, and the SuperSuit 3.10 fragility for the self-play path; shared-policy mode can keep SuperSuit. Disk-backed league sampling already survives subprocesses by design. | ~1 day |
| R2b | Vary `LeagueSampler` seed per worker rank so parallel envs don't sample identical opponent sequences | trivial, part of R2 |
| R2c | League file-list caching / snapshot pre-warm (E2) | ~1 hr |

R2 also restores `rollout/ep_rew_mean` via `Monitor`, which the arena path
currently lacks (ArenaMetricsCallback partially compensates).

### Phase 3 — Usability & tooling
| ID | Item | Effort |
|----|------|--------|
| R3a | **Tournament CLI** (`arena-tournament`): round-robin over a checkpoint directory, Elo/Bradley-Terry ratings, JSON + markdown table output. Builds directly on `run_arena_eval`. | ~½ day |
| R3b | **GUI league dashboard**: league size, snapshot ages, latest win-rate vs league, link to render replays (the GUI already has an arena schema at `gui/app.py:371`) | ~1 day |
| R3c | Headless arena video rendering wired into `render-replay` (force Agg backend, B9 fix) | ~2 hr |
| R3d | Config schema docs for `battle_rules` / `morphology` (incl. growth-vs-health semantics, B6) | ~1 hr |
| R3e | Public `observe(agent)` API on the env; migrate `SelfPlayEnvWrapper` off `_obs` (B11) | ~½ hr |

### Phase 4 — Environment richness (design work, do after 1–3)
- **Body collision** — agents currently overlap freely; contact pushes would make
  positioning meaningful.
- **Energy/food mechanics** — attacks cost energy, food pellets restore it; gives
  the "organism" framing teeth and creates non-combat strategies.
- **N-agent arenas** (the env hardcodes `agent_0`/`agent_1` opponent lookups in
  `_obs` and `step`; generalizing the opponent indexing is a prerequisite).
- **Morphology co-evolution** — let `morph-search` drive both competitors and
  score by tournament Elo instead of single-opponent win rate (depends on R3a).
- **Speed/size tradeoff** — move the hardcoded `0.05` speed into config and scale
  it inversely with `size` so growth is a real strategic choice.

### Testing gaps to close alongside
- A regression test that two eval episodes with different seeds **differ** (locks in R1b).
- A draw-classification test for `arena_eval` (locks in R1a).
- A test that league dirs containing non-numeric `selfplay_*.zip` files don't crash sampling (B7).
- A `step()`-after-done contract test (B8).

---

## What's in good shape (keep as-is)

- PettingZoo Parallel API compliance: consistent dict keys across all five return
  values, correct terminal/truncation semantics, egocentric slot-symmetric obs.
- The disk-backed league design and the vecnorm-sidecar pairing for frozen
  opponents — both solve real cloudpickle/normalization traps and are well
  documented in-line.
- Role-swapped, seed-paired eval design in `run_arena_eval` (it will become fully
  meaningful once B1 is fixed).
- `_ArenaVecEnvAdapter`'s uint8-dones and `seed()` patches are correct and well
  explained.
