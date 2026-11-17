# ADR-005: Derive config key sets and GUI defaults from the typed config

**Status:** Accepted · **Date:** 2026-11-17

## Context

ADR-003 moved the walker and arena configs into typed dataclasses and derived
the live-parameter set from them. Three places still restated config knowledge
by hand:

- `physics/locomotion/validation.py` and `games/arena/validation.py` listed the
  valid keys for terrain specs, push recovery, morphology, resources, presets,
  clearance points and observation versions.
- `physics/locomotion/gui_schema.py` and `games/arena/gui_schema.py` gave each
  field a default, which could drift from the dataclass default the env uses.

A key added to a dataclass but not to the validator was rejected by the wizard
or silently accepted. Nothing caught the drift.

## Decision

- Key sets come from the types. Terrain spec keys and push-recovery keys are
  read off `TERRAIN_LIVE_PARAMS`. Reward and termination keys come from their
  dataclasses. Resource keys come from `ResourceRules`. Morphology keys come
  from `MORPHOLOGY_KEYS`. Presets, clearance points, observation versions and
  food placements are named constants next to the code that branches on them.
- Arena rule types (`BattleRules`, `ResourceRules`, `RewardRules`) moved from
  `arena_parallel.py` to `games/arena/config.py`. Validation can then read the
  defaults without importing the renderer, which pulls in matplotlib.
- Rules a type cannot express stay written out in `validation.py`: ranges,
  signs, and cross-field checks. Their messages are unchanged.
- `tests/test_config_contract.py` pins the behaviour. It has a table of
  accepted and rejected configs, a snapshot of the live-key sets and GUI schema,
  a check that every GUI leaf names a real dataclass field, and a check that
  every GUI default equals its dataclass default unless it is listed in
  `KNOWN_GUI_DEFAULT_DIVERGENCES`.

## Behaviour changes

**None intended.** Every accepted and rejected config in the characterisation
table behaves as before the change, with the same messages. The GUI schema
snapshot is unchanged. An old-vs-new rollout comparison on the walker and arena
envs (fixed seeds, 300 steps, observations, rewards, terminations and truncations
hashed per step) is identical, at commit `4cbd89a` against this change.

The only code-location change is the relocation of the arena rule dataclasses
(see above). `rl_framework.games.arena.arena_parallel` still re-exports
`LIVE_PARAMS`.

## GUI default divergences (not changed; owner decision)

The wizard writes these values into new configs. The env uses the dataclass
value when a config omits the key. Changing either side changes behaviour, so
this change keeps every value and records the divergence. The test fails if a
new one appears.

| Key | Wizard (GUI) | Dataclass (env fallback) |
|---|---|---|
| `walker_bullet.sim.friction` | 0.9 | 0.8 |
| `walker_bullet.sim.mass` | 28.0 | 3.0 |
| `walker_bullet.reward.alive_bonus` | 5.0 | 0.25 |
| `walker_bullet.reward.forward_velocity_weight` | 1.5 | 2.0 |
| `walker_bullet.termination.max_steps` | 800 | 1000 |
| `walker_bullet.reset_randomization.yaw_noise` | 0.08 | 0.1 |
| `walker_bullet.domain_randomization.friction_range` | [0.9, 1.1] | [1.0, 1.0] |
| `walker_bullet.domain_randomization.mass_scale_range` | [0.95, 1.05] | [1.0, 1.0] |
| `organism_arena_parallel.battle_rules.damage` | 0.06 | 0.05 |

`sim.mass` is the one to look at first. `AGENTS.md` says the torso is 28 kg,
which matches the wizard. The env falls back to 3.0 kg when the key is absent.

`walker_bullet.sim.max_force` is a legacy alias for `max_torque`. The wizard
still writes it. The test maps it through `GUI_KEY_ALIASES`.

## Flagged, not applied

These would reject configs that load today, so they are not decided here:

1. `sim.control.mode` accepts any string. An unknown value such as `"bogus"`
   loads, and `dynamics.py` treats anything other than `"pd"` as torque control.
2. Unknown `sim.*` and `gait.*` keys are not rejected by validation.
3. Unknown `battle_rules` keys are not rejected. The env only warns.
4. `ObservationConfig.from_dict` does not reject an unknown `version` at
   construction. Validation does.

## Follow-up

- `SimConfig`, `ObservationConfig` and `GaitConfig.from_dict` repeat their
  default literals instead of reading the dataclass defaults. Changing them to
  read the defaults is a small change, but it would move the GUI-divergence
  table, so it belongs in its own change.

## Related

- [ADR-003](003-env-structure-and-declarative-config.md): the typed config and
  derived live-parameter set this builds on.
