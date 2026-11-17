# ADR-003: Split the walker env; declare config and live parameters as data

**Status:** Accepted · **Date:** 2026-11-16

## Context

A structure audit found the core / physics / games split sound but the env
classes carrying the debt: `WalkerBulletEnv` was a single ~1,400-line class
(terrain, robot construction, contact sensing, perturbations, reset/step); its
raw config dict was read with ~60 `.get()` calls while the valid keys were
restated in `validation.py`, the GUI schema, the reward/termination dataclasses
and hand-written `LIVE_PARAMS` / `_TERRAIN_*_FIELDS` sets; both envs hand-wrote
a per-key `update_live_params`; the 35/45-dim observation layout existed only
as the order of `extend` calls and prose; `arena_parallel.py` mixed rules with
matplotlib rendering.

## Decision

- `physics/locomotion/` env modules: `terrain`, `robot`, `sensing`
  (per-step `FootSnapshot`), `perturbations`, `observation`, `config`.
  `walker_bullet.py` only orchestrates `reset` and `step`.
- `config.py` parses raw dicts once into typed dataclasses. Fields that can
  change mid-run are declared with `live_field(...)`; `LIVE_PARAMS` is derived
  from those declarations (and from the reward/termination dataclasses), so it
  cannot drift.
- `core/config/live_params.py` is the shared mechanism: a `LiveParam` declares
  its cast and whether it applies `immediate`ly or `reset` (terrain). Both
  envs' `update_live_params` dispatch through it.
- `ObservationLayout` is the named-slice description of the observation, with
  `signature()` / `describe_mismatch()` for checkpoint-compatibility checks;
  `STEP_INFO_KEYS` is the `info` schema.
- Arena: `arena_render.py` owns the matplotlib figure; `step` is composed of
  `_phase_movement`, `_phase_combat`, `_phase_knockouts`, `_phase_outcome`.
- Analysis tooling (`quality_study`, `walker_diagnostics`) moved to
  `physics/locomotion/analysis/`. The benchmark implementation moved into the
  package (`rl_framework/benchmark_device_matrix.py`); `scripts/` holds a shim,
  so the package no longer imports from `scripts/`.

## Behaviour preservation

No behaviour change was intended. The accepted live-key sets are identical to
the old ones (checked by diffing the old and new frozensets), and step-for-step
rollouts of the old and new walker (flat, v2 + domain randomisation + latency,
terrain mix, limb collision, with a mid-run live update) and arena (3 agents,
live update) matched exactly, observations, rewards, `info` and `cfg`.
The one visible difference: the walker no longer writes an empty
`reset_randomization: {}` into the caller's `cfg` at first reset. Live-update
errors for a malformed value now name the key (`terrain.push_recovery: must be
a mapping`).

## Not done / left to the owner

- `validation.py` and `gui_schema.py` still restate some key sets; only the
  live-key set and reward/termination keys are derived from the types. Folding
  the rest in changes validation behaviour and deserves its own change.
  ADR-005 derives the config key sets from the types and pins the GUI defaults
  with a test; the GUI defaults that differ from the dataclasses are recorded
  there, not resolved.
- The root `rl_framework/__init__.py` path shim was unchanged here; ADR-004
  removed it.
