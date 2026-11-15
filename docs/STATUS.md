# Current Status

**Updated:** 2026-11-14 · **Branch:** `master` · **Restructure baseline:** `ae6f09b`

## Objective

A modular reinforcement-learning framework for training and evaluating agents
on PyBullet physics environments: PPO via Stable-Baselines3, two environment
families (single-agent bipedal walker, 2-agent organism arena), driven by YAML
experiment configs through one CLI, with a Flask GUI for experiment management.

## Current state

- Single CLI entry point with nine subcommands: `train`, `eval`, `arena-eval`,
  `arena-tournament`, `sweep`, `multi-seed`, `render-replay`, `morph-search`,
  `gui`.
- Walker env on PD position control by default; 240 Hz physics with
  `frame_skip=4` → 60 Hz control; contact-based termination.
- **Per-episode terrain sampling** via `terrain.mix` (added 2026-08-11); the
  drawn terrain is reported in `info["terrain_preset"]`.
- **Terrain rebuilt 2026-11-14** after an audit found three defects (see the
  ledger entry of that date): uneven is a smooth heightfield in [0, height],
  obstacles tile the track, both span x = 0.6–25.6 m and are redrawn per
  episode from pooled bodies; the push acts at the torso COM for
  `duration_s` (default 0.1 s). Transfer-suite numbers before this date are
  not comparable with later ones.
- Arena with self-play (parallelises across cores) and shared-policy
  (single-process only) paths, plus Bradley-Terry Elo tournament ranking.
- Flask GUI with a config wizard.
- Roadmap Priority 1 (runtime and GUI performance) and Priority 2 (evaluation
  and operations) are **complete** per `docs/open_items_todo.md` (2026-08-06).

## Active work

The [core / physics / games restructure](plans/2026-11-12-core-physics-games-restructure.md)
has its Prompt 0 regression tests for environment rollouts, legacy checkpoints,
and callback order. Prompts 1 and 2 are complete: task registry hooks now
provide configuration validation, GUI schemas, evaluation, training environment
construction, and ordered task callbacks. Prompt 3 splits the quality-study
runner into task-contributed walker and arena studies. Prompt 4 moved the
modules into core, physics, and games and enforces their import boundaries.
Prompt 5 makes live updates strict, allows terrain changes at reset, and
updates gait timing when control timing changes. Prompt 6 is next.

Mixed-terrain generalisation study. Two negative results were recorded, then
**reopened** on 2026-11-14 because both ran on defective terrain:

| Version | Date | Result |
|---|---|---|
| v11 uneven-terrain continuation from the v9 flat result | 2026-08-10 | Specialisation, not generalisation — **reopened, pending re-evaluation** |
| v12 mixed-terrain | 2026-08-13 | Hypothesis refuted — **reopened, pending re-evaluation** |

Baseline: three-seed v9 — 2/3 seeds pass gate v3, all three learned to run
(2026-08-05).

## Next

Priority 3 (learning quality and features) in `docs/open_items_todo.md` — the
first two priorities are closed, so this is the live section. Specific items
were **not** enumerated here; read the roadmap rather than trusting a summary.

First, re-run the transfer suite (now 100 episodes per terrain, with fall-rate
intervals) and a `gait.clearance_point: sole` comparison on the v9/v11/v12
checkpoints, or retrain them if they are gone. Only then decide whether the
v11/v12 conclusions stand. The question of what produces cross-terrain
generalisation is open again until that re-run exists.

Research decisions the 2026-11-14 review raised and left to the owner: whether
lateral push recovery is in scope (the robot has no hip roll), a multi-seed
capacity comparison now that `training.net_arch` exists, a frozen gate with
held-out seeds, and a phase-indexed swing-height reference.

## Blockers

None identified from repository evidence.

## Known problems

- Obstacle results stay optimistic unless `sim.limb_collision: true`; it is
  off by default so existing checkpoints keep their physics.
- Contact caching cut a walker step only ~5% (491 → 465 µs measured); most of
  the remaining overhead is outside contact queries and was not profiled.
- Shared-policy arena (`self_play.enabled: false`) is single-process only; a
  guard rejects `num_envs > 1`.
- `SubprocVecEnv` cannot be smoke-tested from a stdin heredoc.
- A root-level `rl_framework/` directory contains a single tracked
  `__init__.py`, distinct from the real package at `src/rl_framework/`. Its
  purpose is **unverified** — possibly a stale import shim.

## Validation state

| Check | Status |
|---|---|
| `pytest -q --cov=src/rl_framework --cov-fail-under=60` | Passed locally on 2026-11-14 after the review fixes: 554 tests, 84% coverage. |
| `ruff check src tests scripts` | Passed locally. |
| `python scripts/check_repo_policy.py` | Passed locally. |
| `mypy src --ignore-missing-imports` | Clean; **blocking** in CI since 2026-11-14. |
| Learning results | Multi-seed evidence recorded in `docs/walker_learning_history.md` through v12. |

Full evidence breakdown, including why "tests pass" is not "the hypothesis
held": [`VALIDATION.md`](VALIDATION.md).

## Unverified

- The purpose of the root-level `rl_framework/__init__.py`.
- Whether the 75 directories under `outputs/` correspond to the runs described
  in the learning history. `outputs/` is gitignored, so this cannot be
  reconstructed from the repository.
- Whether the duplicated skills under `.claude/skills/` and `.agents/skills/`
  (eval-policy, render-policy, repo-check) are intentionally mirrored or have
  drifted. They were byte-identical on 2026-08-17.

## Recent decisions

| Date | Decision | Evidence |
|---|---|---|
| 2026-11-14 | Terrain and push rebuilt; v11/v12 conclusions reopened; config typos and `control: null` now fail | `reviews/code_review_2026-11-14.md`, ledger |
| 2026-08-13 | v12 mixed-terrain hypothesis refuted; recorded rather than discarded | commit `aecf61d`, `docs/walker_learning_history.md` |
| 2026-08-11 | Terrain variation via per-episode `terrain.mix`, not curriculum | commit, `AGENTS.md` |
| 2026-08-10 | v11 result characterised as specialisation, not generalisation | `docs/walker_learning_history.md` |
| 2026-08-06 | Historical review findings folded into the completed summary so finished work stops reading as pending | `docs/open_items_todo.md` |

`docs/walker_learning_history.md` is append-only and is the canonical ledger
for *experiment* results, including failed hypotheses. *Architectural* decisions
now live in [`decisions/`](decisions/) — ADR-001 records terrain sampling over
curriculum.

## Deep context

| Topic | Document |
|---|---|
| Agent instructions | [`AGENTS.md`](../AGENTS.md) |
| Validation evidence | [`VALIDATION.md`](VALIDATION.md) |
| Decision records | [`decisions/`](decisions/) |
| Active roadmap | [`open_items_todo.md`](open_items_todo.md) |
| Walker decision ledger (canonical) | [`walker_learning_history.md`](walker_learning_history.md) |
| Arena config reference | [`organism_arena_config.md`](organism_arena_config.md) |
| Learning quality studies | [`learning_quality_studies.md`](learning_quality_studies.md) |
| Training presets | [`training_presets.md`](training_presets.md) |
| Env review | [`organism_env_review_2026-06-09.md`](organism_env_review_2026-06-09.md) |
| Visualization | [`../VISUALIZATION_GUIDE.md`](../VISUALIZATION_GUIDE.md) |
