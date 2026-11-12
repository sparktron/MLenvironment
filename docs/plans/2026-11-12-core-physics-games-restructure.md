# Restructure into `core` / `physics` / `games` — agent prompt series

**Written:** 2026-11-12 against commit `459511d`.
**Status:** not started.

This file is a sequence of eight prompts. Each one is handed to a fresh coding
agent, in order, one at a time. Every prompt assumes the agent has read the two
shared sections below (**Ground rules** and **Target design**) — each prompt
says so in its first line. Do not start prompt N+1 until prompt N's "Done when"
list is fully met and pushed.

Why this is happening: the walker and the arena share no simulation code, but
the trainer, validator, evaluator and GUI contain `if env_type == ...` branches
for both. Adding a third task means editing about seven modules. The goal is
one repo with a task-agnostic `core` and two task packages that plug into it,
so a new real-time game or a new physics task is one new sub-package plus a
YAML.

---

## Ground rules (apply to every prompt)

**Environment.** `AGENTS.md` says a venv lives at `.venv/`. On 2026-11-12 it
did not exist on this machine, and neither did `outputs/`. Prompt 0 creates the
venv. Every later prompt starts with `source .venv/bin/activate`; if that fails,
stop and say so rather than falling back to system Python.

**The gate.** A prompt is not finished until all of these pass, and the agent
has pasted the tail of each output into its report:

```
pytest -q --cov=src/rl_framework --cov-fail-under=60
ruff check src tests scripts
python scripts/check_repo_policy.py
```

**Behaviour must not change in prompts 1–4.** Those prompts move and re-wire
code; they fix nothing. Prompt 0 installs golden-rollout and legacy-checkpoint
tests as the proof. If one of those tests fails, the refactor is wrong. Never
regenerate a golden fixture, loosen a tolerance, or skip/xfail a test to get
green — stop and report what diverged. Bugs you notice along the way go in the
report, not in the diff; prompts 5 and 6 are where behaviour changes.

**Moves are moves.** Use `git mv` so history follows the file. Do not reformat,
rename symbols, or "tidy" a file in the same commit that moves it. Do not leave
compatibility shims at old import paths — update the importers (tests included)
in the same commit.

**Commits.** Follow `AGENTS.md` § Workflow: direct commits to `master`, logical
groups, push after the gate passes. Each commit must leave the gate green on
its own, so `master` is never broken between prompts.

**Smoke tests.** These are "big changes" under `AGENTS.md`, so the gate alone
is not enough. Where a prompt asks for a training smoke test with
`num_envs > 1`, write the script to a temp `.py` file and run it with
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python <file>` —
`SubprocVecEnv` cannot be driven from a stdin heredoc.

**Out of scope for the whole series.** Do not start any of these, even if they
look adjacent: decomposing `WalkerBulletEnv` into embodiment/scene/observation
terms; typed (dataclass/Pydantic) configs; YAML inheritance; removing SuperSuit
or the shared-policy arena path; changing observation or action layouts;
editing experiment YAMLs' hyperparameters; a GPU simulator backend.
`docs/walker_learning_history.md` is an append-only experiment ledger — this
work is not an experiment, so do not write to it.

**Reporting.** End every prompt with: what changed (by commit), the gate
output tails, the smoke-test output, anything you were asked to do and did not,
and any bug or surprise you found but left alone.

---

## Target design (the end state all prompts build toward)

### Layout

```
src/rl_framework/
  plugins.py        the only place the task packages are named (strings only)
  core/             task-agnostic. Imports nothing from physics or games.
    registry.py       TaskSpec, register_task, get_task, make_env, load_plugins
    config/           load + generic validation, config_merge
    training/         runner (was sb3_runner), eval_runner, evaluation_workers,
                      multi_seed_runner, sweep, rollout_metrics, study runner
    training/callbacks/  curriculum, live_tuning, frame_capture, checkpoint,
                      stop-on-event and the other generic callbacks
    utils/            checkpoint, logging_utils, reproducibility, run_registry
  physics/          PyBullet tasks. May import core. Never imports games.
    locomotion/       walker_bullet, dynamics, gait, rewards, terminations
    ...               walker diagnostics, velocity-target ramp, walker metrics,
                      walker validation, walker GUI schema, walker studies
  games/            real-time game tasks. May import core. Never imports physics.
    arena/            arena_parallel
    ...               self-play wrapper + callback, reward annealing, arena
                      eval, tournament, morphology search (+ evolution/),
                      arena metrics, SuperSuit adapter, arena validation,
                      arena GUI schema, arena studies
  cli/              application layer. May import core, physics and games.
  gui/              application layer. May import core, physics and games.
  configs/          unchanged location (data, not code)
```

`cli/` and `gui/` deliberately stay where they are. `python -m
rl_framework.cli.main ...` and the default `--config-dir` must keep working
unchanged, so the README, Dockerfile and skills do not churn.

### Import rules (enforced by lint in prompt 4)

| Package | May import from `rl_framework` |
|---|---|
| `core` | `core` only |
| `physics` | `core`, `physics` |
| `games` | `core`, `games` |
| `cli`, `gui` | anything |
| `plugins.py` | nothing (it holds module names as strings) |

Two extra rules for `core`, because an import ban alone is easy to satisfy
dishonestly: no string literal `"walker_bullet"` or
`"organism_arena_parallel"`, and no string literal beginning
`"rl_framework.physics"` or `"rl_framework.games"`.

### Registry

A task package registers a `TaskSpec` at import time. Core looks tasks up by
`environment.type` and asks the spec for everything task-specific. Suggested
shape — adjust field names if the code argues for it, but keep the idea that
core holds no task knowledge:

```python
@dataclass(frozen=True)
class TaskSpec:
    name: str                      # the environment.type string
    make_env: Callable             # module-level, picklable; cfg -> env
    validate_config: Callable | None      # task-specific config checks
    gui_schema: Callable | None           # wizard schema for this task
    build_train_vec_env: Callable | None  # None -> core default path
    build_eval_vec_env: Callable | None   # None -> core default path
    train_callbacks: Callable | None      # task callbacks, with ordering slots
    capabilities: frozenset[str]          # e.g. {"off_policy", "best_model_eval"}
```

**Plugin loading.** The repo installs from a lockfile and runs through the
root-level `rl_framework/` path shim, so setuptools entry points are not
available. Instead `rl_framework/plugins.py` holds
`PLUGIN_MODULES = ("rl_framework.physics", "rl_framework.games")`. The registry
imports that list lazily, by `importlib`, the first time a lookup is made, so
it also works inside a `SubprocVecEnv` worker started with the `spawn` method
(which `reproducibility.deterministic` forces). `make_env` handed to a vec env
must be picklable under `spawn`. An `RL_FRAMEWORK_PLUGINS` environment variable
(comma-separated module names) adds to the list, so a task that lives outside
this repo can be loaded without editing it — workers inherit the variable.

---

## Prompt 0 — Baseline and safety net

> Read `AGENTS.md`, then the **Ground rules** and **Target design** sections of
> `docs/plans/2026-11-12-core-physics-games-restructure.md`. You are doing
> prompt 0 of that plan. Change nothing under `src/` in this prompt.
>
> A seven-step restructure of `src/rl_framework` is about to start. Your job is
> to build the safety net that lets later agents prove they changed no
> behaviour. Later agents will trust these tests completely, so a net with
> holes is worse than none.
>
> **1. Environment.** Create `.venv/` and install from `requirements-lock.txt`
> exactly as CI does (see `.github/workflows/ci.yml`). Run the gate. Record the
> commit hash, the number of tests passed, and the coverage percentage — later
> prompts compare against these numbers. If anything fails at this untouched
> commit, stop and report; do not fix it here.
>
> **2. Golden rollouts.** Add `tests/test_golden_rollouts.py` plus fixtures
> under `tests/fixtures/golden/`. For each case below, build the env through
> the same public entry point training uses, reset with a fixed seed, step a
> fixed pre-generated action sequence (a seeded RNG, saved in the fixture, not
> re-drawn at test time) for a few hundred steps across at least two episode
> boundaries, and store observations, rewards, terminated/truncated flags and
> the numeric scalar entries of `info`. The test replays the actions and
> compares. Cases:
> - walker with the `walker_reward_v9_flight_cost` environment block;
> - walker with the `walker_reward_v12_mixed_terrain` environment block (this
>   exercises per-episode terrain sampling — also store `info["terrain_preset"]`
>   per episode);
> - walker with observation v1 defaults (`robot_walk_basic`);
> - arena raw `ParallelEnv` from `organisms_fight_arena`;
> - arena through `SelfPlayEnvWrapper` + `SingleAgentArenaEnv` with an empty
>   league (random-opponent fallback, seeded).
>
> Compare with `np.testing.assert_allclose(rtol=0, atol=1e-6)` for floats and
> exact equality for flags and strings. Run each test three times in fresh
> processes to prove it is deterministic before committing; if a case is not
> reproducible on this machine, find out why and report it rather than widening
> the tolerance.
>
> **3. Legacy checkpoints.** Later prompts move modules, and saved artifacts
> may embed module paths (SB3 cloud-pickles the learning-rate schedule closure
> from `sb3_runner._make_lr_schedule`; league snapshots carry `_vecnorm.pkl`
> sidecars). Using the *current* code, produce tiny artifacts (a couple of
> thousand timesteps, `num_envs: 1`, `device: cpu`) into
> `tests/fixtures/legacy_checkpoints/`:
> - a walker PPO run with `learning_rate_end` set and observation
>   normalisation on (model zip + VecNormalize sidecar + `run_metadata.json`);
> - an arena self-play run long enough to write at least one league snapshot
>   with its sidecar.
>
> Add `tests/test_legacy_checkpoints.py` asserting that each loads through the
> public paths: `evaluate()` for the walker, resume via `train(...,
> resume_from=...)` for a few hundred more steps (including the resume
> provenance check), and `load_frozen_policy` / `LeagueSampler` for the league
> snapshot. Commit the fixtures only if they total under 1 MB; otherwise stop
> and report the size.
>
> **4. Callback order.** `train()` assembles callbacks in an order that
> matters (metrics callbacks must run before `CurriculumCallback` reads them).
> Add a test that captures the ordered list of callback class names passed to
> `model.learn` for three configs — walker with curriculum and best-model
> eval, arena self-play with reward annealing and curriculum, arena
> shared-policy — and asserts it against a literal list.
>
> **Done when:** the gate passes; the new tests pass three runs in a row; no
> file under `src/` changed; baseline numbers are in your report; committed and
> pushed.

---

## Prompt 1 — Registry and `TaskSpec`

> Read `AGENTS.md`, then the **Ground rules** and **Target design** sections of
> `docs/plans/2026-11-12-core-physics-games-restructure.md`. You are doing
> prompt 1. Prompt 0's golden, legacy-checkpoint and callback-order tests must
> pass unchanged when you finish.
>
> `src/rl_framework/envs/registry.py` is a two-branch `if` that imports both
> env classes. Replace it with a real registry, without moving any existing
> file yet.
>
> - Create `src/rl_framework/core/registry.py` with `TaskSpec`,
>   `register_task`, `get_task`, `registered_tasks`, `make_env` and
>   `load_plugins`, as described under **Target design → Registry**. For this
>   prompt only `name` and `make_env` need to be populated on each spec; add
>   the other fields as optional so prompt 2 can fill them in.
> - Create `src/rl_framework/plugins.py`. Until the packages exist (prompt 4)
>   its module list points at the current locations
>   (`rl_framework.envs.locomotion`, `rl_framework.envs.organisms`).
> - Register the walker and the arena from their own packages' `__init__`.
>   Preserve the arena's existing construction detail (`render_mode` is taken
>   from the config and passed as a keyword).
> - Make every existing `make_env` caller go through the new registry and
>   delete `envs/registry.py`.
> - Registering the same name twice with a different spec is an error. An
>   unknown `environment.type` raises a `ValueError` that lists the registered
>   names.
> - `src/rl_framework/envs/base.py` (`EnvContext` and friends) is imported by
>   nothing. Confirm that with a search and delete it.
>
> Tests to add: registration, duplicate rejection, unknown-type message, and —
> the important one — a test that builds a 2-worker `SubprocVecEnv` with
> `start_method="spawn"` for the walker and for the self-play arena and steps
> each once. Under `spawn` the worker is a fresh interpreter with an empty
> registry, so this proves lazy plugin loading works.
>
> **Done when:** gate passes; prompt 0's tests pass untouched; no
> `if env_type ==` chain remains in a registry; committed and pushed.

---

## Prompt 2 — Take the task branches out of the generic layers

> Read `AGENTS.md`, then the **Ground rules** and **Target design** sections of
> `docs/plans/2026-11-12-core-physics-games-restructure.md`. You are doing
> prompt 2. Prompt 0's tests must pass unchanged when you finish. Files still
> do not move in this prompt — you are re-wiring, so that prompt 4 can be a
> pure move.
>
> The generic modules decide behaviour by comparing `environment.type` to
> string literals. Find every one:
>
> ```
> grep -rn "walker_bullet\|organism_arena_parallel" src/rl_framework
> ```
>
> At `459511d` the hits outside the env packages were in
> `training/sb3_runner.py`, `training/eval_runner.py`, `utils/config.py`,
> `gui/app.py`, `training/walker_diagnostics.py`,
> `training/morphology_search.py`. The last two are task code and may keep
> their literals; the first four are generic and must end with none (the GUI
> may still *display* a type string it got from the registry).
>
> For each branch, move the task-specific half behind a `TaskSpec` hook owned
> by the task and leave the generic half in place:
>
> - **`train()` in `sb3_runner.py`.** It is one ~400-line function. Split it
>   into `build_train_vec_env`, `build_model` and `build_callbacks`, with
>   `train()` as the short orchestrator. The three vec-env paths (default
>   Monitor path; arena self-play; arena shared-policy via SuperSuit with the
>   `num_envs > 1` guard) become: a core default plus the arena spec's
>   `build_train_vec_env`. `_ArenaVecEnvAdapter`, `ArenaMetricsCallback`,
>   `WalkerMetricsCallback`, the self-play/league wiring, reward annealing and
>   the velocity-target ramp become things their task spec contributes.
>   **Callback order is load-bearing** — prompt 0's order test must still pass
>   with the same literal list. Design the hook so a task can place callbacks
>   both before and after the generic curriculum callback.
> - **`evaluate()` in `eval_runner.py`.** The multi-agent evaluation loop and
>   its per-agent metrics move behind the arena spec; the seeded-worker path
>   stays as the core default.
> - **`validate_experiment_config` in `utils/config.py`.** Generic checks stay.
>   `_validate_env_specific` splits into each spec's `validate_config`. The
>   checks that gate a feature by env name (SAC/TD3 "walker only",
>   `evaluation.best_model` "walker only") become capability checks against
>   `spec.capabilities` with the same error text, so existing validation tests
>   keep passing.
> - **`get_schema` in `gui/app.py`.** The per-task schema dicts move to each
>   spec's `gui_schema`; `get_schema` assembles the response from
>   `registered_tasks()`. The JSON the endpoint returns must be identical —
>   capture it before you start and diff it after.
>
> Constraints: this is a re-wiring. The same code runs in the same order with
> the same arguments; it just lives behind a hook. Where a branch silently
> ignores a setting for the "other" env (for example `self_play.enabled` on a
> walker config), keep ignoring it and list it in your report — prompts 5 and 6
> decide those.
>
> Smoke tests (in addition to the gate): a few hundred timesteps of training
> for the walker (`num_envs: 2`), arena self-play (`num_envs: 2`) and arena
> shared-policy (`num_envs: 1`); `evaluate()` on each result; start the GUI and
> fetch the schema and config-list endpoints.
>
> **Done when:** gate passes; prompt 0's tests pass untouched; the grep above
> shows no literal in the four generic modules; schema JSON diff is empty;
> smoke output is in your report; committed in logical groups and pushed.

---

## Prompt 3 — Split `quality_study.py`

> Read `AGENTS.md`, then the **Ground rules** and **Target design** sections of
> `docs/plans/2026-11-12-core-physics-games-restructure.md`. You are doing
> prompt 3. Files outside `quality_study.py` and its tests still do not move.
>
> `src/rl_framework/training/quality_study.py` is 2,616 lines and imports both
> walker diagnostics and the arena tournament, so it cannot live in any one of
> the three target packages. Split it along the seam that already exists in
> the file:
>
> - a generic study runner: state file load/save and resume, atomic JSON
>   writes, config preparation and overrides, `_train_run`, `StopOnWallClock`,
>   report formatting, and the `run_quality_study` dispatcher;
> - the walker studies: every `WALKER_*` constant and every `_walker_*` /
>   `_run_walker_*` / slip-recovery / action-memory function;
> - the arena studies: `ARENA_*` constants and the `_arena_*` /
>   `_run_arena_study` functions.
>
> The dispatcher must not name studies itself. Give the registry (or a small
> study registry beside it) a way for a task package to contribute named
> studies, and have the runner look them up. `_run_algorithm_study` compares
> algorithms on a task — read it and place it with whichever task it actually
> exercises; say which and why in your report.
>
> This is a move-and-rewire. Function bodies stay byte-identical apart from
> imports. The CLI behaviour, the study names, the state-file format and the
> report text must not change — `tests/test_quality_study.py` is the check, so
> update only its imports and patch targets, not its assertions. A state file
> written before the split must resume after it; add a test for that if none
> exists.
>
> **Done when:** gate passes; prompt 0's tests pass untouched; the generic
> runner module has no walker or arena import; committed and pushed.

---

## Prompt 4 — Move into `core` / `physics` / `games` and enforce it

> Read `AGENTS.md`, then the **Ground rules** and **Target design** sections of
> `docs/plans/2026-11-12-core-physics-games-restructure.md`. You are doing
> prompt 4. Prompts 1–3 removed the cross-dependencies; this prompt is the
> physical move plus the lint rule. If you find a generic module that still
> needs a task import, stop and report — do not paper over it with a lazy
> import or a string.
>
> **Move.** `git mv` every module to its place in **Target design → Layout**.
> Do it in three commits (core, then physics, then games), each updating all
> importers — `src/`, `tests/`, `scripts/` — so the gate passes after each. At
> `459511d` there were about 144 dotted `rl_framework.<old package>` references
> in `tests/`, plus one string-path `monkeypatch` target; find the rest with
> grep rather than trusting that count. No shims at old paths. `cli/`, `gui/`
> and `configs/` do not move. Update `plugins.py` to the final module names.
> Leave the root-level `rl_framework/__init__.py` path shim alone and confirm
> `python -m rl_framework.cli.main --help` works from the repo root both with
> and without `pip install -e .`.
>
> **Enforce.** Extend `scripts/check_repo_policy.py` (it already runs in CI and
> needs no new dependency) with an import-direction check implemented over the
> AST of every file under `src/rl_framework`:
> - walk *all* `Import`/`ImportFrom` nodes, including ones nested in functions
>   and `if TYPE_CHECKING` blocks, and resolve relative imports;
> - apply the table in **Target design → Import rules**;
> - for files under `core/`, also fail on the banned string literals listed
>   there (docstrings and comments excepted).
>
> Each violation prints `file:line`, the offending import, and the rule it
> broke. Add tests that feed the checker small synthetic trees — one violation
> of each rule, a function-local import, a relative import, and a clean tree —
> and a test that the real tree is clean.
>
> **Prove the legacy fixtures survived.** Prompt 0's legacy-checkpoint test is
> the reason this is safe. If loading a pre-move checkpoint now fails because
> an artifact embeds an old module path, do not regenerate the fixture: report
> exactly which artifact and which path, and propose the narrowest fix (the
> user may have real checkpoints with the same problem).
>
> **Docs.** Update every path that changed in `AGENTS.md` (Repository layout,
> GUI and arena sections, the three gate commands if needed), `README.md`,
> `VISUALIZATION_GUIDE.md`, `docs/organism_arena_config.md`, `docs/STATUS.md`,
> and the skills under `.claude/skills/` and `.agents/skills/` (they are
> mirrored — keep them identical). Leave dated historical documents
> (`docs/plans/*`, `reviews/*`, old review files, the learning history) as they
> are. Add `docs/decisions/002-core-physics-games-split.md` recording the
> decision, the import rules, why `cli`/`gui` are the application layer, and
> why plugin loading uses a module list instead of entry points.
>
> Smoke tests: the same three training runs and evaluations as prompt 2, plus
> one walker run with `reproducibility.deterministic: true` and `num_envs: 2`
> (this forces `spawn` workers), plus `arena-tournament --include-random` over
> the arena smoke checkpoint, plus the GUI endpoints.
>
> **Done when:** gate passes including the new policy check; prompt 0's tests
> pass untouched; test count and coverage are not below prompt 0's baseline;
> `git log --follow` on a moved file shows its pre-move history; smoke output
> is in your report; pushed.

---

## Prompt 5 — Bug fixes: live parameters

> Read `AGENTS.md`, then the **Ground rules** and **Target design** sections of
> `docs/plans/2026-11-12-core-physics-games-restructure.md`. You are doing
> prompt 5. From here on behaviour changes on purpose. For each fix: write a
> test that fails on current `master` for the stated reason, then fix it. If a
> golden-rollout test changes, the fix leaked into default behaviour — stop
> and report.
>
> All three bugs come from one design flaw: live parameter updates fail
> silently. `update_live_params` on both envs skips keys it does not
> recognise, and `CurriculumCallback._apply_level_params` and
> `LiveTuningCallback` wrap the call in `except Exception: pass`. A typo in a
> curriculum therefore trains at the wrong difficulty and reports success.
>
> **5a. Make live parameters strict.** Each task declares the dotted keys it
> can change live (add this to `TaskSpec`, and have the env's
> `update_live_params` use the same declaration). An undeclared key raises
> with the list of valid keys; a value that cannot be coerced raises. Remove
> both `except Exception: pass` blocks — the only tolerated case is a vec env
> that truly has no `env_method`, and that should be detected explicitly, not
> by catching everything. Before changing behaviour, add a test that loads
> every YAML in `configs/experiments/` and checks each curriculum
> `level_params` key (and any velocity-ramp or annealing key) against the
> task's declaration. If a shipped config uses a key that was being silently
> dropped, do not delete it — report it, because it means a recorded
> experiment did not do what its config says.
>
> **5b. Let a curriculum change terrain.** `AGENTS.md` and `docs/STATUS.md`
> list "CurriculumCallback cannot change terrain" as a known problem: the
> walker's `update_live_params` has no `terrain` section. With 5a in place,
> declare `terrain.*` keys (including a full `terrain.mix`) as live
> parameters. Terrain geometry is built at reset, so a terrain update takes
> effect on the next `reset()` — document that in the method, reuse the
> existing `_parse_terrain_mix` / `_apply_terrain_spec` path so validation
> (an empty `mix: []` is an error) still applies, and keep the once-per-env
> `_build_world()` optimisation. Test: change terrain mid-run through
> `env_method` on a 2-worker vec env and assert `info["terrain_preset"]`
> changes on the following episode and not before.
>
> **5c. Stale gait timing.** `WalkerBulletEnv.__init__` passes
> `control_timestep = sim.timestep * sim.frame_skip` to `WalkerGaitTracker`
> once. `update_live_params` accepts `sim.timestep` and `sim.frame_skip` and
> updates the physics, but not the tracker, so stride durations and cadence
> are computed with the old step length afterwards. Check whether anything
> else derived from those two values (the gait phase clock is the obvious
> candidate) has the same problem, fix all of them in one place, and test by
> changing `frame_skip` live and asserting the reported durations scale.
>
> **Docs.** Remove the two resolved entries from Known problems in
> `docs/STATUS.md` and the matching invariant lines in `AGENTS.md`
> ("CurriculumCallback cannot change terrain. Use `terrain.mix`"), replacing
> them with the new rule (unknown live keys are errors; terrain changes apply
> at next reset). ADR-001 chose `terrain.mix` over a curriculum partly because
> the curriculum could not do it — add a short dated note to that ADR saying
> the limitation is gone and the `mix` decision stands on its other grounds;
> do not rewrite it.
>
> Smoke test: train `walker_balance_curriculum` and `organisms_fight_arena`
> (it has a curriculum block) for a few thousand steps with thresholds lowered (in a temp config,
> not the shipped YAML) so at least one level-up fires, and show the level-up
> log line and the changed parameter read back from a worker.
>
> **Done when:** gate passes; each fix has a test that failed before it;
> golden rollouts unchanged; pushed as three commits.

---

## Prompt 6 — Bug fixes: training pipeline

> Read `AGENTS.md`, then the **Ground rules** and **Target design** sections of
> `docs/plans/2026-11-12-core-physics-games-restructure.md`. You are doing
> prompt 6. Same method as prompt 5: failing test first, then the fix, one
> commit per fix. Verify each bug exists before fixing it — these were found
> by reading code at `459511d`, not by running it. If one turns out not to be
> real, say so and skip it.
>
> **6a. `training.policy_kwargs` is silently ignored.** `build_model` only
> ever sets `policy_kwargs = {"log_std_init": ...}` for PPO. A `net_arch` or
> activation written in YAML is dropped, yet `reproducibility.py` records
> `training.policy_kwargs` in the resume fingerprint as if it had been used.
> Pass `training.policy_kwargs` through to the algorithm for PPO, SAC and TD3,
> merge `log_std_init` into it for PPO (error if both specify it with
> different values), validate that it is a mapping, and let YAML name an
> activation function by string. Add the field to the GUI training schema
> only if the wizard can already render a free-form mapping; otherwise leave
> the GUI alone and say so. Default behaviour with the key absent must be
> identical (the legacy-checkpoint tests cover this).
>
> **6b. Curriculum level is lost on resume.** Nothing persists the curriculum
> level: `CurriculumCallback` starts from `curriculum.level` in the config
> (default 0) and the env is rebuilt from the YAML's base parameters, so a
> resumed run restarts at level 0 with weights trained at a higher level.
> Confirm by test. Then persist the level beside the checkpoints (written on
> each level-up and whenever a checkpoint is saved), and on resume restore it
> and re-apply that level's parameters to the config and the live envs before
> the first rollout. A resume with no saved state behaves as today and logs
> that it is starting from the configured level. Do **not** change
> `VelocityTargetRampCallback`: it deliberately measures its ramp from the
> continuation's start, and the v11/v12 configs were written around that.
>
> **6c. Frozen opponents are PPO-only.** `load_frozen_policy`,
> `LeagueSampler._ensure_loaded` and the arena eval/tournament loaders call
> `PPO.load` directly. Route them through one loader that picks the algorithm
> class the same way `build_model` does, so a league or tournament is not tied
> to PPO. Whether SAC/TD3 can then be enabled for arena training is a separate
> question — check what actually blocks it; if it is only the capability flag,
> enable it and add a short smoke test; if not, leave the flag and report the
> blocker. While there: `SelfPlayCallback.sample_opponent` and its
> `_model_cache` are called only by tests (the env-side `LeagueSampler`
> replaced them). Confirm with a search and remove the dead method, its cache
> and its tests.
>
> **6d. A malformed `sim.control` passes validation.** `AGENTS.md` records
> that a flattened `control: null` "looks like a working run and is not one".
> The env tolerates it through `get_section` so old run configs still load —
> keep that tolerance in the env. Make the walker's `validate_config` reject a
> `sim.control` (and the other nested groups the walker reads) that is present
> but not a mapping, so the CLI and the GUI save path refuse it up front.
> Confirm the GUI wizard no longer emits it; add a GUI API test that a
> wizard-built walker config round-trips with a nested `control` block.
>
> **6e. Dockerfile ignores the lockfile.** `AGENTS.md` says to install from
> `requirements-lock.txt`; the `Dockerfile` installs `requirements.txt`. Read
> both files and the policy script first. If `requirements.txt` has no other
> purpose, switch the Dockerfile to the lockfile. You cannot build the image
> in this session unless Docker is available — say whether you built it.
>
> **Docs.** Update `docs/STATUS.md` Known problems and `AGENTS.md` invariants
> for whatever is now fixed, and `docs/training_presets.md` /
> `docs/organism_arena_config.md` for the new config keys.
>
> Smoke tests: a walker run with a non-default `net_arch`, confirming the
> built network's layer sizes; a curriculum run stopped after a level-up and
> resumed, showing the restored level; an arena tournament over a league.
>
> **Done when:** gate passes; one commit per fix, each with its failing-first
> test; golden rollouts and legacy checkpoints unchanged; pushed.

---

## Prompt 7 — Independent audit

> Read `AGENTS.md`, then the **Ground rules** and **Target design** sections of
> `docs/plans/2026-11-12-core-physics-games-restructure.md`. You are doing
> prompt 7. You did not write any of this work. Your job is to find what the
> previous six agents got wrong or left out, not to confirm that they
> finished. Fix small things you find; report large ones.
>
> Review the full diff from `459511d` to `HEAD`.
>
> 1. **The safety net is intact.** Diff `tests/test_golden_rollouts.py`,
>    `tests/test_legacy_checkpoints.py`, the callback-order test and
>    everything under `tests/fixtures/` between prompt 0's commit and `HEAD`.
>    Fixtures must be byte-identical. Any changed assertion, tolerance, skip
>    or xfail needs a justification you can find in a commit message; list any
>    that lack one.
> 2. **The boundary is real.** Run the policy check. Then try to defeat it:
>    look in `core/` for `importlib` calls, `getattr` on modules, task names
>    assembled from pieces, `isinstance` checks against task classes reached
>    indirectly, and hooks on `TaskSpec` that exist for exactly one task and
>    are really an `if` in disguise. Report each with a file and line.
> 3. **The acceptance test for the whole design.** In a scratch directory
>    outside `src/` (do not commit it), write a minimal third task — a trivial
>    real-time 2D game with two agents is ideal — as one package with a
>    `TaskSpec`, add its module name through whatever override `plugins.py`
>    offers, and train it for a few hundred steps through the normal CLI with
>    a YAML. Count the files under `src/rl_framework` you had to edit. The
>    target is zero. Every edit you needed is a finding.
> 4. **Nothing was dropped.** Compare the public names (functions, classes,
>    CLI subcommands and flags, GUI routes) at `459511d` and at `HEAD`.
>    Anything missing must be accounted for by a prompt (`envs/base.py`,
>    `sample_opponent`). Compare test count and coverage with prompt 0's
>    baseline.
> 5. **Docs match the code.** Every path and command in `AGENTS.md`,
>    `README.md`, `docs/STATUS.md` and both skills directories must exist and
>    run. Bring `docs/STATUS.md` fully up to date (commit, current state,
>    known problems, validation state with what you actually ran) and update
>    `docs/VALIDATION.md`.
> 6. **Full run.** The gate, then every smoke test named in prompts 2, 4, 5
>    and 6, then each CLI subcommand at least once.
>
> Report as a list of findings ordered by severity, each with evidence, then
> what you fixed, then what you ran. Say plainly if you found nothing in a
> category — and how hard you looked.

---

## Notes for whoever runs the series

- **What was verified when this was written.** `ruff check src tests scripts`
  passed at `459511d`. The test suite was *not* run — there was no venv on the
  machine — so prompt 0's baseline is the first real measurement.
- **Bugs in prompts 5 and 6 were found by reading, not running.** Each prompt
  tells the agent to reproduce before fixing. 6b (curriculum resume) is the
  least certain.
- **Risk is concentrated in prompts 2 and 4.** Prompt 2 rewires `train()`;
  prompt 4 touches every import. If either agent reports a golden-rollout or
  legacy-checkpoint failure, treat that as a real regression, not a flaky
  test.
- **Existing checkpoints elsewhere.** `outputs/` is gitignored and was empty
  here. If real v9–v12 checkpoints exist on another machine, load one after
  prompt 4 before relying on the fixtures alone.
