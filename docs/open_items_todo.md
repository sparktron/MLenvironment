# Open Items / TODO Plan

## Near-term (small/medium)
- ~~Add strict resume compatibility checks for model + `vecnormalize.pkl` provenance.~~ **Done** — `reproducibility.check_resume_provenance` compares a resuming run against the source checkpoint's `run_metadata.json` and flags silent drift SB3's space check misses (changed `environment` block, flipped `normalize_observations`, different policy/gamma). Strict mode hard-fails on drift or an unverifiable manifest; non-strict warns. Wired into `train()`.
- ~~Standardize all model path handling through one helper module.~~ **Done** — `utils/checkpoint.py` now owns `model_zip_path`, `vecnormalize_path_for_model`, `legacy_vecnormalize_path_for_model`, `find_vecnormalize_path_for_model`, and `validate_resume_path`. `sb3_runner` (private re-export aliases), `eval_runner`, `reproducibility` (`_as_zip_path`), and `morphology_search` (`_as_model_zip_path`) all delegate to it instead of re-implementing the `.zip`/sidecar resolution chain.
- ~~Persist run manifest (`git SHA`, resolved config, dependency snapshot, runtime device info).~~ **Done** — `reproducibility.write_run_metadata` writes `run_metadata.json` per run (git commit/branch/full worktree dirty state including staged and untracked files, full config + sha256, lockfile sha256, python/platform/host; device via persisted config). Wired into `train()`, with a strict mode that hard-fails on missing provenance.
- ~~Audit June arena evaluation timeout accounting.~~ **Done** — `eval_runner.evaluate()` now recognizes arena `episode_outcome: timeout` as a truncation and reuses the arena VecEnv adapter so SuperSuit's `uint8` done arrays cannot skew VecNormalize masking.
- ~~Track repository-specific agent workflow guidance.~~ **Done** — `AGENTS.md` is now part of the repo and documents local setup, CLI conventions, environment quirks, GUI behavior, and commit/push workflow for agent-assisted maintenance.
- ~~Add pytest coverage tooling to dev dependencies.~~ **Done** — `pytest-cov` is declared in `pyproject.toml`, pinned in `requirements-lock.txt`, and supports the documented `--cov=src/rl_framework --cov-fail-under=60` check. The dev pytest constraint excludes the vulnerable 9.0.2 release.
- ~~Add JSON output mode for CLI commands for automation-safe parsing.~~ **Done** — every non-GUI subcommand returns a result dict; `--json` emits it as a single JSON line to stdout (suppressing human output) and `--json-out <path>` writes the payload to a file. CLI-level coverage in `tests/test_cli_json_output.py`.
- ~~Add CSV schema stability checks/versioning for metrics files.~~ **Done** — `append_metrics_csv` now reads the existing header and raises `CsvSchemaError` on any column add/remove/reorder instead of silently dropping new keys (`extrasaction="ignore"`) or padding missing ones (`restval=""`). Covered in `tests/test_logging_utils.py`.

## Larger changes (separate session)
- ~~Rework experiment storage layout to avoid name mutation in multi-seed and sweep orchestration.~~ **Done** — variants now route through `output.run_id` (`<experiment>/runs/<run_id>/seed_<seed>/`); multi-seed no longer mutates the name. See `create_experiment_paths`.
- ~~Replace file-based GUI tuning/status IPC with a durable queue.~~ **Done
  (2026-07-12)** — `utils.run_registry.RunRegistry` stores GUI tuning commands,
  status/metric events, and run state in a SQLite WAL database under the output
  root. The training callback atomically claims queued tuning commands at the
  next rollout boundary; status is persisted independently of the GUI process.
- ~~Work through the GUI correctness and layout plan in `docs/ui_roadmap.md`.~~ **Done** — fixed stale walker metadata, template-to-environment state reset, dashboard disabled/empty states, variant run labels, compact mobile wizard progress, and denser desktop dashboard layout. The roadmap file now records implementation status and the browser checklist.
- ~~Introduce end-to-end reproducibility mode (deterministic settings + enforcement + metadata).~~
  **Done (2026-07-12)** — `reproducibility.deterministic: true` seeds Python,
  NumPy, and PyTorch; enables deterministic PyTorch algorithms; resolves to one
  PyTorch thread and spawned workers; and persists the resolved configuration
  in `run_metadata.json`.
- ~~Add CI pipeline for lint/test/type checks and lockfile validation.~~ **Done** — `.github/workflows/ci.yml` runs pytest+coverage, ruff, advisory mypy, security audit, and `check_repo_policy.py` (lockfile completeness). Checkout/setup-python use Node 24-compatible action majors.
- ~~Remove tracked generated artifacts (`__pycache__`, `.egg-info`) and enforce clean repository hygiene.~~ **Done** — `.egg-info` untracked; `check_repo_policy.py` now fails on any tracked `__pycache__`/`.pyc`/`.egg-info`/`.venv` unconditionally (was gated behind `STRICT_REPO_CLEAN`).

## Known limitations currently retained
- Single active GUI run policy.
- GUI still permits only one active in-process training run.

## Next development roadmap (2026-07-05) — full code review

This section is the **current execution roadmap**, produced by a fresh full
review of every module under `src/rl_framework/`, the scripts, bundled
configs, tests, and the GUI frontend (2026-07-05). It supersedes the
2026-06-30 section below for ordering; the still-open 2026-06-30 items
(best-checkpoint eval, walker reward rebalance, observation-v2, throughput
presets, run registry, IPC event stream) remain valid and are referenced
rather than duplicated.

Verification evidence: the P0-1 diagnosis was checked against the pinned
`stable-baselines3==2.8.0` wheel source, not inferred. In
`common/on_policy_algorithm.py`, `callback.on_rollout_end()` fires at the end
of `collect_rollouts` (line 266), while `rollout/ep_rew_mean` is recorded and
immediately dumped inside `dump_logs` (lines 291, 335), which runs *after*
the callbacks; `Logger.name_to_value` is a `defaultdict(float)` (logger.py:486)
cleared on every `dump()` (logger.py:542). Reads at `on_rollout_end` therefore
return 0.0 for all `rollout/*` keys, never raise, and *insert* the key.

### Priority 0: training-breaking bugs

- ~~Fix curriculum/live-tuning metric reads from the SB3 logger.~~
  **Done (2026-07-07)** — `training/rollout_metrics.py` resolves
  `rollout/ep_rew_mean` / `ep_len_mean` from `model.ep_info_buffer` (the same
  source SB3's `dump_logs` uses) and any other key via a membership-checked
  logger read, so absent ≠ 0.0 and nothing is inserted. Both callbacks use
  it, and `LiveTuningCallback` now publishes its status snapshot every
  rollout (previously only after a tuning event was applied, so GUI metrics
  never streamed on untouched runs). Regression tests replace the masking
  plain-dict stub with real defaultdict semantics, cover the
  missing-metric/no-pollution case, and pin ArenaMetricsCallback before
  CurriculumCallback in `train()`'s callback list. Verified by a walker
  smoke run where the default-metric curriculum leveled up twice.
  Original diagnosis:
  `CurriculumCallback._on_rollout_end` (`curriculum_callback.py:93`) and
  `LiveTuningCallback._write_status` (`live_tuning_callback.py:121-134`) read
  `logger.name_to_value[...]` at rollout end. Per the evidence above, all
  `rollout/*` keys read as 0.0 there. Consequences:
  * The walker curriculum (default metric `rollout/ep_rew_mean`) **never
    levels up, silently** — or levels up instantly if a threshold ≤ 0 is set.
  * The GUI dashboard's Mean Reward / Ep Length cards are always `0.0` and
    the reward-history chart is a flat zero line; `train/*` values shown are
    one iteration stale (recorded by the previous `train()`, dumped later).
  * The defaultdict access inserts the key, so the next TensorBoard dump gets
    spurious 0.0-valued `rollout/*` rows.
  * The **arena** curriculum works only by callback-ordering accident:
    `ArenaMetricsCallback` records `arena/*_win_rate` in its own
    `_on_rollout_end`, which `sb3_runner.train()` registers *before*
    `CurriculumCallback`, so the value is still in `name_to_value` when the
    curriculum reads it.
  * `tests/test_curriculum_annealing.py` masks all of this: `_StubLogger`
    uses a plain dict pre-populated with the metric.
  Fix plan: compute `ep_rew_mean` / `ep_len_mean` from
  `self.model.ep_info_buffer` in both callbacks (that buffer is exactly what
  `dump_logs` reads); for any other metric use a membership check
  (`key in logger.name_to_value`) so absent ≠ 0.0 and nothing is inserted.
  Validation: regression test whose stub logger reproduces the real semantics
  (`defaultdict(float)` with `rollout/*` cleared at rollout end); a test that
  pins ArenaMetricsCallback ordering before CurriculumCallback in
  `sb3_runner.train`; a short walker smoke run with a low curriculum
  threshold confirming a level-up actually fires.

- ~~Fix self-play snapshot cadence at `num_envs > 1`.~~
  **Done (2026-07-07)** — `_on_step` now fires on the first callback call
  at/after each `snapshot_freq` boundary, with the schedule anchored to
  multiples of the frequency so overshoot never accumulates. Regression test
  drives strides 1/7/24 and asserts one snapshot per window; verified
  in-vivo with a `num_envs: 8` self-play run producing snapshots at exactly
  2000/4000/6000 for freq 2000. Original diagnosis:
  `SelfPlayCallback._on_step` (`self_play_callback.py:88`) used
  `num_timesteps % snapshot_freq == 0`, but `num_timesteps` advances by
  `num_envs` per callback call, so snapshots only land when the stride
  happens to hit an exact multiple — effectively an LCM cadence. With the
  bundled `organisms_fight_arena.yaml` (`snapshot_freq: 5000`) at the
  config-comment-recommended `num_envs: 24`, snapshots fire every **15,000**
  steps (a 3× sparser league than configured); at `num_envs: 7`, every
  35,000. The runner already fixed this exact class of bug for checkpoints
  via `_callback_freq_from_timesteps` (`sb3_runner.py:278`) but passes
  `snapshot_freq` raw to `SelfPlayCallback` (`sb3_runner.py:520`).
  Fix plan: threshold trigger inside the callback
  (`num_timesteps - last_snapshot >= snapshot_freq`), keeping config units in
  env timesteps. Validation: regression test stepping `num_timesteps` in
  strides of 7 and 24 and asserting snapshot count.

- ~~Make league snapshot writes atomic and sampler loads race-safe.~~
  **Done (2026-07-07)** — `_save_snapshot` saves the model under a temp name
  the sampler's `selfplay_*.zip` glob cannot match, writes the vecnorm
  sidecar, then `os.replace`s the zip into place, so a visible snapshot
  always has a complete archive and sidecar; `LeagueSampler._ensure_loaded`
  additionally re-probes the sidecar on cached entries that have none
  (pre-fix leagues). Tests cover write ordering, no leftover temp files, and
  late-sidecar attachment. Original diagnosis:
  `SelfPlayCallback._save_snapshot` wrote the `.zip` and then the
  `_vecnorm.pkl` sidecar non-atomically while `LeagueSampler`
  (`self_play_env_wrapper.py:142-212`) re-globs from subprocess workers on
  directory-mtime change. Two races: (a) a worker can `PPO.load` a
  half-written zip and crash the whole `SubprocVecEnv`; (b) `_ensure_loaded`
  (line 190) caches a `FrozenPolicy` with `normalizer=None` if `sample()`
  picks the newest snapshot before its sidecar lands — that opponent then
  plays on unnormalized observations **permanently in that worker** (the
  sidecar gate at line 186 protects only the pre-warm path, not `sample()`).
  Fix plan: write the snapshot to a temp name and `os.rename` it into place
  (atomic on the same filesystem), writing the sidecar *before* the final zip
  rename so a visible zip always implies a complete sidecar; in
  `_ensure_loaded`, when normalization is in use, re-check for a sidecar on
  cache entries that have none. Validation: unit test staging the write
  sequence (zip present, sidecar absent → not sampled / normalizer picked up
  later) plus a parallel self-play smoke run at `num_envs: 8`.

### Priority 1: correctness bugs

- ~~Stop the GUI wizard from stripping non-schema config sections.~~
  **Done (2026-07-08)** — `assembleConfig()` now mutates the existing
  `currentConfig` object (setting only `experiment_name`/`seed`/
  `environment`/`training`/`evaluation`/`output`, the sections it actually
  renders) instead of replacing it with a fresh 5-key object literal, so a
  loaded template's `self_play`/`reward_annealing`/`curriculum`/`sweep`/
  `multi_seed`/`reproducibility` sections survive to launch. The env-card
  click handler's existing `currentConfig = {}` reset on an actual
  environment-type change is unchanged. No JS test harness exists in this
  repo, so verification was a Playwright browser run: loading
  `organisms_fight_arena`, advancing to Review & Launch, confirming
  `self_play`/`reward_annealing`/`curriculum` and `training.num_envs: 8` in
  the assembled preview, then actually clicking Launch and confirming
  `POST /api/train/start` returns `{run_id, status: "started"}` with no
  validation error (previously rejected `num_envs: 8` without self_play as a
  false positive, since self_play was silently dropped). Original diagnosis:
  `assembleConfig()` (`gui/static/app.js:363-422`) rebuilt `currentConfig`
  from scratch with only `environment` / `training` / `evaluation` /
  `output.base_dir`, so a loaded template loses `self_play`,
  `reward_annealing`, `curriculum`, `sweep`, `multi_seed`, and
  `reproducibility` on the way to launch. Concretely: the bundled
  `organisms_fight_arena.yaml` template (self-play, `num_envs: 8`) becomes
  un-launchable — the validator rejects `num_envs: 8` without self_play —
  with a confusing error; worse, the checked-by-default "Save config as YAML
  template" (`app.js:430`; `POST /api/configs` names the file from
  `experiment_name`) **overwrites the original YAML with the stripped
  config** whenever the stripped version still validates (e.g. any template
  whose only extra section is `sweep`). Fix plan: merge form values over a
  deep clone of the loaded `currentConfig` (preserving unknown sections)
  instead of rebuilding; reset only when the environment type changes (that
  path already clears state). Validation: GUI API test that a load→launch
  round-trip preserves `self_play`; browser check of the fight-arena
  template flow end-to-end.

- ~~Generalize `ArenaMetricsCallback` beyond 2 agents.~~ **Done (2026-07-08)**
  — win counts are now keyed lazily by whatever winner name is observed
  (`self._wins.get(winner, 0) + 1`), with `agent_0`/`agent_1` seeded up
  front so their rate keeps logging as 0.0 in rollouts they don't win
  (matching prior 2-agent behavior exactly). Regression test drives a
  synthetic 4-agent outcome stream and asserts `arena/agent_2_win_rate` is
  logged and that it persists at 0.0 in a later rollout with no agent_2
  wins. Original diagnosis: `sb3_runner.py:158` hardcoded
  `self._wins = {"agent_0": 0, "agent_1": 0}`;
  for N-agent arenas (the GUI schema allows `num_agents` up to 8) wins by
  `agent_2+` increment `_outcomes` but are recorded nowhere, so the logged
  win rates are silently wrong. Fix plan: accumulate winner names into a
  dict built lazily (or from the env's `possible_agents`) and log per-agent
  rates. Validation: unit test feeding a synthetic N=4 outcome stream.

- ~~Normalize `section: null` tolerance across the config surface.~~
  **Done (2026-07-08)** — `utils/config_merge.get_section(cfg, key)` treats
  a missing key and an explicit `key: null` identically, normalizing to
  `{}` and writing the result back into `cfg` so later direct reads are
  also safe; `set_nested(strict=False)` now descends through it too. Used
  at every site the review flagged: `OrganismArenaParallelEnv.__init__` and
  `update_live_params`, `WalkerBulletEnv.__init__`/`_build_world`/
  `_apply_domain_randomization`/`reset`/`update_live_params`, and
  `LiveTuningCallback._on_rollout_end`. Regression tests construct each env
  with every relevant section explicitly `null`, exercise
  `update_live_params`/live tuning against a null section, and confirm no
  crash plus correct value propagation. Original diagnosis: the walker
  constructor deliberately guarded with `or {}`
  (`walker_bullet.py:33-35`) because the GUI has historically written
  `key: null` for nested groups, but sibling paths don't:
  `OrganismArenaParallelEnv.__init__` uses `cfg.get("sim", {})` /
  `get("battle_rules", {})` / `get("morphology", {})`
  (`arena_parallel.py:45-63`) which return `None` when the key exists as
  null → `AttributeError` at construction; `WalkerBulletEnv.
  update_live_params` does `self.cfg.setdefault(section, {})`
  (`walker_bullet.py:589,595`) where setdefault returns the stored `None`;
  `LiveTuningCallback._on_rollout_end` does `param in self._env_cfg[section]`
  (`live_tuning_callback.py:82`) — `in None` raises uncaught at rollout end
  and **kills the training thread**; `set_nested(strict=False)`
  (`utils/config_merge.py:24`) setdefaults into a `None` intermediate. Fix
  plan: one shared `_section(cfg, key)` helper (get-or-{} that also replaces
  a stored `None` with `{}`), used at all these sites. Validation: unit
  tests feeding null sections through arena construction, walker
  `update_live_params`, live tuning, and curriculum overrides.

- ~~Actually persist `episode_outcome` to arena Monitor CSVs.~~ **Done
  (2026-07-08)** — `Monitor(env, filename=filename, info_keywords=
  ("episode_outcome",))`; the key is guaranteed present on the live agent's
  terminal-step info in every reachable terminal state (elimination win,
  elimination draw, and timeout all set it before the wrapper surfaces a
  terminated/truncated step). Regression test trains a short self-play run
  and asserts the Monitor CSV header includes `episode_outcome` with a
  non-empty value on at least one row. Original diagnosis: the comment in
  `_build_arena_selfplay_env` (`sb3_runner.py:257-259`) claimed
  `info_keywords` persists the per-episode outcome, but the `Monitor` is
  constructed without it. Fix plan: pass
  `info_keywords=("episode_outcome",)` (the key is present on the live
  agent's terminal-step info). Validation: test that the Monitor CSV gains
  the column after a terminated episode.

- ~~Migrate `robot_push_recovery.yaml` to the current robot.~~ **Done
  (2026-07-08)** — `sim.mass: 3.2 -> 28.0` and `max_force: 45.0 -> 35.0`
  (the current Atlas-class defaults / 1.0× torque baseline used elsewhere).
  Left the filename and the config's balance-focused reward/termination/
  domain-randomization tuning as-is: README already describes the preset
  accurately as "aggressive randomization ... for robustness" rather than
  claiming a push mechanic, so renaming wasn't required to fix the
  correctness bug and would have added README/doc churn for a cosmetic
  concern. The perturbation mechanism itself remains an open Priority 3
  feature. Regression test loads the config and pins the migrated values so
  they cannot silently drift back to the pre-overhaul body. Original
  diagnosis: it still shipped pre-overhaul values — `sim.mass: 3.2` (the
  torso would be lighter than a single 7 kg thigh on the Atlas-class body)
  and `max_force: 45`, which the new dynamics reinterpret as a 45/35 ≈
  1.29× global scale over the per-joint Atlas torque caps.

### Priority 2: usability and performance

- **Capture GUI frames from one env, on a wall-clock budget.**
  `TrainingManager._train_worker` forces `render_mode: "rgb_array"` for
  every GUI run, and `FrameCaptureCallback._capture_frame`
  (`frame_capture_callback.py:82`) calls `training_env.render()`, which on a
  VecEnv renders **all** workers (PyBullet TinyRenderer, 640×480, CPU) and
  tiles them into a mosaic. At `num_envs: 24` with `capture_interval=50`
  env-steps this stalls every worker every ~2-3 vector steps and the
  dashboard shows a 24-up grid instead of one robot. Fix: render env 0 only
  (`env_method("render", indices=[0])` / `envs[0].render()`), throttle
  captures by wall-clock (≥1 s apart), and set `render_mode` only when frame
  capture is enabled.

- **Throttle `RewardAnnealingCallback` env updates.**
  The scale changes every step until annealing completes
  (`reward_annealing_callback.py:29-37`), so the `scale == last_scale`
  dedupe only helps *after* annealing — until then it is an `env_method`
  pipe round-trip to every subprocess worker per vector step. Fix: push the
  update at `_on_rollout_end` (per-rollout granularity is plenty for a
  schedule spanning hundreds of thousands of steps), or only when the scale
  moves by ≥ epsilon.

- **Close GUI schema gaps.** `get_schema` (`gui/app.py:161-465`) offers no
  `self_play` / `reward_annealing` / `curriculum` groups, so the documented
  preferred arena path cannot be configured in the wizard (compounding the
  template-stripping bug above); arena `battle_rules` omits
  `sensing_radius` and `attack_falloff`; `create_config` accepts an empty
  `experiment_name` and writes a hidden `.yaml` file. Fix: add the missing
  groups/fields with the bundled-YAML defaults; reject empty/whitespace
  names.

- **Warn on unknown walker `reward`/`termination` keys.**
  `WalkerBulletEnv.__init__` filters kwargs by `__annotations__`
  (`walker_bullet.py:54-68`) with no warning, so a typo'd YAML key (e.g.
  `foward_velocity_weight`) silently trains with defaults. The arena already
  warns for unknown `battle_rules` keys (`arena_parallel.py:53-59`); mirror
  that. Unknown `sim` keys stay tolerated per the documented back-compat
  policy.

- **Guard multi-seed oversubscription.** `run_multi_seed` defaults
  `max_workers = min(len(seeds), cpu_count)` while each seed's training may
  itself spawn `num_envs: 24` SubprocVecEnv workers (5 × 24 processes on 24
  cores). Fix: default to 1 worker when `training.num_envs > 1` and warn
  otherwise; fold the guidance into the Priority 2 (2026-06-30) presets item.

- **Treat arena mirror-eval scoring as broken, not just limited.** Shared-
  policy `evaluate` of a zero-sum arena sums to ≈ 0 by construction, so
  `run_morphology_search` (which ranks trials by `mean_return`) effectively
  selects noise. The existing Priority 3 (2026-06-30) item "let morph-search
  score candidates by tournament Elo" is the fix; it should be scheduled as
  a correctness repair rather than a feature.

## Next development roadmap (2026-06-30)

This section was the execution roadmap until the 2026-07-05 review above,
which now takes ordering precedence. The unfinished items below remain
valid; the older review sections further down remain as evidence and
background.

### Priority 0: unblock known correctness bugs
- ~~End N-agent self-play episodes when the live policy is eliminated.~~ **Done
  (2026-07-12)** — the arena retains eliminated agents for PettingZoo's fixed
  population, but the single-agent self-play wrapper now ends the learner's SB3
  episode immediately. A three-agent regression test confirms the underlying
  free-for-all may continue while the learner receives a terminal transition.

### Completed Runtime Controls (2026-07-12)
- ~~Set bundled MLP experiment configs to CPU.~~ **Done** — all shipped MLP
  configs and GUI defaults use `device: cpu`; `auto` and CUDA remain available
  for larger policies.
- ~~Pin floating direct runtime/dev dependency ranges.~~ **Done** —
  `pyproject.toml` now uses the exact versions in `requirements-lock.txt`.
- ~~Add optional PyTorch-thread and worker-start controls.~~ **Done** —
  `training.torch_num_threads` and `training.worker_start_method` are validated
  and applied to PPO/SubprocVecEnv only when present in a config.
- ~~Fix `validate_experiment_config()` for arena self-play parallelism.~~
  **Done (2026-06-30)** — validation now allows `training.num_envs > 1` only
  when `self_play.enabled: true`, while preserving the shared-policy SuperSuit
  guard. Regression tests cover the rejected shared-policy case, the accepted
  self-play case, and the bundled `organisms_fight_arena.yaml` config.
- ~~Align GUI arena defaults with the same rule.~~ **Done (2026-06-30)** — the
  wizard still defaults arena `num_envs` to 1 for shared-policy safety, but no
  longer caps the field at 1 so loaded self-play templates can keep their
  parallel worker count.
- ~~Refresh stale README schema facts while touching docs.~~ **Done
  (2026-06-30)** — README now points at the current walker dimensions, current
  arena replay/support limitations, and the self-play parallel arena path.

### Priority 1: walker training stability and learning quality
- ~~Fix the NaN PPO action mean at high `num_envs`.~~ **Done (2026-07-02)** —
  root cause was `WalkerBulletEnv.step()` feeding raw, un-sanitized
  `pos`/`quat`/`lin_vel` from PyBullet into reward and termination (only the
  observation copy was `nan_to_num`-guarded). A rare solver-divergence frame
  (more likely to be hit per wall-clock second at higher parallelism) could
  emit a `NaN` reward that survived uncaught until `max_steps` truncation,
  poisoning GAE for that episode and NaN-ing the next PPO gradient update.
  `step()` now detects non-finite physics reads, sanitizes them, and treats
  divergence as an immediate fall (torso-contact-equivalent) so it can never
  linger. Verified with a smoke run reproducing the original report exactly
  (`num_envs: 24`, one 49,152-step rollout, `training.check_nans: true`) —
  first PPO update now completes cleanly.
- ~~Establish a stable high-throughput walker preset before changing
  defaults.~~ **Done (2026-07-03)** — with the NaN-source fixed, benchmarked 5
  presets at `num_envs: 24` (this machine's CPU-saturating value) for 4 PPO
  rollouts each, `training.check_nans: true`, verifying the saved policy's
  action-distribution mean stayed finite after every update (not just
  `explained_variance`/loss sanity):

  | preset | n_steps | batch_size | lr | ent_coef | clip_range | FPS | stable |
  |---|---|---|---|---|---|---|---|
  | current_default | 2048 | 512 | 3e-4 | 0.005 | 0.2 | **5605** | yes |
  | lower_lr_tighter_clip | 2048 | 512 | 1e-4 | 0.005 | 0.2 | 4948 | yes |
  | hybrid_1024_256 | 1024 | 256 | 3e-4 | 0.005 | 0.2 | 4592 | yes |
  | pybullet_locomotion | 512 | 128 | 3e-4 | 0.0 | 0.2 | 3728 | yes |
  | rlzoo_bipedal | 2048 | 64 | 3e-4 | 0.0 | 0.18 | 3159 | yes |

  All five were numerically stable (the NaN fix holds across hyperparameter
  choices, not just the one config originally reported). `batch_size` is the
  dominant throughput lever on this CPU-only setup: SB3's reported FPS
  includes the PPO update, and small batches (64, 128) mean far more
  minibatch gradient steps per rollout (RL-Zoo's `batch_size: 64` runs ~7,680
  backprop steps per rollout here vs. `current_default`'s 960), so they lose
  on wall-clock throughput despite identical env-step cost. `current_default`
  (already the shipped hyperparameters, just previously capped at
  `num_envs: 8`) was both the fastest and stable, so `robot_walk_basic.yaml`
  and `my_walker.yaml` now ship with `num_envs: 24` instead of `8` — no other
  hyperparameters changed. RL-Zoo/PyBullet-style small-batch presets remain
  documented above as options, not defaults, given the throughput cost on
  this hardware.
- ~~Add a best-checkpoint evaluation path.~~ **Done (2026-07-12)** — walker
  configs can enable `evaluation.best_model`; a separate normalized eval env
  selects `best_model.zip` and writes `best_model_vecnormalize.pkl` alongside
  it. Integration coverage verifies both artifacts.
- Rebalance the default walker reward so standing still is not overpaid. Test
  lower `alive_bonus`, stronger forward-progress incentives, energy/torque
  costs, and curriculum schedules that ramp target velocity and perturbations
  after balance is learned.
- ~~Add walker observation v2 with foot contacts and a coordinate-free mode.~~
  **Done (2026-07-12)** — `environment.observation.version: v2` adds right/left
  contact bits, and `coordinate_free: true` drops global x/y. v1 remains for
  existing checkpoints; cross-version resume is intentionally rejected.

### Priority 2: throughput and experiment operations
- ~~Produce documented local training presets for this 24-core machine.~~
  **Done (2026-07-12)** — `docs/training_presets.md` covers quick smoke,
  reliable overnight walker, high-throughput walker, arena self-play, and
  multi-seed evaluation, including measured FPS and worker guidance.
- ~~Add optional `training.torch_num_threads` and `training.worker_start_method`
  controls.~~ **Done (2026-07-12)** — the controls are validated and the
  `walker_smoke_cpu` CLI run completed with `torch_num_threads: 1` and
  `worker_start_method: spawn` at roughly 1,300 FPS.
- ~~Replace GUI tuning/status IPC with a durable queue.~~ **Done (2026-07-12)**
  — the SQLite WAL run registry persists status/metrics events and tuning
  commands, while retaining the polling API.
- ~~Add a first-class run registry.~~ **Done (2026-07-12)** — immutable run IDs,
  configs, artifacts, sidecars, status events, and resume lineage are stored in
  `run_registry.sqlite3`; comparison/analysis views remain future GUI work.
- ~~Make sweeps and benchmark matrices resumable.~~ **Done (2026-07-12)** —
  sweeps persist fingerprinted `sweep_summary/state.json` plus per-run
  `completion.json` markers and resume with `--resume-incomplete`; the benchmark
  persists each completed regime in a matching JSON state file and resumes with
  `--resume`.

### Priority 3: feature additions
- Walker curricula: add terrain presets (`flat`, `uneven`, `obstacle/stump`,
  `push_recovery`) and example configs that progress from balance to locomotion
  to perturbation recovery.
- ~~Add SAC and TD3 walker baselines.~~ **Done (2026-07-12)** —
  `training.algorithm` selects PPO/SAC/TD3 through the shared runner; SAC/TD3
  are validated as walker-only and ship with CPU v2 baseline configs.
- ~~Add arena collision, energy/food mechanics, and speed/size tradeoffs.~~
  **Done (2026-07-12)** — organisms now have size-scaled collision separation,
  movement/attack energy costs, food pickup and deterministic respawn, and
  inverse size/speed scaling. The policy receives energy, size, and nearest-food
  features in the expanded 13D arena observation; prior 8D checkpoints are
  incompatible.
- ~~Extend arena tooling beyond head-to-head and score morphology by Elo.~~
  **Done (2026-07-12)** — slot-ordered N-agent eval, rotating N-agent
  tournaments, multi-opponent replay, and `morphology_search.scoring:
  tournament_elo` are implemented.
- ~~Add GUI analysis views backed by the run registry.~~ **Done (2026-07-12)**
  — the Analysis tab compares persisted run metrics/statuses, lists recorded
  artifacts with best-checkpoint emphasis, launches replay jobs, and starts
  asynchronous league-snapshot Elo ratings without blocking training.

### Validation expectations for roadmap work
- Bug fixes get focused regression tests plus the narrowest relevant CLI smoke
  check.
- Walker environment, reward, termination, or observation changes get
  `pytest`, a reset/step smoke test, and a short training or replay smoke.
- Training-pipeline or benchmark changes get `pytest`, `ruff`, repo policy, and
  one short command-line run that exercises the changed path.
- GUI workflow changes get API tests plus browser verification for the affected
  desktop and mobile flows.

## Bipedal walker training review plan (2026-06-29)

### Confirmed bugs / correctness gaps
- ~~Scale checkpoint cadence by `training.num_envs`.~~ **Done** — SB3 callback calls advance by vector-env step, so `training.checkpoint_every` must be converted from environment timesteps to callback calls. Without this, `robot_walk_basic` with `num_envs: 8` saved every 400k env steps instead of every 50k.
- ~~Load `VecNormalize` statistics in `render-replay` for `walker_bullet`, matching `eval_runner.evaluate()`.~~ **Done** — Gymnasium replay now runs through `DummyVecEnv` and loads the model-specific/legacy VecNormalize sidecar when present before `model.predict()`.
- ~~Penalize all terminal fall modes consistently.~~ **Done** — low-height, torso-contact, and max-height terminal paths now pass `fell=True` to `WalkerReward.compute()`; pure truncation remains unpenalized.
- ~~Make action clipping explicit in `WalkerBulletEnv.step()` before both control and reward accounting.~~ **Done** — env step clips to `[-1, 1]` before action-latency buffering, dynamics, and reward calculation.
- ~~Either wire or remove `environment.sim.body_half_extents`.~~ **Done** — removed the ignored knob from GUI schema, bundled walker YAMLs, README, and tests; old configs that contain it still load because unknown `sim` keys are tolerated.
- ~~Add NaN diagnostics for walker training.~~ **Done** — optional `training.check_nans: true` wraps the vector env in SB3 `VecCheckNan` and fails fast on NaN/Inf values without changing defaults.
- ~~Add regression tests for reward/termination edge cases, clipped action penalty, and normalized replay.~~ **Done** — targeted tests cover low-height/max-height terminal penalties, reward-side clipped actions, normalized walker replay, hidden geometry schema, and `training.check_nans` validation.

### Training efficiency / operator defaults
- Benchmark before raising `robot_walk_basic.training.num_envs` toward the local CPU-saturating value. A smoke run with `num_envs: 24` collected one 49,152-step rollout at ~3,100 FPS but produced NaN PPO action means on the first update, so higher parallelism needs optimizer/reward-scale validation before becoming the default.
- Benchmark `num_envs` x `n_steps` x `batch_size` on this machine with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, then record recommended presets. Compare current `24 x 2048 x 512` against RL Zoo-style BipedalWalker PPO settings (`n_envs: 32`, `n_steps: 2048`, `batch_size: 64`, `gamma: 0.999`, `ent_coef: 0.0`, `clip_range: 0.18`) and PyBullet locomotion defaults (`n_envs: 16`, `n_steps: 512`, `batch_size: 128`, ReLU 256x256 policy, `use_sde: true`).
- Add optional `training.torch_num_threads` / `training.worker_start_method` controls after benchmarking. The runner sets BLAS env vars for subprocesses, but PyTorch CPU update threading is still implicit.
- Add an `EvalCallback`/best-checkpoint path with a separately normalized eval env so long walker runs keep the best policy, not only periodic and final checkpoints.

### Walker environment and learning features
- Add foot contact indicators to the observation. Gymnasium BipedalWalker exposes leg ground contact, and contact phase is useful for gait learning; this will require an observation shape/version change and compatibility notes for old checkpoints.
- Add a configurable observation mode that can remove absolute x/y position from policy input while retaining velocity and height. Gymnasium BipedalWalker deliberately omits coordinates; keeping unbounded position in a flat-plane task may encourage time/progress overfitting and makes normalization drift with long episodes.
- Add terrain variation presets: flat, uneven, obstacle/stump, and push-recovery perturbations. This aligns the custom PyBullet walker more closely with normal/hardcore BipedalWalker training curricula.
- Rebalance the default reward so survival does not dominate locomotion. Current `alive_bonus: 5.0` over 800 steps can make standing still more attractive than learning gait; compare against forward-progress-plus-energy-cost reward structures and use curriculum to ramp target speed and posture penalties.
- Add curriculum examples for walker: start with balance/short horizon/low target velocity, then increase `target_velocity`, horizon, terrain difficulty, perturbations, and domain randomization.
- Add optional SAC/TD3 experiment configs for the continuous-control walker baseline. PPO remains the default, but off-policy algorithms are useful comparison points for sample efficiency.
