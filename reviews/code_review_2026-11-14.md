# Repository Audit: `MLenvironment` (2026-11-14)

Reviewed at `837b50d` plus an uncommitted working tree (live terrain parameters
in `walker_bullet.py`, in progress by another session; not touched here).
Labels: **[Measured]** = I ran it this session; **[Read]** = from code;
**[Inference]** = my reasoning from measured facts, not itself tested.

## 1. Executive summary

- **What it is:** an SB3/PyBullet RL framework with a bipedal walker task, an
  N-agent arena task with self-play, a single CLI and a Flask GUI.
- **Stack:** Python 3.11, Stable-Baselines3 (PPO/SAC/TD3), PyBullet,
  PettingZoo/SuperSuit, Flask. About 14.3k lines of source, 529 tests.
- **Maturity:** research framework with unusually good engineering hygiene and
  an honest experiment ledger. Not the weak point.
- **Gates [Measured]:** 529 tests pass, 83.79% coverage, ruff clean, repo policy
  clean, mypy reports 6 errors (advisory).

**Top findings**

1. **The push-recovery terrain does not apply a lateral push.** It applies a yaw
   torque that grows with distance walked. The "100% falls on push recovery"
   result in every checkpoint is at least partly an environment bug.
2. **The "uneven" terrain is three 2.5 cm raised platforms, not undulation.**
   Half its slabs are buried under the ground plane. It is effectively a
   2.5 cm step-edge test, which sits exactly on the measured 2–3 cm obstacle
   cliff.
3. **Terrain exists only in the first 3.6 m.** v9 policies walk about 13.7 m per
   episode, so most of every "terrain" episode is flat ground.
4. **The foot-clearance metric measures the foot's centre of mass, not its
   lowest point.** This is a candidate answer to the open "why is 5 cm
   clearance never reached" question.
5. **Statistical power is low for the claims being made**: 3 seeds, 20 episodes,
   binary gates, and a gate that was revised four times while results came in.
6. **The May review's three findings were never implemented.** The June arena
   review's fixes were, and they are tested.

Findings 1–3 mean the v11 and v12 negative results on generalisation should be
treated as unproven rather than refuted until the terrains are fixed and the
transfer suite is re-run on existing checkpoints.

## 2. Were the last review's changes implemented?

### `reviews/code_review_2026-05-09_to_2026-05-13.md` — not implemented

| # | Finding | Status | Evidence [Read] |
|---|---|---|---|
| 1 | Clamp negative `since` on `/api/train/frames` | **Open** | `gui/app.py:398` parses `int(...)` with no lower bound |
| 2 | Bound the client frame buffer | **Open** | `gui/static/app.js:1306` pushes every frame; no eviction anywhere |
| 3 | Count and expose frame-capture failures | **Open** | `core/training/callbacks/frame_capture_callback.py:150` still prints only when `verbose >= 1`; no counter, nothing in status |

None of the three is mentioned as done or deferred in `docs/open_items_todo.md`,
`docs/STATUS.md` or `docs/ui_roadmap.md`. STATUS says historical review findings
were folded into a completed summary; these were not.

### `docs/organism_env_review_2026-06-09.md` — implemented and tested

Spot-checked in current code, all present: seeded spawn jitter (B1), draw
classification and `draw_rate` (B2), norm-clamped movement (B3), visibility
flag (B4), unknown-key warnings (B5), health coupled to growth (B6), numeric
league-file filter (B7), inert step-after-done (B8), single backend switch
(B9), public `observe()` (B11). Regression tests exist for the four "testing
gaps" the review listed (`test_arena_spawn_jitter_differs_across_seeds`,
`test_play_episode_classifies_simultaneous_ko_as_draw`,
`test_league_sampler_skips_non_numeric_snapshots`,
`test_arena_step_after_episode_end_is_inert`).

One leftover: the comment in `LeagueSampler._league_files` says the zip lands
before the sidecar. `SelfPlayCallback._save_snapshot` now writes the sidecar
first and renames the zip last, so the comment is stale. The behaviour is
correct.

### Known problems carried in STATUS/VALIDATION

- `CurriculumCallback` cannot change terrain: being fixed in the uncommitted
  tree. `837b50d` already made unknown live parameters raise, and the curriculum
  callback no longer swallows the exception.
- Flattened `control: null` silently falling back to defaults: **still open**.
  `WalkerBulletEnv.__init__` uses `get_section`, and
  `physics/locomotion/validation.py` does not check `sim.control`.

## 3. Confirmed bugs

### 3.1 Push recovery applies a torque about the world origin

- **Location:** `physics/locomotion/walker_bullet.py:916`
- **What:** `applyExternalForce(..., posObj=[0,0,0], flags=p.WORLD_FRAME)`. In
  world frame the position is a world coordinate, so the force acts at the
  world origin, not at the torso. The lever arm is the robot's x position.
- **Measured:** one 180 N push with the robot placed at x metres:

  | x (m) | yaw-rate change (rad/s) | lateral velocity change (m/s) |
  |---:|---:|---:|
  | 0 | +0.007 | −0.021 |
  | 2 | +1.17 | −0.014 |
  | 5 | +4.54 | +0.000 |
  | 10 | +11.40 | +0.052 |

- **Second defect:** the force is applied once before `frame_skip` physics
  ticks, and PyBullet clears external forces after one tick. The intended
  impulse is 180 N × 1/240 s = 0.75 N·s, about 0.011 m/s on a 65.9 kg body.
  The lateral push the terrain is named for is negligible.
- **Why it matters:** the walker has no yaw or roll actuator (all ten joints
  rotate about Y), so it cannot resist a yaw kick. The ledger records 100%
  falls on push recovery for every checkpoint at around 390–455 steps and
  7–8 m. Pushes fire every 120 steps from step 60, so the push at step 420
  lands near 7 m, where the kick is roughly 7 rad/s. **[Inference]** that is
  what ends those episodes.
- **Fix:** apply at the torso with `p.LINK_FRAME` and `posObj=[0,0,0]` (rotate
  the force into the link frame), or pass the current base position in world
  frame. Re-apply on each of the `frame_skip` ticks, or specify the push as an
  impulse and duration. Add a test asserting the yaw-rate change is independent
  of x and the lateral velocity change matches impulse/mass.

### 3.2 "Uneven" terrain is half buried

- **Location:** `walker_bullet.py`, `_build_terrain`, `uneven` branch (line 535)
- **Measured:** slab top surfaces are +0.025, −0.025, +0.025, −0.025, +0.025 m.
  The ground plane is at z = 0, so the two negative slabs are below it and have
  no effect.
- **Result:** three 64 cm platforms, 2.5 cm high, with flat ground between and
  after them. Six 2.5 cm edges in total.
- **Why it matters:** the obstacle ladder put the failure cliff at 2–3 cm.
  **[Inference]** "uneven" and "obstacles" are the same test at different
  heights, which is why v11 moved uneven but nothing transferred to 10 cm
  obstacles, and why the ledger's "terrain types compete rather than compose"
  conclusion may be about step height rather than terrain type.
- **Fix:** build real undulation (a heightfield via `GEOM_HEIGHTFIELD`, or
  slabs that are all above the plane with varying heights), and randomise it
  per episode from the env RNG.

### 3.3 Terrain covers only the first 3.6 m and never varies

- **Location:** same function. Uneven slabs span x 0.48–3.52 m; obstacles sit at
  x = 1.2, 2.4, 3.6 m. Positions are fixed constants.
- **Why it matters:** a policy walking 13.7 m per episode spends about a quarter
  of a mixed-terrain episode on terrain. The 10M-step v12 budget "split four
  ways" bought far less terrain experience than the config comment assumed, and
  a policy that survives the first 3.6 m scores as if it had handled the
  terrain for 800 steps. Fixed positions also make the task memorisable through
  the gait clock.
- **Fix:** tile terrain along the whole reachable track (or generate it ahead of
  the robot), and jitter spacing and height per episode.

### 3.4 Best-model evaluation workers replay identical episodes

- **Location:** `core/training/sb3_runner.py:94–109`
- **Measured:** two envs built with `_build_single_env(rank=0)` and `rank=1`
  return identical first observations for three consecutive resets. `rank` only
  names the Monitor file; nothing seeds the eval env, and SB3 seeds only the
  training env.
- **Impact:** with a deterministic policy, N workers × M episodes yields M
  distinct episodes. Latent today, because no shipped config enables
  `evaluation.best_model`.
- **Fix:** build it with `make_seeded_vec_env` like `eval_runner` does.

## 4. Likely bugs and risk areas

- **Foot clearance is measured at the link centre of mass [Read].**
  `_foot_positions` returns `getLinkState(...)[0]`, which is the COM, placed
  4 cm forward of the ankle by `linkInertialFramePositions`. The foot box is
  17 cm long. A foot pitched toe-down by 0.3 rad has its toe about 2.5–3.7 cm
  below what the metric reports. Separately, the foot collision box is centred
  on the ankle while the visual and inertial frames are shifted 4 cm forward,
  so videos show the foot where it does not collide. **[Inference]** this is a
  plausible reason policies "deliver 21–26 mm" against a 50 mm target and why
  clearance fails to rank seeds on obstacles. Check: log the minimum z of the
  four sole corners during swing next to the current metric on a v9 checkpoint.
  No training needed.
- **Only the feet and torso collide [Read]** (`walker_bullet.py:330`). Thighs,
  shins, arms and head pass through the ground and through obstacles. A fall
  registers only when the torso box touches. This makes obstacle results
  optimistic (shins cannot catch) and makes kneeling a non-event.
- **League prune race [Possible]** (`games/arena/self_play_env_wrapper.py`,
  `sample`). A worker can stat the directory, then the main process prunes the
  oldest snapshot, then the worker calls `PPO.load` on the deleted file and the
  env raises at reset. The window is small. Catch `FileNotFoundError` and
  resample.
- **Config typos warn instead of failing [Read].** Unknown `reward` and
  `termination` keys emit a warning that is easy to miss in a 24-worker log;
  `validation.py` range-checks 10 of the 30+ reward fields; an invalid
  `velocity_mode` raises only at the first `step()`. The repo's own invariant
  list calls this failure mode out for `control: null`.
- **mypy errors point at real `None` arithmetic [Measured]:**
  `gait.py:363` (`int - None`) and `walker_diagnostics.py:804` (`None + int`).
  Both are advisory in CI and worth reading before dismissing.

## 5. Methodology assessment

**What is strong.** The ledger records refuted hypotheses and corrects itself
(the obstacle ladder overturned its own earlier claim the same day). Telemetry
before training is the right instinct. Reward terms are decomposed and logged.
Truncation is separated from termination. Checkpoints carry their own
normaliser. Regression fixtures pin rollouts and legacy checkpoints.

**What will limit success.**

1. **Evaluation instruments have been wrong more than once, and the env was not
   audited.** The transfer suite could not see progress at 10 cm; this review
   adds three terrain defects. Every terrain result since v9 rests on them.
2. **Three seeds and 20 episodes cannot support pass/fail gates.** A fall rate
   from 20 episodes has a 95% interval of roughly ±20 points at 50% (so "5%,
   20%, 15%" across seeds is one distribution). `multi_seed_runner` aggregates
   mean return only, with population standard deviation. Evaluation is cheap
   here: 100+ episodes per cell costs minutes.
3. **Gate drift.** Gates v1 to v4 were revised after seeing results (admitting
   running, changing the slip metric). The ledger names this itself. Thresholds
   tuned on the same three seeds they judge are not a held-out test.
4. **Reward complexity.** 18 terms and a clock were added to force a walk, and
   the result was still a run. Each term was "satisfied by a non-walking gait".
   That pattern usually means the objective is under-constrained by shaping and
   would be better served by a reference (motion-imitation or a phase-indexed
   foot-height target), which the phase clock already half-implements.
5. **The morphology cannot do what two of the four terrains ask.** Sagittal-only
   joints give no active roll or yaw control. The ledger correctly rejected this
   as the cause of short strides, but lateral push recovery is a different
   demand.
6. **PPO setup is narrow [Read].** The network is SB3's default 64×64 and is not
   configurable (`build_model` passes only `log_std_init`). There is no reward
   normalisation, no `target_kl`, and no entropy schedule. For a 37-D
   observation, 10-D action humanoid, capacity is an untested variable.
7. **Throughput [Measured].** A full env step costs 808 µs against 416 µs for
   physics alone. Roughly half of each step is contact and link-state queries
   (`getContactPoints` is called five times per step, `getLinkState` three).
   One contact query per foot, reused, would raise the 5,500 fps figure
   materially.
8. **Self-play is minimal [Read].** The live agent is always `agent_0`, frozen
   opponents always act deterministically, and the league is a 10-snapshot FIFO
   that discards its oldest members. That invites cycling and makes the learner
   exploit deterministic opponents. Bradley-Terry ranking exists but is not fed
   back into opponent sampling.

## 6. Test coverage gaps

- **Critical:** no test asserts terrain geometry (surface heights above the
  plane, extent along the track) or push physics (direction, magnitude,
  position-independence). These are the three confirmed bugs.
- **Medium:** `arena_eval.py` 58%, `games/arena/quality_study.py` 59%,
  `walker_diagnostics.py` 44%. The last one holds the transfer suite that the
  ledger's conclusions depend on.
- **Long-tail:** eval-env seeding across workers; league prune during sampling.

## 7. Prioritised action plan

**Phase 1 — fix the instrument (no training required)**

1. Fix the push (3.1) and add the physics test.
2. Rebuild uneven as real undulation, extend all terrain along the track,
   randomise per episode (3.2, 3.3).
3. Measure sole-corner clearance against the current metric on v9 checkpoints.
4. Re-run the transfer suite on the existing v9/v11/v12 checkpoints at 100
   episodes per cell, and record the result in the ledger, whichever way it
   goes. Mark the v11/v12 conclusions as pending until then.
5. Close the three May findings (about an hour in total).

**Phase 2 — make claims defensible**

1. Five or more seeds for any promoted claim; report bootstrap intervals and
   IQM in `multi_seed_runner`, on behaviour metrics as well as return.
2. Freeze the gate and designate held-out seeds that are never used for tuning.
3. Turn unknown config keys and `control: null` into errors.
4. Seed the best-model eval env; make mypy blocking for `physics/` and `core/`.
5. Cache contact queries in `step()`.

**Phase 3 — capability**

1. Expose `net_arch`, `target_kl` and reward normalisation; run a multi-seed
   capacity comparison (64×64 against 256×256) on v9.
2. Decide whether lateral push recovery is in scope. If it is, add hip roll
   (checkpoint-incompatible). If not, drop it from the suite.
3. Add collision shapes to shins and thighs before trusting obstacle results.
4. Replace per-property shaping with a phase-indexed swing-height reference.
5. Self-play: randomise the live slot, sample opponent actions stochastically
   for some fraction of episodes, keep a few permanent anchors in the league,
   and weight sampling by rating.

## 8. Top patch targets

1. `physics/locomotion/walker_bullet.py` — push, terrain, collision shapes,
   contact-query caching. Highest return in the repo.
2. `physics/locomotion/walker_diagnostics.py` — episode counts, intervals,
   sole-corner clearance.
3. `core/training/multi_seed_runner.py` — proper aggregation.
4. `physics/locomotion/validation.py` — strict keys.
5. `gui/app.py`, `gui/static/app.js`, `frame_capture_callback.py` — the May
   review.

## 9. Verdict

- **Solid:** code quality, tests, reproducibility plumbing, arena fixes, and
  the discipline of the ledger.
- **Fragile:** the walker environment's terrain and disturbance model, and the
  statistical basis of pass/fail conclusions.
- **Ready to support the research question today?** No. The question "what
  produces cross-terrain generalisation" cannot be answered with terrains that
  are mis-built. Fix Phase 1, re-evaluate existing checkpoints, then decide
  what to train next.

## Not verified

- I did not run any training, and did not load the v9/v11/v12 checkpoints; the
  link between the push bug and the recorded 100% falls is inference from the
  measured torque and the ledger's step counts.
- The April review (`docs/exhaustive_repo_review_2026-04-22.md`) was not
  re-audited item by item; `docs/fixes_2026-04-22.md` records its fixes.
- GUI security (auth, path handling) was not reviewed in depth this pass.
- The uncommitted terrain live-parameter change was read but not reviewed.
