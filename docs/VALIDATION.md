# Validation

**Updated:** 2026-08-17 · **Commit:** `aecf61d` · **Branch:** `master`

> **The distinction this document exists to protect:** a green test suite means
> the code runs. It does not mean a learning result is real. In RL the common
> failure is a *statistically* invalid claim from a passing pipeline — one lucky
> seed, a reward that rewards the wrong thing, an env that permits a degenerate
> exploit. "Tests pass" and "the hypothesis held across seeds" are different
> claims and are kept in different sections here.

---

## Verified

### Unit tests

`pytest -q --cov=src/rl_framework --cov-fail-under=60`. Coverage floor of 60%
is a CI gate — note that is a *floor*, and 40% of the package may be uncovered.

**Status: suite present and CI-gated. Not executed in this session.**

### Integration tests

No separate integration tier. Functional verification is by convention
(`AGENTS.md` Workflow): a big change — env behaviour, training pipeline,
reward, or termination — requires a smoke test of env reset/step, a few hundred
training timesteps, or an eval/replay of an existing checkpoint. **This is a
convention, not an enforced gate.**

### Static analysis

| Tool | Scope | Blocking |
|---|---|---|
| `ruff check src tests scripts` | lint | yes |
| `python scripts/check_repo_policy.py` | no tracked `__pycache__`/`.venv`/`.egg-info`; lockfile completeness vs `pyproject.toml` | yes |
| `mypy src --ignore-missing-imports` | types | **no — non-blocking** |

Type errors can therefore reach `master`.

### CI

Runs the above. Installs from `requirements-lock.txt`, not `pip install -e .`;
a lockfile missing any direct dependency fails the build.

### Host simulation

PyBullet *is* the simulation — there is no other execution target. Everything
below is simulated by definition; "host simulation" and "target" do not
separate here the way they do for firmware.

What is genuinely verified by running the sim:

- **Multi-seed learning results**, recorded in
  `docs/walker_learning_history.md` (append-only, canonical). Through v12,
  including refuted hypotheses:

| Version | Date | Result |
|---|---|---|
| v9 flat, three seeds | 2026-08-05 | 2/3 pass gate v3; all three learned to run |
| v11 uneven continuation | 2026-08-10 | **specialisation, not generalisation** |
| v12 mixed terrain | 2026-08-13 | **hypothesis refuted** |

- **Terrain-sampling cost**, measured: +0.23 ms/reset, ~0.04% of an episode.
- **Arena ranking** via Bradley-Terry Elo over round-robin checkpoints.

### Target hardware

**Not applicable.** No physical robot. Every result is simulation. A policy that
walks in PyBullet has *not* been shown to walk anywhere else, and the sim-to-real
gap for contact-rich locomotion is large.

### Field testing

**Not applicable** for the same reason.

---

## Not yet verified

- **Whether the suite passes at `aecf61d`** — not run here.
- **What actually produces cross-terrain generalisation.** v11 and v12 both
  failed. The open question is unanswered, not merely untested.
- **Whether the 60% coverage floor covers the parts that matter** — env
  dynamics, reward, termination.
- **Purpose of the root-level `rl_framework/__init__.py`**, distinct from
  `src/rl_framework/`. Possibly a stale import shim.
- **Whether `.claude/skills/` and `.agents/skills/` have drifted.**
  Byte-identical on 2026-08-17; nothing enforces that.
- **Reproducibility of runs in `outputs/`** — gitignored, so the 75 run
  directories cannot be tied back to commits from the repository alone.

## Known environment limitations

- **Seeded ≠ deterministic.** Seeding fixes the terrain sequence per rank, but
  `SubprocVecEnv` workers diverge deliberately so a batch mixes terrains.
  Reproducing an exact run requires matching `num_envs` and worker count too.
- **A misconfiguration can look like a successful run.** A flattened
  `control: null` is silently tolerated and falls back to defaults. There is no
  validation error — the run completes and the numbers are wrong.
- **`CurriculumCallback` cannot change terrain.** `update_live_params` has no
  `terrain` section, so a curriculum-based terrain experiment silently does
  nothing.
- **`SubprocVecEnv` cannot be smoke-tested from a stdin heredoc** —
  multiprocessing re-runs the parent and fails on `<stdin>`.
- **Shared-policy arena is single-process only**; a guard rejects `num_envs > 1`.
- **Throughput is machine-specific.** `num_envs: 24` is tuned to this host's 24
  physical cores; results elsewhere will differ in wall-clock, not in outcome.
- **`outputs/` is gitignored and disposable.** Evidence that lives only there is
  not evidence.

## Required release qualification

This is a research framework, not a shipped product, so "release" means
**publishing a learning claim** — into `walker_learning_history.md`, a report,
or anywhere it will be believed later.

| # | Requirement | Status |
|---|---|---|
| 1 | `pytest` with the 60% coverage gate passes | not run |
| 2 | `ruff` clean, `check_repo_policy.py` passes | not run |
| 3 | Claim supported by **multi-seed** evidence, not a single run | enforced by convention |
| 4 | Config recorded — the exact experiment YAML, not a description | — |
| 5 | Result written to `docs/walker_learning_history.md` **whether or not the hypothesis held** | practised; v11 and v12 are both negative results |
| 6 | Degenerate solutions ruled out (e.g. the PD-jackhammer self-launch that `max_height: 1.5` guards) | manual review |
| 7 | Terrain reported per episode via `info["terrain_preset"]` for mixed runs — aggregates otherwise hide which terrain caused failures | available |

Requirement 5 is the one that makes this ledger worth trusting: a history that
records only successes cannot be used to reason about what does not work.
