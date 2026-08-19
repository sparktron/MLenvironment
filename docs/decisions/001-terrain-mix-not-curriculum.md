# ADR-001: Vary terrain by per-episode sampling, not by curriculum

Date: 2026-08-11
Status: Accepted

## Context

Walker policies trained on a single terrain were suspected of specialising
rather than generalising. Varying terrain during training was the obvious
response, and `CurriculumCallback` was the obvious mechanism — it already
existed for progressively adjusting training parameters.

It cannot do the job. `update_live_params` has **no `terrain` section**, so a
curriculum configured to change terrain silently changes nothing: the run
completes, the numbers look plausible, and the experiment is void.

## Decision

Introduce `terrain.mix` — a list of full terrain specs, each with its params
and an optional relative `weight`. One is drawn per `reset()` from the env's
seeded RNG.

## Alternatives considered

| Option | Why not |
|---|---|
| **Extend `CurriculumCallback` with a `terrain` section** | Curriculum implies a *schedule* — easy to hard. That bakes in an ordering assumption the evidence did not support, and progressive schedules risk catastrophic forgetting of earlier terrains. Sampling makes every episode a fresh draw. |
| **One terrain per training run, ensemble afterwards** | Trains specialists, which is the problem being solved, not the solution. |
| **Rebuild the whole world per episode** | Correct but expensive. `_build_world()` per reset would forfeit the existing reset optimisation. |

## Consequences

**Accepted:** each `reset()` draws a terrain, so a seeded run replays the same
sequence while rollout workers deliberately diverge — a batch therefore mixes
terrains. Sampling rebuilds **only** the static terrain boxes; `_build_world()`
still runs once per env, preserving the reset optimisation. Measured cost:
**+0.23 ms per reset, ~0.04% of an episode.**

The episode's terrain is reported in `info["terrain_preset"]`, without which a
mixed run's aggregate metrics hide *which* terrain caused a fall — the metric
would be uninterpretable.

**Costs accepted:** two config shapes to support (with and without `mix`), and a
new failure mode — an explicit but empty `mix: []`, which is treated as an
error rather than silently meaning "no mix", precisely to avoid repeating the
silent-no-op failure that motivated this ADR.

**Outcome so far: the hypothesis did not hold.** v11 (uneven continuation,
2026-08-10) produced specialisation rather than generalisation, and v12 (mixed
terrain, 2026-08-13) was recorded as refuted. The mechanism works as designed;
what it was meant to demonstrate remains unproven. This ADR is about the
mechanism, and it stands — the open research question does not.

## Evidence / references

- Commit 2026-08-11, "Add per-episode terrain sampling (terrain.mix)"
- Commit `aecf61d` 2026-08-13, "Record v12 mixed-terrain result: the hypothesis is refuted"
- `docs/walker_learning_history.md` — v9, v11, v12 entries
- `AGENTS.md` — "Walker env quirks", "Important invariants"

## Follow-up

- The generalisation question is open. Neither continuation nor mixing produced
  it; the next hypothesis needs to be something else.
- `CurriculumCallback`'s inability to change terrain is a latent trap for anyone
  who has not read `AGENTS.md`. Consider making a `terrain` key under curriculum
  a hard error rather than a silent no-op.
