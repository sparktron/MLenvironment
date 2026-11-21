# Single-push recovery screening — 2026-11-20

The protocol and scored result are in `docs/walker_learning_history.md`.
This directory retains the numerical evidence independently of disposable
`outputs/` files.

- `seed_21.json.gz` through `seed_25.json.gz`: actual configuration, software
  versions, source/checkpoint/normalization hashes, eligibility, and every raw
  trial record. Each record includes its matched baseline, motion/contact
  metrics and sampled trajectories. JSON compression has no timestamps.
- `summary.json`: per-seed direction/strength/phase cells, episode-weighted
  summaries, and paired bootstrap intervals. Phase and direction trials from
  the same reference episode are correlated; combined summaries average
  within that episode before bootstrapping.

## Reproduce

Activate `.venv/`, then run each seed 21–25 with:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/walker_push_diagnostic.py \
  --seed 21 --out outputs/_reports/push_diagnostic/seed_21
```

Run `python scripts/walker_push_summary.py` after all five seeds finish.
The defaults implement the registered grid. The full run used diagnostic
code at `939486a`, with five parallel CPU processes. Metadata flags an
untracked v14 YAML being prepared concurrently; this experiment loads only
the committed v9 configuration recorded in that metadata. It does not use v14.

Every trial verifies exact pre-push matching (tolerance 1e-9) and a complete
24-physics-tick pulse. These checks abort on failure. Environment and policy
RNGs are reseeded for every fresh-world replay.

## Reading the records

`trajectory_samples` rows contain:

```text
step, x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz, right_contact, left_contact
```

Positions are metres, angles radians, velocities metres/second and angular
velocities radians/second; contact flags are 0/1. Samples are before actions,
at push-relative steps 0, 6, 15, 30, 60, 120, 180, when available. A terminated
episode has no sample of its automatically reset world. Missing late samples
are censored. Extrema use all pre-action samples, not only the retained sparse
samples, and exclude the terminal state to avoid sampling an auto-reset.

`fell` measures failure within the three-second observation window. A survivor
is right-censored there; it is not proof of long-term recovery. A time-to-fall
median includes only falls. A baseline is reused across conditions, so do not
pool all trial rows as independent observations.

Each individual direction/force/phase cell has at most 20 episodes. With zero
falls in 20, the Wilson upper 95% bound is still about 16%. An empirical
bootstrap interval of [0, 0] in such a cell does not imply zero population
uncertainty. Phase comparisons use the first eligible occurrence in a fixed
window, so they also compare different states and trigger times; they do not
identify an isolated causal effect of contact phase.

Detailed full-resolution traces are supplementary, regenerable files at
`outputs/_reports/push_diagnostic/seed_S/traces.jsonl.gz`.
