---
name: eval-policy
description: Run side-by-side stochastic vs deterministic evaluation of a walker_bullet checkpoint and print summary stats (mean/std episode length, return, fall rate, peak z, forward displacement). Use when the user asks how a trained policy performs, to compare two checkpoints, or to diagnose whether a policy is learning a real gait vs a reward hack.
disable-model-invocation: true
---

# eval-policy

The single most useful diagnostic on this project. PPO policies here often have a **huge gap between stochastic and deterministic eval** — a fact that's invisible from `rollout/ep_rew_mean` alone. Always run both.

## Inputs (`$ARGUMENTS`)

```
/eval-policy <checkpoint.zip> [episodes=20] [config_name=my_walker]
```

## Procedure

1. Activate the venv: `source .venv/bin/activate`.
2. Load the YAML, build a `DummyVecEnv`, load sibling `vecnormalize.pkl` (`training=False, norm_reward=False`).
3. Load the PPO model with `device="cpu"`.
4. For each mode in `["deterministic", "stochastic"]`:
   - Run `episodes` rollouts (default 20). For each: track episode length, total return (sum of step rewards), `info["torso_contact"]` at end, peak torso z (read from raw env obs `vec.envs[0]._get_obs()[2]`), final `info["x_position"]` minus initial.
5. Print a two-column comparison table with these per-mode aggregates: `ep_len_mean ± std`, `return_mean ± std`, `fall_rate` (fraction with `torso_contact=True` at end), `peak_z_mean`, `Δx_mean ± std`.
6. Print a **zero-action baseline** row at the top of the table (same env, `action=np.zeros((1,10))`, 5 episodes is enough). This is the "do nothing" floor — a learned policy that scores worse than this is broken.
7. End with a one-sentence verdict: which mode is better and by how much, and whether the policy looks like (a) learned walking, (b) reward hack (huge peak_z), (c) deterministic collapse (stochastic >> deterministic), or (d) untrained-equivalent (worse than zero baseline).

## Gotchas

- `vec.envs[0]._get_obs()` reads raw (unnormalised) observation — needed because `VecNormalize` wraps the obs the model sees.
- For the verdict heuristic: peak_z > 1.4 m (near the `max_height: 1.5` cap) suggests the policy is trying to launch; stochastic_return / deterministic_return > 2× is the deterministic-collapse signature; ep_len < zero-baseline length means worse-than-nothing.
- If episodes hit `max_steps=800` AND `peak_z < 1.0`, that's the success case — real bipedal locomotion.
