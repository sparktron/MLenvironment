---
name: render-policy
description: Render an mp4 video of a trained walker_bullet policy from a checkpoint. Pass the model zip path; optionally pass `deterministic` or `stochastic` (default: stochastic). Picks the longest-surviving of N=5 rollouts. Use this whenever the user asks to see what a trained policy looks like, or to compare deterministic vs stochastic behavior.
disable-model-invocation: true
---

# render-policy

Render an mp4 of a trained walker policy. The deterministic vs stochastic distinction matters: a PPO policy with un-annealed `log_std` can have a useless deterministic mean (saturated at action bounds) while the stochastic policy walks fine. **Default to stochastic** unless the user explicitly says "deterministic".

## Inputs (`$ARGUMENTS`)

```
/render-policy <checkpoint.zip> [stochastic|deterministic] [config_name=my_walker]
```

- First positional: path to the `final_model.zip` (or any `ppo_model_*.zip`) checkpoint. A sibling `vecnormalize.pkl` is required.
- Second positional (optional): `stochastic` (default) or `deterministic`.
- Third optional: which YAML config in `src/rl_framework/configs/experiments/` defines the env (default `my_walker`).

## Procedure

1. Activate the venv: `source .venv/bin/activate`.
2. Load the YAML, force `render_mode="rgb_array"`, wrap the env in `DummyVecEnv`, load the sibling `vecnormalize.pkl` with `training=False, norm_reward=False`.
3. Load the PPO model with `device="cpu"`.
4. Run 5 rollouts with the requested action mode (`deterministic=True` or `False` in `model.predict`). For each rollout, render each step with `env.render()` and stash frames in a list. Track episode length and `Δx` (final `info["x_position"]` minus first).
5. Pick the rollout with the longest length (or the one that survived without `info["torso_contact"]` if multiple full-length).
6. Write the mp4 via `imageio.v2.mimwrite(path, frames, fps=60, quality=8, macro_block_size=1)`. Output location: `outputs/<experiment_name>/seed_<seed>/videos/<mode>_<short_checkpoint_name>.mp4`.
7. Print a one-line summary per rollout (length, Δx, fell?), then the chosen file path and its size.
8. Send the file to the user with `SendUserFile` (status `normal`, with a caption describing length and Δx).

## Gotchas

- `imageio-ffmpeg` is required. If `pip show imageio-ffmpeg` errors, install both `imageio` and `imageio-ffmpeg` before rendering.
- If every rollout has `len < 30`, mention that the policy may be broken (deterministic collapse, or the env's `max_height: 1.5` cap is firing — check `info["torso_contact"]` and torso z at termination).
- The render camera now follows the robot's z, but if z exceeds the `max_height: 1.5` cap the episode terminates immediately, so an exploit-style policy will look like ~5-frame clips.
