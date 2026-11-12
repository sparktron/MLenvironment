# Restructure baseline fixtures

These fixtures were captured on 2026-11-12 from the production code at
`ae6f09b6977052b0da274cd032f633a2b1f424ee` with Python 3.11.16, PyBullet 3.2.7,
NumPy 2.2.6, Stable-Baselines3 2.8.0, and PyTorch 2.15.0.

The gzip JSON traces contain the saved float32 actions, reset seed,
observations, rewards, flags, and scalar info for each rollout. The action
sequences were generated once from NumPy's `default_rng(1200)`. The reset seed
is 21 except for walker v12, which uses 25 to cover all four terrain presets.
The arena uses the shipped 400-step episode limit, so both traces are 820
steps long. The empty-league test supplies a seeded Box for the opponent
because the current arena returns a fresh, unseeded Box on every
`action_space()` call.

The checkpoint files come from two 2,048-step PPO runs on CPU with one env,
seed 11, and observation normalisation. The walker used a decaying learning
rate and keeps its model, normaliser, and run manifest. The manifest omits
the capture host's name and local paths. The pickled learning rate schedules
use a repository-relative source filename. The arena used the
`organisms_fight_arena` environment block and keeps its 1,024-step league
snapshot and matching normaliser. The five checkpoint files total 358,991
bytes.

Keep the fixtures fixed when moving production code. See the
[restructure plan](../../docs/plans/2026-11-12-core-physics-games-restructure.md)
for the compatibility checks they support.
