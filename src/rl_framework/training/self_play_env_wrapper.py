"""Parallel-env wrapper that routes the opponent slot through a frozen policy.

In the default shared-policy arena setup, SuperSuit converts the 2-agent
PettingZoo env into a 2-slot vec env and the *same* live policy drives both
agents. For league self-play we instead want the live (training) policy to
control one agent while a frozen past-self plays the other.

:class:`SelfPlayEnvWrapper` exposes only the live agent to the outside
(SuperSuit/SB3), so SB3 collects and trains on a single agent's transitions.
The opponent's action is computed internally from a frozen policy sampled from
the league at the start of each episode. When the league is empty (early
training) the opponent falls back to random actions.

**Why disk-backed sampling.** SuperSuit's ``concat_vec_envs_v1`` replicates the
env by ``cloudpickle``-cloning it (every instance, even a single one). A live
``SelfPlayCallback`` reference handed to the wrapper would therefore be cloned
into a disconnected copy whose in-memory league never fills — the opponent
would stay random forever. Instead the wrapper reads the league from the
snapshot *directory* that :class:`SelfPlayCallback` writes to; disk is the one
channel that survives both cloudpickle cloning and true multiprocessing.

Observation normalisation: the wrapper sits *inside* training's
``VecNormalize``, so it hands the frozen policy raw egocentric observations.
To keep the opponent on-distribution, each league snapshot is saved with its
obs normaliser (a ``<stem>_vecnorm.pkl`` sidecar written by
``SelfPlayCallback``), and :class:`FrozenPolicy` re-applies it before
predicting. If a snapshot has no sidecar (training ran without normalisation)
the raw observation is used, which is then correct.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from pettingzoo.utils import BaseParallelWrapper
from stable_baselines3 import PPO

# Suffix for the per-snapshot observation-normaliser sidecar written next to
# each league snapshot, so a frozen opponent can normalise its observations the
# same way it did during training (see module docstring's "Known limitation").
VECNORM_SUFFIX = "_vecnorm.pkl"


def load_obs_normalizer(path: str | Path) -> Any | None:
    """Load a saved ``VecNormalize`` from *path* for obs normalisation only.

    Returns the (venv-less) ``VecNormalize`` instance, or ``None`` if the file
    is missing. ``VecNormalize`` drops its ``venv`` on pickling, so the loaded
    object supports ``normalize_obs`` without an attached env.
    """
    path = Path(path)
    if not path.exists():
        return None
    with path.open("rb") as fh:
        return pickle.load(fh)


class FrozenPolicy:
    """A frozen policy paired with the obs normaliser it was trained under.

    Wraps a loaded SB3 model so callers can hand it *raw* observations: if a
    normaliser is present the observation is normalised before prediction,
    matching the distribution the policy saw during training. This closes the
    raw-vs-normalised mismatch that would otherwise make league opponents play
    off-distribution.
    """

    def __init__(self, model: Any, normalizer: Any | None = None) -> None:
        self._model = model
        self._normalizer = normalizer

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        if self._normalizer is not None:
            obs = self._normalizer.normalize_obs(obs)
        return self._model.predict(obs, deterministic=deterministic)


class RandomPolicy:
    """Uniform-random opponent baseline (no model, no normalisation)."""

    def __init__(self, action_space: Any) -> None:
        self._action_space = action_space

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        return self._action_space.sample(), None


def load_frozen_policy(path: str | Path, action_space: Any) -> Any:
    """Load a :class:`FrozenPolicy` from *path*, or a random baseline.

    ``path == "random"`` returns a :class:`RandomPolicy` over *action_space*.
    Otherwise the SB3 model is loaded and paired with an obs normaliser
    discovered next to it: a per-snapshot ``<stem>_vecnorm.pkl`` sidecar (league
    snapshots) if present, then a model-specific ``<stem>_vecnormalize.pkl``
    sidecar, else a sibling ``vecnormalize.pkl`` (legacy final normaliser). If
    neither exists, observations are used raw.
    """
    if str(path) == "random":
        return RandomPolicy(action_space)
    path = Path(path)
    model = PPO.load(str(path))
    sidecar = path.with_name(path.stem + VECNORM_SUFFIX)
    normalizer = load_obs_normalizer(sidecar)
    if normalizer is None:
        normalizer = load_obs_normalizer(
            path.with_name(path.stem + "_vecnormalize.pkl")
        )
    if normalizer is None:
        normalizer = load_obs_normalizer(path.with_name("vecnormalize.pkl"))
    return FrozenPolicy(model, normalizer)


class LeagueSampler:
    """Sample frozen opponent policies from a snapshot directory on disk.

    Mirrors :meth:`SelfPlayCallback.sample_opponent` but reads the league from
    disk rather than an in-memory deque, so it stays correct after the owning
    env is cloudpickle-cloned across vec-env workers. Snapshots are expected to
    be named ``selfplay_<timesteps>.zip`` (as written by ``SelfPlayCallback``).
    """

    def __init__(
        self,
        snapshot_dir: str | Path,
        sampling_mode: str = "uniform",
        recent_bias_alpha: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self._dir = Path(snapshot_dir)
        self._mode = sampling_mode
        self._alpha = recent_bias_alpha
        self._rng = np.random.default_rng(seed)
        self._cache: dict[Path, FrozenPolicy] = {}

    def _league_files(self) -> list[Path]:
        if not self._dir.exists():
            return []
        # Only numeric-suffixed snapshots belong to the league; a stray file
        # like selfplay_best.zip must not crash sampling at episode reset.
        files = [
            p
            for p in self._dir.glob("selfplay_*.zip")
            if p.stem.rsplit("_", 1)[-1].isdigit()
        ]
        return sorted(files, key=lambda p: int(p.stem.rsplit("_", 1)[-1]))

    def sample(self) -> FrozenPolicy | None:
        """Return a frozen opponent from the on-disk league, or ``None`` if empty.

        Each opponent is paired with its per-snapshot obs normaliser sidecar so
        it predicts on the same observation distribution it trained under.
        """
        files = self._league_files()
        if not files:
            return None
        if self._mode == "recent_bias":
            weights = np.arange(1, len(files) + 1, dtype=np.float64) ** self._alpha
            idx = int(self._rng.choice(len(files), p=weights / weights.sum()))
        else:
            idx = int(self._rng.integers(0, len(files)))
        path = files[idx]

        # Drop cached models whose snapshot was pruned from disk.
        live = set(files)
        for cached in [p for p in self._cache if p not in live]:
            self._cache.pop(cached, None)

        if path not in self._cache:
            model = PPO.load(str(path))
            normalizer = load_obs_normalizer(path.with_name(path.stem + VECNORM_SUFFIX))
            self._cache[path] = FrozenPolicy(model, normalizer)
        return self._cache[path]


class SelfPlayEnvWrapper(BaseParallelWrapper):
    """Drive the opponent slot with a frozen policy, exposing only the live agent.

    Parameters
    ----------
    env:
        The underlying 2-agent arena ``ParallelEnv``.
    sampler:
        A :class:`LeagueSampler` that yields frozen opponent policies from disk.
    """

    LIVE_AGENT = "agent_0"
    FROZEN_AGENT = "agent_1"

    def __init__(self, env: Any, sampler: LeagueSampler) -> None:
        super().__init__(env)
        self._sampler = sampler
        self._frozen_policy: PPO | None = None
        # Present a single-agent env to SuperSuit/SB3.
        self.possible_agents = [self.LIVE_AGENT]
        self.agents = [self.LIVE_AGENT]

    # -- spaces: always the live agent's -------------------------------------
    def observation_space(self, agent: str):
        return self.env.observation_space(self.LIVE_AGENT)

    def action_space(self, agent: str):
        return self.env.action_space(self.LIVE_AGENT)

    # -- episode lifecycle ---------------------------------------------------
    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, infos = self.env.reset(seed=seed, options=options)
        # Sample a fresh frozen opponent for this episode (None if league empty).
        self._frozen_policy = self._sampler.sample()
        self.agents = [self.LIVE_AGENT] if self.LIVE_AGENT in self.env.agents else []
        live = self.LIVE_AGENT
        return {live: obs[live]}, {live: infos[live]}

    def step(self, actions: dict[str, np.ndarray]):
        live = self.LIVE_AGENT
        full_actions = dict(actions)
        if self.FROZEN_AGENT in self.env.agents:
            full_actions[self.FROZEN_AGENT] = self._opponent_action()

        obs, rewards, terminations, truncations, infos = self.env.step(full_actions)

        live_term = bool(terminations.get(live, False))
        live_trunc = bool(truncations.get(live, False))
        # The arena marks only the *loser* terminated, but empties its agent
        # list once the episode is over. If the opponent was knocked out, the
        # live agent is the winner — surface a terminal so SB3 sees the boundary.
        if not self.env.agents and not live_term and not live_trunc:
            live_term = True

        self.agents = [live] if self.env.agents else []
        return (
            {live: obs[live]},
            {live: rewards[live]},
            {live: live_term},
            {live: live_trunc},
            {live: infos[live]},
        )

    # -- internals -----------------------------------------------------------
    def _opponent_action(self) -> np.ndarray:
        """Return the frozen opponent's action for the current state."""
        if self._frozen_policy is None:
            # Empty league — random opponent keeps early training moving.
            return self.env.action_space(self.FROZEN_AGENT).sample()
        frozen_obs = self.env.unwrapped.observe(self.FROZEN_AGENT)
        action, _ = self._frozen_policy.predict(
            frozen_obs[np.newaxis], deterministic=True
        )
        return np.asarray(action[0], dtype=np.float32)
