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

import gymnasium as gym
import numpy as np
from pettingzoo.utils import BaseParallelWrapper
from stable_baselines3 import PPO

# A prune can race a load more than once in a row only if the league turns
# over completely between attempts; a handful of retries is ample.
_SAMPLE_ATTEMPTS = 5

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
    model = PPO.load(str(path), device="cpu")
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
        # File-list cache, refreshed only when the snapshot dir's mtime changes.
        self._files: list[Path] = []
        self._dir_mtime_ns: int | None = None

    def _league_files(self) -> list[Path]:
        """Return league snapshot paths, re-globbing only on a directory change.

        Adding or pruning a snapshot bumps the directory's mtime, so the sorted
        file list is cached and re-scanned only when that mtime moves. This
        avoids a glob + sort + int-parse on every episode reset, which is hot
        under parallel self-play with short episodes. ``st_mtime_ns`` is used so
        rapid successive changes are still detected.
        """
        try:
            mtime_ns = self._dir.stat().st_mtime_ns
        except FileNotFoundError:
            self._files = []
            self._dir_mtime_ns = None
            return self._files
        if mtime_ns == self._dir_mtime_ns:
            return self._files

        self._dir_mtime_ns = mtime_ns
        # Only numeric-suffixed snapshots belong to the league; a stray file
        # like selfplay_best.zip must not crash sampling at episode reset.
        files = [
            p
            for p in self._dir.glob("selfplay_*.zip")
            if p.stem.rsplit("_", 1)[-1].isdigit()
        ]
        self._files = sorted(files, key=lambda p: int(p.stem.rsplit("_", 1)[-1]))

        # Drop cached models whose snapshot was pruned from disk.
        live = set(self._files)
        for cached in [p for p in self._cache if p not in live]:
            self._cache.pop(cached, None)

        # Pre-warm the newest snapshot off the hot sampling path — it is the
        # most likely opponent under recent_bias and stays in the league until
        # it ages out, so the load is inevitable. SelfPlayCallback writes the
        # sidecar first and renames the zip into place last, so a visible zip
        # is complete. The sidecar gate remains for leagues written before
        # snapshots became atomic, where the zip could appear first. When no
        # normaliser is ever saved the sidecar is absent and the pre-warm is
        # skipped; those raw-obs loads are correct and cheap to do lazily.
        if self._files:
            newest = self._files[-1]
            if newest.with_name(newest.stem + VECNORM_SUFFIX).exists():
                try:
                    self._ensure_loaded(newest)
                except (FileNotFoundError, OSError, EOFError):
                    pass  # pruned between glob and load; sample() handles it
        return self._files

    def _ensure_loaded(self, path: Path) -> FrozenPolicy:
        """Return the cached :class:`FrozenPolicy` for *path*, loading on miss.

        Each opponent is paired with its per-snapshot obs normaliser sidecar so
        it predicts on the same observation distribution it trained under.
        Snapshots written by the current SelfPlayCallback are only visible once
        their sidecar is complete, but a league written before snapshots became
        atomic can expose a zip before its sidecar — so a cache entry that has
        no normaliser re-probes the sidecar and attaches it when it appears,
        instead of playing unnormalised for the rest of training.
        """
        sidecar = path.with_name(path.stem + VECNORM_SUFFIX)
        cached = self._cache.get(path)
        if cached is None:
            cached = FrozenPolicy(
                PPO.load(str(path), device="cpu"), load_obs_normalizer(sidecar)
            )
            self._cache[path] = cached
        elif cached._normalizer is None:
            normalizer = load_obs_normalizer(sidecar)
            if normalizer is not None:
                cached._normalizer = normalizer
        return cached

    def sample(self) -> FrozenPolicy | None:
        """Return a frozen opponent from the on-disk league, or ``None`` if empty.

        The main process prunes old snapshots while workers sample. A worker
        can list a snapshot, lose it to a prune, then fail to load it; that
        used to raise at env reset and kill the worker. The vanished file is
        dropped and another snapshot drawn instead.
        """
        for _ in range(_SAMPLE_ATTEMPTS):
            files = self._league_files()
            if not files:
                return None
            if self._mode == "recent_bias":
                weights = np.arange(1, len(files) + 1, dtype=np.float64) ** self._alpha
                idx = int(self._rng.choice(len(files), p=weights / weights.sum()))
            else:
                idx = int(self._rng.integers(0, len(files)))
            try:
                return self._ensure_loaded(files[idx])
            except (FileNotFoundError, OSError, EOFError):
                if files[idx].exists():
                    raise
                self._dir_mtime_ns = None  # force a re-glob
        return None

    def draw(self, probability: float) -> bool:
        """Bernoulli draw from this sampler's seeded stream."""
        return bool(probability > 0.0 and self._rng.random() < probability)


class SelfPlayEnvWrapper(BaseParallelWrapper):
    """Drive every opponent slot with a frozen policy, exposing only the live agent.

    For the 2-agent arena this is the single ``agent_1`` slot; for N-agent
    free-for-alls one sampled frozen past-self fills all of ``agent_1`` …
    ``agent_{N-1}``.

    Parameters
    ----------
    env:
        The underlying N-agent arena ``ParallelEnv`` (N≥2).
    sampler:
        A :class:`LeagueSampler` that yields frozen opponent policies from disk.
    """

    LIVE_AGENT = "agent_0"
    # Retained for back-compat; the wrapper now drives *every* non-live slot.
    FROZEN_AGENT = "agent_1"

    def __init__(
        self,
        env: Any,
        sampler: LeagueSampler,
        opponent_stochastic_prob: float = 0.0,
        randomize_live_slot: bool = False,
    ) -> None:
        super().__init__(env)
        if not 0.0 <= opponent_stochastic_prob <= 1.0:
            raise ValueError("opponent_stochastic_prob must be in [0, 1]")
        self._sampler = sampler
        self._frozen_policy: FrozenPolicy | None = None
        # Frozen opponents always acting deterministically let the learner
        # overfit to one fixed reply per state. A fraction of episodes sample
        # the opponent's actions from its policy distribution instead.
        self._opponent_stochastic_prob = float(opponent_stochastic_prob)
        self._opponent_deterministic = True
        # The learner always held agent_0, so any asymmetry in spawn order or
        # tie-breaking was baked into the policy. With this on, the learner
        # takes a random slot each episode; the outside world still sees it
        # under the LIVE_AGENT key.
        self._randomize_live_slot = bool(randomize_live_slot)
        self._live_slot = self.LIVE_AGENT
        # Every non-live slot is driven by the frozen opponent (one sampled
        # past-self fills all of them). For the 2-agent arena this is just
        # agent_1; for N-agent free-for-alls it is agents 1..N-1.
        self._opponent_agents = [a for a in env.possible_agents if a != self._live_slot]
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
        if self._randomize_live_slot:
            slots = list(self.env.possible_agents)
            self._live_slot = slots[int(self._sampler._rng.integers(0, len(slots)))]
            self._opponent_agents = [a for a in slots if a != self._live_slot]
        # Sample a fresh frozen opponent for this episode (None if league empty).
        self._frozen_policy = self._sampler.sample()
        self._opponent_deterministic = not (
            self._frozen_policy is not None
            and self._opponent_stochastic_prob > 0.0
            and self._sampler.draw(self._opponent_stochastic_prob)
        )
        slot = self._live_slot
        self.agents = [self.LIVE_AGENT] if slot in self.env.agents else []
        return {self.LIVE_AGENT: obs[slot]}, {self.LIVE_AGENT: infos[slot]}

    def _relabel_outcome(self, info: dict[str, Any]) -> dict[str, Any]:
        """Report the learner as LIVE_AGENT in episode outcomes.

        Win-rate gates compare ``winner`` with ``agent_0``; when the learner
        holds another slot the two names are swapped so the gate stays right.
        """
        outcome = info.get("episode_outcome")
        if self._live_slot == self.LIVE_AGENT or not isinstance(outcome, dict):
            return info
        swap = {self._live_slot: self.LIVE_AGENT, self.LIVE_AGENT: self._live_slot}
        winner = outcome.get("winner")
        info = dict(info)
        info["episode_outcome"] = {
            **outcome,
            "winner": swap.get(winner, winner) if isinstance(winner, str) else winner,
        }
        return info

    def step(self, actions: dict[str, np.ndarray]):
        slot = self._live_slot
        full_actions = {slot: actions[self.LIVE_AGENT]}
        for opp in self._opponent_agents:
            if opp in self.env.agents:
                full_actions[opp] = self._opponent_action(opp)

        obs, rewards, terminations, truncations, infos = self.env.step(full_actions)
        live = slot

        live_term = bool(terminations.get(live, False))
        live_trunc = bool(truncations.get(live, False))
        # The arena retains knocked-out agents as inert spectators until the
        # whole free-for-all is over. That keeps its PettingZoo population
        # constant, but SB3 must end the learner's transition immediately when
        # the live slot is eliminated rather than collect spectator steps.
        eliminated = not self.env.unwrapped.is_alive(live)
        if eliminated:
            live_term = True
            if self.env.agents:
                # The underlying free-for-all continues for the frozen
                # opponents, but from the learner's perspective this is a
                # completed loss. Preserve that boundary for metrics and the
                # curriculum win-rate gate.
                infos[live] = dict(infos[live])
                infos[live]["episode_outcome"] = {
                    "winner": None,
                    "outcome": "eliminated",
                    "step": self.env.unwrapped.step_count,
                }
        # If an opponent was knocked out and the arena ended, the live agent is
        # the winner — surface a terminal so SB3 sees the boundary.
        if not self.env.agents and not live_term and not live_trunc:
            live_term = True

        key = self.LIVE_AGENT
        self.agents = [key] if self.env.agents and not live_term else []
        return (
            {key: obs[live]},
            {key: rewards[live]},
            {key: live_term},
            {key: live_trunc},
            {key: self._relabel_outcome(infos[live])},
        )

    # -- internals -----------------------------------------------------------
    def _opponent_action(self, agent: str) -> np.ndarray:
        """Return the frozen opponent's action for *agent* at the current state."""
        if self._frozen_policy is None:
            # Empty league — random opponent keeps early training moving.
            return self.env.action_space(agent).sample()
        frozen_obs = self.env.unwrapped.observe(agent)
        action, _ = self._frozen_policy.predict(
            frozen_obs[np.newaxis], deterministic=self._opponent_deterministic
        )
        return np.asarray(action[0], dtype=np.float32)


class SingleAgentArenaEnv(gym.Env):
    """Expose a single-agent :class:`SelfPlayEnvWrapper` as a Gymnasium env.

    In league self-play the live policy controls one slot while a frozen
    past-self drives the other, so the wrapped env presents exactly one agent.
    That makes SuperSuit's multi-agent vec conversion unnecessary: this adapter
    lets the arena ride SB3's native ``DummyVecEnv``/``SubprocVecEnv`` path
    instead. The win over SuperSuit is twofold:

    * **Parallel workers.** SB3's ``SubprocVecEnv.env_method`` reaches each
      worker's env through pipes, so curriculum / reward-annealing updates fire
      correctly across processes — unlike SuperSuit's ``ConcatVecEnv``, whose
      in-process chain is empty once envs are forked (the reason arena training
      was previously capped at ``num_envs == 1``).
    * **Monitor metrics.** Wrapping in SB3's ``Monitor`` restores
      ``rollout/ep_rew_mean`` and ``ep_len_mean``.

    The disk-backed :class:`LeagueSampler` already survives cloudpickle cloning
    into subprocess workers, so no opponent state needs to cross the process
    boundary.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, env: SelfPlayEnvWrapper) -> None:
        self._env = env
        self._agent = env.LIVE_AGENT
        self.observation_space = env.observation_space(self._agent)
        self.action_space = env.action_space(self._agent)
        self.render_mode = getattr(env.unwrapped, "render_mode", None)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, infos = self._env.reset(seed=seed, options=options)
        a = self._agent
        return obs[a], infos.get(a, {})

    def step(self, action: np.ndarray):
        a = self._agent
        obs, rewards, terms, truncs, infos = self._env.step({a: action})
        return (
            obs[a],
            float(rewards[a]),
            bool(terms[a]),
            bool(truncs[a]),
            infos.get(a, {}),
        )

    def render(self):
        return self._env.unwrapped.render()

    def close(self):
        self._env.close()

    def update_live_params(self, params: dict[str, Any]) -> None:
        """Forward live curriculum / annealing overrides to the arena env.

        SB3's ``env_method('update_live_params', ...)`` resolves to this method
        on the (Monitor-wrapped) adapter; we delegate to the underlying arena
        env so the update reaches the object that actually steps — including in
        subprocess workers.
        """
        self._env.unwrapped.update_live_params(params)
