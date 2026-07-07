from __future__ import annotations

import os
from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class SelfPlayCallback(BaseCallback):
    """SB3 callback that maintains a league of frozen past-self opponents.

    In the default shared-policy multi-agent setup (PettingZoo + SuperSuit),
    both agents use the *same* continuously-updating policy.  This means the
    opponent never stays still long enough for the learner to exploit and
    improve — training often cycles.

    This callback periodically:
    1. Saves a **snapshot** of the current policy to disk.
    2. Adds it to an in-memory *league* of past opponents.
    3. Exposes ``sample_opponent()`` so that an environment wrapper or
       evaluation harness can query a frozen past policy for one agent's
       actions while the other agent continues training with the live model.

    Parameters
    ----------
    snapshot_dir:
        Directory to persist policy snapshots.
    snapshot_freq:
        Save a new snapshot every *snapshot_freq* timesteps.
    max_league_size:
        Maximum number of snapshots kept in the league (oldest are pruned).
    verbose:
        Verbosity (0 = silent, 1 = snapshot messages).

    Usage
    -----
    ::

        cb = SelfPlayCallback(snapshot_dir=paths.checkpoints_dir / "league",
                              snapshot_freq=5000)

        # During rollout, the env wrapper can call:
        opponent = cb.sample_opponent()
        if opponent is not None:
            action, _ = opponent.predict(obs, deterministic=True)

    The callback does **not** modify the training environment itself — it
    provides the building block so that a ``SelfPlayEnvWrapper`` (see below)
    or external orchestration can route one agent's actions through a frozen
    opponent model.
    """

    def __init__(
        self,
        snapshot_dir: str | Path,
        snapshot_freq: int = 5000,
        max_league_size: int = 10,
        sampling_mode: str = "uniform",
        recent_bias_alpha: float = 1.0,
        seed: int | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        if snapshot_freq <= 0:
            raise ValueError(f"snapshot_freq must be > 0, got {snapshot_freq}")
        if max_league_size <= 0:
            raise ValueError(f"max_league_size must be > 0, got {max_league_size}")
        if sampling_mode not in {"uniform", "recent_bias"}:
            raise ValueError(
                f"sampling_mode must be 'uniform' or 'recent_bias', got {sampling_mode}"
            )
        if recent_bias_alpha <= 0:
            raise ValueError(f"recent_bias_alpha must be > 0, got {recent_bias_alpha}")
        self._snapshot_dir = Path(snapshot_dir)
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot_freq = snapshot_freq
        self._max_league_size = max_league_size
        self._sampling_mode = sampling_mode
        self._recent_bias_alpha = recent_bias_alpha
        self._league: deque[Path] = deque()
        self._model_cache: dict[Path, PPO] = {}
        self._rng = np.random.default_rng(seed)
        self._next_snapshot_at = snapshot_freq

    # ------------------------------------------------------------------
    def _on_step(self) -> bool:
        # num_timesteps advances by num_envs per callback call, so an exact
        # modulo test only fires when the stride happens to hit a multiple —
        # an LCM cadence (e.g. every 15k steps for freq=5000 at num_envs=24).
        # Fire on the first call at/after each freq boundary instead, keeping
        # the schedule anchored to multiples of snapshot_freq so per-snapshot
        # overshoot never accumulates.
        if self.num_timesteps >= self._next_snapshot_at:
            self._save_snapshot()
            while self._next_snapshot_at <= self.num_timesteps:
                self._next_snapshot_at += self._snapshot_freq
        return True

    def _save_snapshot(self) -> None:
        tag = f"selfplay_{self.num_timesteps}"
        path = self._snapshot_dir / tag
        # LeagueSampler workers glob selfplay_*.zip as soon as the directory
        # mtime moves, so the snapshot must never be visible half-written:
        # save under a temp name the glob cannot match (SB3 keeps the name
        # as-is because it already has a suffix), put the obs-normaliser
        # sidecar in place, then atomically rename the zip last. A visible
        # snapshot therefore always has a complete archive and sidecar.
        tmp_path = self._snapshot_dir / f".{tag}.tmp"
        self.model.save(str(tmp_path))
        vec_normalize = self.model.get_vec_normalize_env()
        if vec_normalize is not None:
            vec_normalize.save(str(self._snapshot_dir / f"{tag}_vecnorm.pkl"))
        os.replace(tmp_path, path.with_suffix(".zip"))
        self._league.append(path)
        if self.verbose >= 1:
            print(
                f"[SelfPlay] Snapshot saved: {path}  (league size={len(self._league)})"
            )

        # Prune oldest if league exceeds max size.
        while len(self._league) > self._max_league_size:
            old = self._league.popleft()
            old_zip = old.with_suffix(".zip")
            if old_zip.exists():
                old_zip.unlink()
            old_vecnorm = old.with_name(f"{old.name}_vecnorm.pkl")
            if old_vecnorm.exists():
                old_vecnorm.unlink()
            self._model_cache.pop(old, None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_opponent(self) -> PPO | None:
        """Return a randomly sampled frozen opponent from the league, or None."""
        if not self._league:
            return None
        if self._sampling_mode == "recent_bias":
            weights = (
                np.arange(1, len(self._league) + 1, dtype=np.float64)
                ** self._recent_bias_alpha
            )
            probs = weights / weights.sum()
            idx = int(self._rng.choice(len(self._league), p=probs))
            path = self._league[idx]
        else:
            path = self._league[int(self._rng.integers(0, len(self._league)))]

        if path not in self._model_cache:
            self._model_cache[path] = PPO.load(str(path))
        return self._model_cache[path]

    @property
    def league_size(self) -> int:
        return len(self._league)

    @property
    def league_paths(self) -> list[Path]:
        return list(self._league)
