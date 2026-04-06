from __future__ import annotations

from pathlib import Path
from typing import Any

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
        verbose: int = 0,
    ):
        super().__init__(verbose)
        if snapshot_freq <= 0:
            raise ValueError(f"snapshot_freq must be > 0, got {snapshot_freq}")
        if max_league_size <= 0:
            raise ValueError(f"max_league_size must be > 0, got {max_league_size}")
        self._snapshot_dir = Path(snapshot_dir)
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot_freq = snapshot_freq
        self._max_league_size = max_league_size
        self._league: list[Path] = []
        self._rng = np.random.default_rng(42)

    # ------------------------------------------------------------------
    def _on_step(self) -> bool:
        if self.num_timesteps % self._snapshot_freq == 0 and self.num_timesteps > 0:
            self._save_snapshot()
        return True

    def _save_snapshot(self) -> None:
        tag = f"selfplay_{self.num_timesteps}"
        path = self._snapshot_dir / tag
        self.model.save(str(path))
        self._league.append(path)
        if self.verbose >= 1:
            print(f"[SelfPlay] Snapshot saved: {path}  (league size={len(self._league)})")

        # Prune oldest if league exceeds max size.
        while len(self._league) > self._max_league_size:
            old = self._league.pop(0)
            old_zip = old.with_suffix(".zip")
            if old_zip.exists():
                old_zip.unlink()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_opponent(self) -> PPO | None:
        """Return a randomly sampled frozen opponent from the league, or None."""
        if not self._league:
            return None
        path = self._rng.choice(self._league)
        return PPO.load(str(path))

    @property
    def league_size(self) -> int:
        return len(self._league)

    @property
    def league_paths(self) -> list[Path]:
        return list(self._league)
