from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class ArenaMetricsCallback(BaseCallback):
    """SB3 callback that logs per-rollout win rates and episode counts for arena training."""

    def __init__(self):
        super().__init__()
        self._ep_wins: dict[str, int] = {"agent_0": 0, "agent_1": 0}
        self._ep_count: int = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode_outcome" in info:
                winner = info["episode_outcome"].get("winner")
                if winner and winner in self._ep_wins:
                    self._ep_wins[winner] += 1
                self._ep_count += 1
        return True

    def _on_rollout_end(self) -> None:
        if self._ep_count > 0:
            self.logger.record(
                "arena/agent_0_win_rate", self._ep_wins["agent_0"] / self._ep_count
            )
            self.logger.record(
                "arena/agent_1_win_rate", self._ep_wins["agent_1"] / self._ep_count
            )
            self.logger.record("arena/ep_count", self._ep_count)
            self._ep_wins = {"agent_0": 0, "agent_1": 0}
            self._ep_count = 0
