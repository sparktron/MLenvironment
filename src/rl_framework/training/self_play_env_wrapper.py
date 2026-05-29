from __future__ import annotations

from typing import Any

import numpy as np


class SelfPlayEnvWrapper:
    """Routes agent_1's actions through a frozen opponent policy.

    The live training policy controls agent_0. agent_1's observations are
    passed through SelfPlayCallback.sample_opponent() at each step. The
    frozen policy is re-sampled from the league at the start of each episode
    (on reset()). Falls back to random actions when the league is empty.
    """

    FROZEN_AGENT = "agent_1"
    LIVE_AGENT = "agent_0"

    def __init__(self, env: Any, self_play_callback: Any) -> None:
        self.env = env
        self._callback = self_play_callback
        self._frozen_policy: Any = None
        self.possible_agents = env.possible_agents
        self.metadata = env.metadata

    # ------------------------------------------------------------------
    # PettingZoo ParallelEnv API forwarding
    # ------------------------------------------------------------------

    @property
    def agents(self) -> list[str]:
        return self.env.agents

    def observation_space(self, agent: str):
        return self.env.observation_space(agent)

    def action_space(self, agent: str):
        return self.env.action_space(agent)

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._frozen_policy = self._callback.sample_opponent()
        return obs, info

    def step(self, actions: dict[str, np.ndarray]):
        actions = dict(actions)
        if self.FROZEN_AGENT in self.env.agents:
            if self._frozen_policy is not None:
                frozen_obs = self.env._obs(self.FROZEN_AGENT)
                frozen_action, _ = self._frozen_policy.predict(
                    frozen_obs[np.newaxis], deterministic=True
                )
                actions[self.FROZEN_AGENT] = frozen_action[0]
            else:
                actions[self.FROZEN_AGENT] = self.action_space(
                    self.FROZEN_AGENT
                ).sample()
        return self.env.step(actions)

    def render(self):
        return self.env.render()

    def close(self) -> None:
        self.env.close()
