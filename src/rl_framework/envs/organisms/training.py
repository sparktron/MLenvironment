"""Arena vector environments and training callbacks."""
from pathlib import Path
from typing import Any
import numpy as np
import supersuit as ss
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper
from rl_framework.core.registry import make_env
from rl_framework.training.reward_annealing_callback import RewardAnnealingCallback
from rl_framework.training.self_play_callback import SelfPlayCallback
from rl_framework.training.self_play_env_wrapper import LeagueSampler, SelfPlayEnvWrapper, SingleAgentArenaEnv
from rl_framework.training.sb3_runner import _worker_env_config, _advertise_selected_render_mode, _make_subproc_vec_env

class _ArenaVecEnvAdapter(VecEnvWrapper):
    """Adapt SuperSuit's SB3 arena vec env to SB3's expectations.

    Two SuperSuit 3.10 ↔ Stable-Baselines3 incompatibilities are patched here:

    1. **Missing ``seed()``.** SuperSuit's ``ConcatVecEnv`` only supports seeding
       through ``reset(seed=)`` and exposes no ``seed()`` method, yet SB3's
       ``set_random_seed`` calls ``env.seed(seed)`` during ``PPO`` construction —
       otherwise raising ``AttributeError`` before training can start. Routing
       through SuperSuit's ``reset(seed=)`` is not viable: its
       ``SB3VecEnvWrapper.reset`` re-enters the same broken ``seed()`` and
       recurses. The arena env's RNG is already seeded from ``cfg["seed"]`` at
       construction (and preserved across ``reset(seed=None)``), so ``seed()``
       only needs to seed the action/observation spaces for reproducible
       sampling.

    2. **Missing ``render_mode`` forwarding.** SuperSuit's SB3 wrapper tries
       to ask its Gymnasium ``ConcatVecEnv`` for ``render_mode`` via a missing
       ``get_attr`` method. SB3 emits a warning and defaults to ``None``. We
       provide that metadata directly while preserving normal delegation for
       every other attribute.

    3. **``uint8`` dones.** SuperSuit returns the ``dones`` array as ``uint8``.
       ``VecNormalize.step_wait`` does ``self.returns[dones] = 0``, which numpy
       interprets as *integer fancy-indexing* rather than a boolean mask — at
       ``num_envs == 1`` (single-agent self-play) ``returns[[1]]`` is out of
       bounds and crashes; at higher counts it silently resets the wrong slots.
       Casting ``dones`` to ``bool`` restores mask semantics.
    """

    def reset(self):  # type: ignore[override]
        return self.venv.reset()

    def get_attr(self, attr_name, indices=None):
        """Supply render metadata without crossing SuperSuit's broken boundary."""
        if attr_name == "render_mode":
            return [getattr(self.venv, "render_mode", None)] * self.num_envs
        return self.venv.get_attr(attr_name, indices)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return obs, rewards, np.asarray(dones, dtype=bool), infos

    def seed(self, seed: int | None = None):
        if seed is not None:
            self.action_space.seed(seed)
            self.observation_space.seed(seed)
        return [seed] * self.num_envs

    def _arena_envs(self) -> list:
        """Reach the live arena ``ParallelEnv`` instances inside the chain.

        SuperSuit's ``ConcatVecEnv`` is a gymnasium vector env and does not
        implement SB3's ``env_method``, so curriculum / reward-annealing updates
        cannot propagate natively. We walk the structure
        (``SB3VecEnvWrapper.venv`` → ``ConcatVecEnv.vec_envs`` →
        ``MarkovVectorEnv.par_env`` → ``.unwrapped``) to the underlying arena
        envs, unwrapping any SelfPlayEnvWrapper.
        """
        envs: list = []
        concat = getattr(self.venv, "venv", None)  # SB3VecEnvWrapper -> ConcatVecEnv
        for markov in getattr(concat, "vec_envs", []):
            par_env = getattr(markov, "par_env", None)
            if par_env is not None:
                envs.append(par_env.unwrapped)
        return envs

    def env_method(self, method_name, *args, indices=None, **kwargs):
        """Call a method on each underlying arena env (SB3 ``env_method`` shim)."""
        return [
            getattr(env, method_name)(*args, **kwargs) for env in self._arena_envs()
        ]


class ArenaMetricsCallback(BaseCallback):
    """Log per-episode arena outcomes (win rates, timeouts) to TensorBoard.

    The arena env is wrapped through SuperSuit rather than SB3's ``Monitor``,
    so ``rollout/ep_rew_mean`` is unavailable. This callback instead reads the
    ``episode_outcome`` annotation that the arena attaches to terminal/truncated
    steps (or that ``SelfPlayEnvWrapper`` attaches when a live N-agent learner is
    eliminated) and records aggregate win rates at the end of each rollout.

    Note: SuperSuit's vec-env conversion surfaces one ``info`` per agent slot,
    so a finished episode contributes its outcome once per *active* agent
    (2 for a duel, N for an N-agent free-for-all — spectators stay active
    until the episode ends). Win rates are computed as a fraction of *outcome
    observations*, which keeps the ratios correct regardless of N since both
    the numerator and denominator scale by the same active-agent count.

    Win totals are keyed by whatever agent names are actually observed as
    winners, so this scales to N-agent free-for-alls (``environment.
    num_agents`` up to 8 in the GUI schema) rather than only the 2-agent
    default; ``agent_0``/``agent_1`` are seeded up front so their rate is
    always logged (as 0.0) even in rollouts where they didn't win, matching
    prior behavior for the common duel case.
    """

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self._wins: dict[str, int] = {"agent_0": 0, "agent_1": 0}
        self._timeouts = 0
        self._draws = 0
        self._eliminations = 0
        self._outcomes = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            outcome = info.get("episode_outcome")
            if not outcome:
                continue
            self._outcomes += 1
            kind = outcome.get("outcome")
            if kind == "timeout":
                self._timeouts += 1
            elif kind == "draw":
                self._draws += 1
            elif kind == "eliminated":
                self._eliminations += 1
            else:
                winner = outcome.get("winner")
                if winner is not None:
                    self._wins[winner] = self._wins.get(winner, 0) + 1
        return True

    def _on_rollout_end(self) -> None:
        if self._outcomes == 0:
            return
        total = self._outcomes
        for agent, count in self._wins.items():
            self.logger.record(f"arena/{agent}_win_rate", count / total)
        self.logger.record("arena/timeout_rate", self._timeouts / total)
        self.logger.record("arena/draw_rate", self._draws / total)
        self.logger.record("arena/elimination_rate", self._eliminations / total)
        self.logger.record("arena/episode_outcomes", total)
        # Keep any agent keys discovered this run (so their metric keeps
        # being logged, at 0.0, in rollouts where they don't win).
        self._wins = dict.fromkeys(self._wins, 0)
        self._timeouts = 0
        self._draws = 0
        self._eliminations = 0
        self._outcomes = 0


def _build_arena_selfplay_env(
    env_cfg: dict[str, Any],
    self_play_cfg: dict[str, Any],
    league_dir: Path,
    base_seed: int,
    monitor_dir: Path | None = None,
    rank: int = 0,
    render_env_index: int | None = None,
):
    """Factory for one self-play arena env on SB3's native vec-env path.

    The arena ``ParallelEnv`` is wrapped in a :class:`SelfPlayEnvWrapper`
    (frozen opponent in the second slot) and then a :class:`SingleAgentArenaEnv`
    Gymnasium adapter, so it behaves like any single-agent env under
    ``DummyVecEnv``/``SubprocVecEnv`` — no SuperSuit, working ``env_method``,
    and a ``Monitor`` for ``rollout/ep_rew_mean``.

    Each rank is seeded independently (env construction + league sampler) so
    parallel workers explore distinct spawns and sample distinct opponent
    sequences rather than running identical episodes.
    """

    def _make():
        rank_cfg = _worker_env_config(env_cfg, rank, render_env_index)
        # Spread per-worker seeds so spawn jitter and the RNG differ by rank.
        rank_cfg["seed"] = base_seed + 1000 * rank
        par_env = make_env(rank_cfg["type"], rank_cfg)
        sampler = LeagueSampler(
            league_dir,
            sampling_mode=str(self_play_cfg.get("sampling_mode", "uniform")),
            recent_bias_alpha=float(self_play_cfg.get("recent_bias_alpha", 1.0)),
            seed=base_seed + 1000 * rank,
        )
        env = SingleAgentArenaEnv(SelfPlayEnvWrapper(par_env, sampler))
        env = _advertise_selected_render_mode(env, env_cfg, rank, render_env_index)
        filename = (
            str(monitor_dir / f"monitor_env{rank}.csv")
            if monitor_dir is not None
            else None
        )
        # info_keywords persists the per-episode arena outcome into the Monitor
        # CSV; the live agent's info carries it on terminal/timeout steps.
        return Monitor(env, filename=filename, info_keywords=("episode_outcome",))

    return _make


def build_train_vec_env(cfg, paths, render_env_index):
    env_cfg = cfg["environment"]
    train_cfg = cfg["training"]
    num_envs = int(train_cfg.get("num_envs", 1))
    self_play_cfg = cfg.get("self_play", {})
    enabled = bool(self_play_cfg.get("enabled", False))
    league_dir = paths.checkpoints_dir / "league"
    if num_envs > 1 and not enabled:
        raise ValueError(
            "Shared-policy organism_arena_parallel training (self_play.enabled: "
            f"false) requires training.num_envs == 1 (got {num_envs}). It runs "
            "single-process via SuperSuit; num_envs > 1 forks into subprocesses, "
            "silently disabling live env_method updates (reward annealing, "
            "curriculum) and is unstable in SuperSuit 3.10. Either set "
            "training.num_envs: 1, or enable self_play to use the parallel path."
        )
    if enabled:
        env_fns = [_build_arena_selfplay_env(env_cfg, self_play_cfg, league_dir,
            base_seed=int(cfg["seed"]), monitor_dir=paths.logs_dir, rank=i,
            render_env_index=render_env_index) for i in range(max(num_envs, 1))]
        return _make_subproc_vec_env(env_fns, train_cfg) if num_envs > 1 else DummyVecEnv(env_fns)
    arena_cfg = _worker_env_config(env_cfg, 0, render_env_index)
    par_env = make_env(arena_cfg["type"], arena_cfg)
    vec_env = ss.pettingzoo_env_to_vec_env_v1(par_env)
    vec_env = ss.concat_vec_envs_v1(vec_env, 1, num_cpus=1, base_class="stable_baselines3")
    return _ArenaVecEnvAdapter(vec_env)


def train_callbacks(cfg, paths, slot):
    if slot == "before_curriculum":
        callbacks = [ArenaMetricsCallback()]
        anneal_cfg = cfg.get("reward_annealing", {})
        if anneal_cfg.get("enabled", False):
            callbacks.append(RewardAnnealingCallback(
                anneal_steps=int(anneal_cfg.get("anneal_steps", 500_000)),
                min_eliminations=int(anneal_cfg.get("min_eliminations", 0)), verbose=1))
        return callbacks
    if slot != "after_curriculum":
        return []
    self_play_cfg = cfg.get("self_play", {})
    if not self_play_cfg.get("enabled", False):
        return []
    return [SelfPlayCallback(snapshot_dir=paths.checkpoints_dir / "league",
        snapshot_freq=int(self_play_cfg.get("snapshot_freq", 5000)),
        max_league_size=int(self_play_cfg.get("max_league_size", 10)),
        sampling_mode=str(self_play_cfg.get("sampling_mode", "uniform")),
        recent_bias_alpha=float(self_play_cfg.get("recent_bias_alpha", 1.0)),
        seed=cfg["seed"], verbose=1)]
