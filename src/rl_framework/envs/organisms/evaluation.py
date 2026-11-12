"""Arena evaluation hooks."""

import numpy as np


def build_eval_vec_env(env_cfg):
    import supersuit as ss
    from rl_framework.core.registry import make_env
    from rl_framework.envs.organisms.training import _ArenaVecEnvAdapter

    par_env = make_env(env_cfg["type"], env_cfg)
    vec_env = ss.pettingzoo_env_to_vec_env_v1(par_env)
    vec_env = ss.concat_vec_envs_v1(vec_env, 1, num_cpus=1, base_class="stable_baselines3")
    return _ArenaVecEnvAdapter(vec_env)


def evaluate_rollouts(vec_env, model, episodes):
    from rl_framework.training.eval_runner import _was_truncated

    num_agents = vec_env.num_envs
    returns = []
    per_agent_returns = {i: [] for i in range(num_agents)}
    terminated_episodes = 0
    truncated_episodes = 0
    episode_lengths = []
    for _ in range(episodes):
        obs = vec_env.reset()
        done = False
        ep_ret = 0.0
        ep_agent_ret = np.zeros(num_agents, dtype=np.float64)
        ep_len = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = vec_env.step(action)
            reward_arr = np.asarray(reward).flatten()
            ep_ret += float(np.sum(reward_arr))
            ep_agent_ret[: len(reward_arr)] += reward_arr.astype(np.float64)
            ep_len += 1
            done = bool(np.any(dones))
        if _was_truncated(infos):
            truncated_episodes += 1
        else:
            terminated_episodes += 1
        episode_lengths.append(ep_len)
        returns.append(ep_ret)
        for i in range(num_agents):
            per_agent_returns[i].append(float(ep_agent_ret[i]))
    metrics = {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "terminated_rate": float(terminated_episodes / episodes),
        "truncated_rate": float(truncated_episodes / episodes),
        "mean_episode_length": float(np.mean(episode_lengths)),
    }
    for i in range(num_agents):
        metrics[f"agent_{i}_mean_return"] = float(np.mean(per_agent_returns[i]))
        metrics[f"agent_{i}_std_return"] = float(np.std(per_agent_returns[i]))
    return metrics
