from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from omegaconf import OmegaConf
from stable_baselines3 import PPO

from rl_framework.envs.registry import make_env
from rl_framework.training.eval_runner import evaluate
from rl_framework.training.multi_seed_runner import run_multi_seed
from rl_framework.training.sb3_runner import train
from rl_framework.training.sweep import run_sweep
from rl_framework.utils.config import load_config, to_container


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL experiment framework CLI")
    parser.add_argument("command", choices=["train", "eval", "sweep", "multi-seed", "render-replay"])
    parser.add_argument("--config-name", required=True, help="YAML file name without extension")
    parser.add_argument("--config-dir", default="src/rl_framework/configs/experiments")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--seeds", default="", help="Comma-separated seeds for multi-seed runs (e.g. 0,1,2,3,4)")
    return parser.parse_args()


def _render_replay(cfg: dict, model_path: str) -> None:
    env_cfg = cfg["environment"]
    env_cfg["render_mode"] = "rgb_array"
    env = make_env(env_cfg["type"], env_cfg)
    if not isinstance(env, gym.Env):
        raise ValueError("Replay rendering currently supports Gymnasium envs only")
    out_dir = Path(cfg["output"]["base_dir"]) / cfg["experiment_name"] / f"seed_{cfg['seed']}" / "videos"
    wrapped = RecordVideo(env, video_folder=str(out_dir), episode_trigger=lambda idx: idx == 0)
    model = PPO.load(model_path)

    obs, _ = wrapped.reset(seed=cfg["seed"])
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = wrapped.step(action)
        done = bool(term or trunc)
    wrapped.close()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config_name, args.config_dir)
    cfg_dict = to_container(cfg)

    if args.command == "train":
        out = train(cfg_dict)
        print(f"saved_model={out}")
    elif args.command == "eval":
        if not args.model_path:
            raise ValueError("--model-path is required for eval")
        print(OmegaConf.to_yaml(evaluate(cfg_dict, args.model_path)))
    elif args.command == "sweep":
        run_sweep(cfg_dict)
    elif args.command == "multi-seed":
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
        agg = run_multi_seed(cfg_dict, seeds=seeds)
        print(f"mean={agg['mean_return_mean']:.4f}  std={agg['mean_return_std']:.4f}")
    elif args.command == "render-replay":
        if not args.model_path:
            raise ValueError("--model-path is required for render-replay")
        _render_replay(cfg_dict, args.model_path)


if __name__ == "__main__":
    main()
