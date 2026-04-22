from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import yaml
from gymnasium.wrappers import RecordVideo
from omegaconf import OmegaConf
from stable_baselines3 import PPO

from rl_framework.envs.registry import make_env
from rl_framework.training.eval_runner import evaluate
from rl_framework.training.multi_seed_runner import run_multi_seed
from rl_framework.training.sb3_runner import train
from rl_framework.training.sweep import run_sweep
from rl_framework.utils.config import load_config, to_container, validate_experiment_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL experiment framework CLI")
    parser.add_argument("command", choices=["train", "eval", "sweep", "multi-seed", "render-replay", "gui"])
    parser.add_argument("--config-name", default="", help="YAML file name without extension (required for all commands except gui)")
    parser.add_argument("--config-dir", default="src/rl_framework/configs/experiments")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--seeds", default="", help="Comma-separated seeds for multi-seed runs (e.g. 0,1,2,3,4)")
    parser.add_argument("--max-workers", type=int, default=None, help="Parallel worker processes for multi-seed runs (default: cpu_count)")
    parser.add_argument("--dry-run", action="store_true", help="Plan runs without executing training (sweep only)")
    parser.add_argument("--resume", default="", help="Path to a saved PPO model (.zip) to resume training from")
    parser.add_argument("--host", default="127.0.0.1", help="GUI server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="GUI server port (default: 5000)")
    return parser.parse_args()


def _save_frames_as_gif(frames: list, out_path: Path, fps: int = 30) -> None:
    """Save a list of HxWx3 uint8 numpy arrays as an animated GIF via PIL."""
    from PIL import Image

    if not frames:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(f) for f in frames]
    duration_ms = max(int(1000 / max(fps, 1)), 20)
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def _render_replay(cfg: dict, model_path: str) -> None:
    from pettingzoo.utils.env import ParallelEnv

    env_cfg = cfg["environment"]
    env_cfg["render_mode"] = "rgb_array"
    env = make_env(env_cfg["type"], env_cfg)
    out_dir = Path(cfg["output"]["base_dir"]) / cfg["experiment_name"] / f"seed_{cfg['seed']}" / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = PPO.load(model_path)

    if isinstance(env, ParallelEnv):
        frames = []
        observations, _ = env.reset(seed=cfg["seed"])
        while env.agents:
            actions = {}
            for agent, obs in observations.items():
                action, _ = model.predict(obs, deterministic=True)
                actions[agent] = action
            observations, _, _, _, _ = env.step(actions)
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        env.close()
        gif_path = out_dir / "replay.gif"
        _save_frames_as_gif(frames, gif_path, fps=env.metadata.get("render_fps", 30))
        print(f"saved_replay={gif_path}  frames={len(frames)}")
        return

    if not isinstance(env, gym.Env):
        raise ValueError(f"Unsupported env type for replay: {type(env).__name__}")
    wrapped = RecordVideo(env, video_folder=str(out_dir), episode_trigger=lambda idx: idx == 0)
    obs, _ = wrapped.reset(seed=cfg["seed"])
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = wrapped.step(action)
        done = bool(term or trunc)
    wrapped.close()


def main() -> None:
    args = _parse_args()

    if args.command == "gui":
        from rl_framework.gui.app import run_gui
        run_gui(host=args.host, port=args.port)
        return

    if not args.config_name:
        raise SystemExit("--config-name is required for the '{}' command".format(args.command))

    cfg = load_config(args.config_name, args.config_dir)
    cfg_dict = to_container(cfg)
    validate_experiment_config(cfg_dict)

    if args.command == "train":
        out = train(cfg_dict, resume_from=args.resume or None)
        print(f"saved_model={out}")
    elif args.command == "eval":
        if not args.model_path:
            raise ValueError("--model-path is required for eval")
        print(yaml.dump(evaluate(cfg_dict, args.model_path), default_flow_style=False))
    elif args.command == "sweep":
        planned = run_sweep(cfg_dict, dry_run=args.dry_run)
        print(f"planned_runs={len(planned)} dry_run={args.dry_run}")
    elif args.command == "multi-seed":
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
        agg = run_multi_seed(cfg_dict, seeds=seeds, max_workers=args.max_workers)
        print(f"mean={agg['mean_return_mean']:.4f}  std={agg['mean_return_std']:.4f}")
    elif args.command == "render-replay":
        if not args.model_path:
            raise ValueError("--model-path is required for render-replay")
        _render_replay(cfg_dict, args.model_path)


if __name__ == "__main__":
    main()
