from __future__ import annotations

import argparse
import json as _json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL experiment framework CLI")
    parser.add_argument(
        "command",
        choices=[
            "train",
            "eval",
            "arena-eval",
            "arena-tournament",
            "sweep",
            "multi-seed",
            "render-replay",
            "gui",
            "morph-search",
        ],
    )
    parser.add_argument(
        "--config-name",
        default="",
        help="YAML file name without extension (required for all commands except gui)",
    )
    parser.add_argument("--config-dir", default="src/rl_framework/configs/experiments")
    parser.add_argument("--model-path", default="")
    parser.add_argument(
        "--policy",
        default="",
        help="Path to the policy checkpoint under test (arena-eval).",
    )
    parser.add_argument(
        "--opponent",
        default="random",
        help="Opponent checkpoint path, or 'random' for a random baseline (arena-eval).",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=100,
        help="Episodes per spawn orientation for arena-eval (default: 100).",
    )
    parser.add_argument(
        "--no-swap-roles",
        action="store_true",
        help="Disable spawn-slot swapping in arena-eval (keeps positional bias).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to write the arena-eval result as JSON.",
    )
    parser.add_argument(
        "--checkpoints",
        default="",
        help="Comma-separated checkpoint paths and/or directories for "
        "arena-tournament (each directory contributes its *.zip files).",
    )
    parser.add_argument(
        "--include-random",
        action="store_true",
        help="Add a random-action baseline competitor to the tournament.",
    )
    parser.add_argument(
        "--markdown-out",
        default="",
        help="Optional path to write the arena-tournament report as markdown.",
    )
    parser.add_argument(
        "--replay-opponent",
        default="",
        help="Arena render-replay only: checkpoint path (or 'random') to drive "
        "agent_1. Default mirrors the main policy (shared-policy replay).",
    )
    parser.add_argument(
        "--seeds",
        default="",
        help="Comma-separated seeds for multi-seed runs (e.g. 0,1,2,3,4)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Parallel worker processes for multi-seed runs (default: cpu_count)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan runs without executing training (sweep only)",
    )
    parser.add_argument(
        "--resume",
        default="",
        help="Path to a saved PPO model (.zip) to resume training from",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override training.total_timesteps for this run.",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Override training device for this run: auto | cpu | cuda | cuda:<N> (e.g. cuda:0)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of trials for morph-search (default: 5)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="GUI server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=5001, help="GUI server port (default: 5001)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write the result as a single JSON line to stdout; suppresses human-readable output. "
        "Useful for scripting and CI pipelines.",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to write JSON result payload. Useful for robust automation without parsing stdout.",
    )
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


def _render_replay(
    cfg: dict, model_path: str, opponent_path: str | None = None
) -> dict:
    """Render a replay and return a result dict with saved path and frame count.

    For the 2-agent arena, *model_path* drives ``agent_0``. By default the same
    policy mirrors into ``agent_1`` (a shared-policy replay); pass
    *opponent_path* (a checkpoint or ``"random"``) to watch a real matchup. Both
    slots are loaded via :func:`load_frozen_policy`, so a saved obs-normaliser
    sidecar is applied and each policy sees the distribution it trained under.
    """
    import gymnasium as gym
    import numpy as np
    from gymnasium.wrappers import RecordVideo
    from pettingzoo.utils.env import ParallelEnv
    from stable_baselines3 import PPO

    from rl_framework.envs.registry import make_env

    env_cfg = cfg["environment"]
    env_cfg["render_mode"] = "rgb_array"
    env = make_env(env_cfg["type"], env_cfg)
    from rl_framework.utils.logging_utils import create_experiment_paths

    paths = create_experiment_paths(
        cfg["output"]["base_dir"],
        cfg["experiment_name"],
        cfg["seed"],
        run_id=cfg["output"].get("run_id"),
    )
    out_dir = paths.videos_dir

    if isinstance(env, ParallelEnv):
        from rl_framework.training.self_play_env_wrapper import load_frozen_policy

        agents = list(env.possible_agents)
        policy_slot = agents[0]
        action_space = env.action_space(policy_slot)
        policy = load_frozen_policy(model_path, action_space)
        # No explicit opponent -> mirror the main policy (shared-policy replay).
        opponent = (
            load_frozen_policy(opponent_path, action_space) if opponent_path else policy
        )
        try:
            frames = []
            observations, _ = env.reset(seed=cfg["seed"])
            while env.agents:
                actions = {}
                for agent, obs in observations.items():
                    actor = policy if agent == policy_slot else opponent
                    action, _ = actor.predict(obs, deterministic=True)
                    actions[agent] = np.asarray(action, dtype=np.float32)
                observations, _, _, _, _ = env.step(actions)
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            gif_path = out_dir / "replay.gif"
            _save_frames_as_gif(
                frames, gif_path, fps=env.metadata.get("render_fps", 30)
            )
            return {
                "saved_replay": str(gif_path),
                "frames": len(frames),
                "opponent": opponent_path or "self",
            }
        finally:
            env.close()

    model = PPO.load(model_path)

    if not isinstance(env, gym.Env):
        raise ValueError(f"Unsupported env type for replay: {type(env).__name__}")
    wrapped = RecordVideo(
        env, video_folder=str(out_dir), episode_trigger=lambda idx: idx == 0
    )
    try:
        obs, _ = wrapped.reset(seed=cfg["seed"])
        done = False
        frame_count = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = wrapped.step(action)
            done = bool(term or trunc)
            frame_count += 1
        # RecordVideo writes an mp4; find it for the result dict.
        mp4_files = list(out_dir.glob("*.mp4"))
        saved = str(mp4_files[0]) if mp4_files else str(out_dir)
        return {"saved_replay": saved, "frames": frame_count}
    finally:
        wrapped.close()


def main() -> None:
    args = _parse_args()

    if args.command == "gui":
        from rl_framework.gui.app import run_gui

        run_gui(host=args.host, port=args.port)
        return

    from rl_framework.utils.config import (
        load_config,
        to_container,
        validate_experiment_config,
    )

    if not args.config_name:
        raise SystemExit(
            "--config-name is required for the '{}' command".format(args.command)
        )

    cfg = load_config(args.config_name, args.config_dir)
    cfg_dict = to_container(cfg)
    if args.device:
        cfg_dict.setdefault("training", {})["device"] = args.device
    if args.total_timesteps is not None:
        cfg_dict.setdefault("training", {})["total_timesteps"] = args.total_timesteps
    validate_experiment_config(cfg_dict)

    result: dict = {}

    if args.command == "train":
        from rl_framework.training.sb3_runner import train

        out = train(cfg_dict, resume_from=args.resume or None)
        result = {"saved_model": str(out)}
        if not args.json:
            print(f"saved_model={out}")

    elif args.command == "eval":
        import yaml
        from rl_framework.training.eval_runner import evaluate

        if not args.model_path:
            raise ValueError("--model-path is required for eval")
        metrics = evaluate(cfg_dict, args.model_path)
        result = metrics
        if not args.json:
            print(yaml.dump(metrics, default_flow_style=False))

    elif args.command == "arena-eval":
        from rl_framework.training.arena_eval import run_arena_eval

        if not args.policy:
            raise ValueError("--policy is required for arena-eval")
        result = run_arena_eval(
            args.policy,
            args.opponent,
            cfg_dict,
            n_episodes=args.n_episodes,
            swap_roles=not args.no_swap_roles,
            output_path=args.output or None,
        )
        if not args.json:
            print(
                f"policy_win_rate={result['policy_win_rate']:.3f}  "
                f"opponent_win_rate={result['opponent_win_rate']:.3f}  "
                f"draw_rate={result['draw_rate']:.3f}  "
                f"timeout_rate={result['timeout_rate']:.3f}  "
                f"n_episodes={result['n_episodes']}"
            )

    elif args.command == "arena-tournament":
        from rl_framework.training.arena_tournament import run_tournament

        checkpoints = [c for c in args.checkpoints.split(",") if c.strip()]
        if not checkpoints and not args.include_random:
            raise ValueError(
                "arena-tournament needs --checkpoints (and/or --include-random)"
            )
        result = run_tournament(
            checkpoints,
            cfg_dict,
            n_episodes=args.n_episodes,
            swap_roles=not args.no_swap_roles,
            include_random=args.include_random,
            output_path=args.output or None,
            markdown_path=args.markdown_out or None,
        )
        if not args.json:
            print(f"Tournament: {len(result['competitors'])} competitors")
            print(f"{'Rank':>4}  {'Elo':>6}  {'W-L-D':>10}  Competitor")
            for s in result["standings"]:
                wld = f"{s['wins']}-{s['losses']}-{s['draws']}"
                print(f"{s['rank']:>4}  {s['elo']:>6.0f}  {wld:>10}  {s['competitor']}")

    elif args.command == "sweep":
        from rl_framework.training.sweep import run_sweep

        planned = run_sweep(cfg_dict, dry_run=args.dry_run)
        result = {"planned_runs": len(planned), "dry_run": args.dry_run}
        if not args.json:
            print(f"planned_runs={len(planned)} dry_run={args.dry_run}")

    elif args.command == "multi-seed":
        from rl_framework.training.multi_seed_runner import run_multi_seed

        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
        agg = run_multi_seed(cfg_dict, seeds=seeds, max_workers=args.max_workers)
        result = agg
        if not args.json:
            print(
                f"mean={agg['mean_return_mean']:.4f}  std={agg['mean_return_std']:.4f}"
            )

    elif args.command == "render-replay":
        if not args.model_path:
            raise ValueError("--model-path is required for render-replay")
        result = _render_replay(
            cfg_dict, args.model_path, opponent_path=args.replay_opponent or None
        )
        if not args.json:
            print(f"saved_replay={result['saved_replay']}  frames={result['frames']}")

    elif args.command == "morph-search":
        from rl_framework.training.morphology_search import run_morphology_search

        result = run_morphology_search(
            cfg_dict, trials=args.trials, seed=cfg_dict["seed"]
        )
        if not args.json:
            print(
                f"best_trial={result['best_trial']}  "
                f"best_score={result['best_score']:.4f}  "
                f"best_params={result['best_params']}"
            )

    if args.json:
        print(_json.dumps(result))
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(_json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
