"""Quick eval to count lines cleared."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import glob
import os

import gymnasium as gym
from stable_baselines3 import DQN
import tetris_gymnasium.envs  # noqa
from tetris_macro_env import TetrisPlacementEnv
from dqn.episode_stats import EpisodeStats
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000, help="Placement budget when --episodes is omitted")
    parser.add_argument("--episodes", type=int, default=None, help="Evaluate this many separately seeded games")
    parser.add_argument("--max-placements", type=int, default=10000, help="Per-game cap; capped games are reported separately")
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument('--invalid-action-mode', choices=['reject', 'legacy_drop'], default='legacy_drop')
    parser.add_argument("--obs", type=str, default="board")
    parser.add_argument("--prefix", type=str, default="dqn_tetris_board")
    parser.add_argument(
        "--hold-actions",
        action="store_true",
        help="Enable hold/swap as part of the macro-action space (must match the checkpoint).",
    )
    parser.add_argument(
        "--next-n",
        type=int,
        default=1,
        help="How many upcoming pieces from the queue to include in observation (0 disables). Must match the checkpoint.",
    )
    parser.add_argument(
        "--reward-profile",
        type=str,
        default="tetris",
        choices=["tetris", "survival", "balanced"],
        help="Reward shaping profile (only affects reward, not lines).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Env seed (for reproducible eval).")
    args = parser.parse_args()

    base = gym.make("tetris_gymnasium/Tetris", render_mode=None, gravity=False)
    env = TetrisPlacementEnv(
        base,
        include_hold=bool(args.hold_actions),
        next_n=int(args.next_n),
        reward_profile=str(args.reward_profile),
        observation_mode=args.obs,
        invalid_action_mode=args.invalid_action_mode,
    )

    pattern = str((_REPO_ROOT / "models") / f"{args.prefix}_*_steps.zip")
    zips = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not zips and not args.model_path:
        raise SystemExit(f"No checkpoints found for prefix {args.prefix}")
    model_path = args.model_path or zips[0]
    print(f"Loading: {model_path}")

    model = DQN.load(model_path, env=env)

    if args.steps < 1 or args.max_placements < 1 or (args.episodes is not None and args.episodes < 1):
        parser.error("Budgets and episode count must be positive")
    stats = EpisodeStats()
    rows = []
    try:
        if args.episodes is None:
            obs, _ = env.reset(seed=args.seed)
            for _ in range(args.steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated
                stats.add(info.get("lines_cleared", 0), done)
                if done:
                    obs, _ = env.reset(seed=args.seed + stats.completed)
            result = dict(placements=args.steps, completed_episodes=stats.completed,
                mean_lines_completed=stats.mean_lines, mean_placements_completed=stats.mean_steps,
                unfinished_lines=stats.partial_lines, unfinished_placements=stats.partial_steps,
                best_completed_lines=stats.best_lines)
        else:
            for episode in range(args.episodes):
                obs, _ = env.reset(seed=args.seed+episode)
                lines = 0
                for step in range(1, args.max_placements+1):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = env.step(int(action))
                    lines += int(info.get("lines_cleared", 0))
                    if terminated or truncated:
                        break
                rows.append(dict(seed=args.seed+episode, lines=lines, placements=step,
                                 completed=bool(terminated), capped=not bool(terminated),
                                 environment_truncated=bool(truncated)))
            completed = [r for r in rows if r['completed']]
            result = dict(episodes=rows, completed_episodes=len(completed),
                capped_episodes=len(rows)-len(completed),
                mean_lines_completed=sum(r['lines'] for r in completed)/len(completed) if completed else None,
                mean_lines_at_cap_or_death=sum(r['lines'] for r in rows)/len(rows))
        print(json.dumps(result, indent=2))
        if args.json_output:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(json.dumps(result, indent=2)+'\n')
    finally:
        env.close()


if __name__ == "__main__":
    main()
