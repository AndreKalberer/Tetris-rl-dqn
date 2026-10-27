"""Watch a candidate policy or record a GIF, without a legacy SB3 checkpoint."""
import argparse
from pathlib import Path

import imageio.v2 as imageio
from dqn.candidates import enumerate_candidates, heuristic_scores
from dqn.train_candidates import Config, load_policy, make_env
from dqn.lookahead import choose_lookahead


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', choices=['heuristic', 'learned'], default='heuristic')
    parser.add_argument('--checkpoint', type=Path, help='Weights-only policy.pt export, required for learned policy')
    parser.add_argument('--seed', type=int, default=10000)
    parser.add_argument('--max-placements', type=int, default=1000)
    parser.add_argument('--sleep', type=float, default=.05)
    parser.add_argument('--gif', type=Path, help='Record instead of opening a window')
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--beam', type=int, default=1, help='Learned policy: 1 is original; >1 previews an extra move')
    args = parser.parse_args()
    if args.policy == 'learned' and not args.checkpoint:
        parser.error('--policy learned requires --checkpoint')
    if args.max_placements < 1 or args.fps < 1 or args.beam < 1:
        parser.error('Placement cap, fps, and beam must be positive')
    if args.policy != 'learned' and args.beam != 1:
        parser.error('Extra-move search requires --policy learned')
    config, model = load_policy(args.checkpoint) if args.checkpoint else (Config(), None)
    env = make_env(config, render_mode='rgb_array' if args.gif else 'human')
    if not args.gif:
        env.unwrapped.render_scaling_factor = 20
    frames = []
    lines, done = 0, False
    try:
        env.reset(seed=args.seed)
        for step in range(1, args.max_placements+1):
            cs = enumerate_candidates(env, board_input=config.board_input)
            if args.policy == 'learned':
                idx, _ = choose_lookahead(env, cs, model, gamma=config.gamma,
                    beam=args.beam, board_input=config.board_input)
            else:
                idx = int(heuristic_scores(cs).argmax())
            _, _, terminated, truncated, info = env.step(int(cs.actions[idx]))
            lines += info['lines_cleared']
            frame = env.render()
            if args.gif:
                frames.append(frame)
            else:
                import cv2
                # OpenCV windows need an event pump to paint and accept input.
                cv2.setWindowTitle(env.unwrapped.window_name,
                    f'Tetris bot | {lines:,} lines | {step:,}/{args.max_placements:,} pieces | beam {args.beam} | Q/Esc')
                key = cv2.waitKey(max(1, int(args.sleep*1000))) & 0xff
                if key in (27, ord('q')) or cv2.getWindowProperty(
                    env.unwrapped.window_name, cv2.WND_PROP_VISIBLE
                ) < 1:
                    break
            done = terminated or truncated
            if done:
                break
        print(f'policy={args.policy} seed={args.seed} placements={step} lines={lines} completed={done}')
        if args.gif:
            args.gif.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(args.gif, frames, duration=1000/args.fps, loop=0)
    finally:
        env.close()


if __name__ == '__main__':
    main()
