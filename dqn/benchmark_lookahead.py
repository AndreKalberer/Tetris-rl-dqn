"""Matched-seed development comparison of one- and two-ply frozen policies."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from dqn.candidates import enumerate_candidates
from dqn.lookahead import choose_lookahead
from dqn.train_candidates import load_policy, make_env


def benchmark(checkpoint, seeds, cap, beams, output, *, on_progress=None, stop_file=None):
    config, model = load_policy(checkpoint)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        for beam in beams:
            env = make_env(config)
            env.reset(seed=seed)
            lines = second_moves = 0
            started = time.perf_counter()
            try:
                for step in range(1, cap+1):
                    cs = enumerate_candidates(env, board_input=config.board_input)
                    idx, search = choose_lookahead(env, cs, model, gamma=config.gamma,
                                                  beam=beam, board_input=config.board_input)
                    second_moves += search['evaluated_second_moves']
                    _, _, terminated, truncated, info = env.step(int(cs.actions[idx]))
                    assert not info['invalid_macro_action']
                    lines += info['lines_cleared']
                    if on_progress and step % 500 == 0:
                        on_progress(dict(seed=seed, beam=beam, lines=int(lines), placements=step))
                    if stop_file and step % 50 == 0 and Path(stop_file).exists():
                        raise KeyboardInterrupt('Benchmark stop requested')
                    if terminated or truncated:
                        break
            finally:
                env.close()
            elapsed = time.perf_counter()-started
            row = dict(seed=seed, beam=beam, lines=int(lines), placements=step,
                       completed=bool(terminated), capped=not bool(terminated),
                       seconds=elapsed, placements_per_second=step/elapsed,
                       evaluated_second_moves=second_moves)
            rows.append(row)
            print(json.dumps(row), flush=True)
            if on_progress:
                on_progress(dict(row, episode_finished=True))
            with (output/'episodes.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
    results = dict(scope='Bounded development comparison, one frozen training seed; no historical claim',
                   checkpoint=str(checkpoint), config=asdict(config), placement_cap=cap, episodes=rows)
    (output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[30000, 30001, 30002])
    parser.add_argument('--cap', type=int, default=1000)
    parser.add_argument('--beams', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--output', type=Path, default=Path('runs/lookahead-development'))
    args = parser.parse_args()
    if args.cap < 1 or min(args.beams) < 1:
        parser.error('Cap and beams must be positive')
    benchmark(args.checkpoint, args.seeds, args.cap, args.beams, args.output)


if __name__ == '__main__':
    main()
