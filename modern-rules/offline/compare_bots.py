"""Head-to-head: learned bot (offline.bot) vs the ported NES bot (offline.classic_bot).

Both bots use the same engine, seeds, reaction wait, input rate and path family
for the move they execute. Each game runs in its own single-threaded process.
Every finished 40-line game is verified by replaying its primitive inputs.

  python -m offline.compare_bots --mode sprint --seeds 91000-91019 --output runs/h2h-sprint
  python -m offline.compare_bots --mode endurance --cap-lines 5000 --seeds 92000-92007 --output runs/h2h-end
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import time


def make_bot(name, beam, depth):
    if name == 'learned':
        import torch
        from dqn.train_candidates import load_policy
        from offline.bot import Bot
        from offline.evaluate import checkpoint_default
        config, model = load_policy(checkpoint_default())
        config = replace(config, threads=1)
        torch.set_num_threads(1)
        return Bot(config, model, beam or 4)
    from offline.classic_bot import ClassicBot, Planner, warmup
    planner = Planner(beam=beam or 8, depth=depth)
    warmup(planner)
    return ClassicBot(planner)


def play(job):
    name, seed, mode, cap_lines, beam, depth, max_seconds, gravity = job
    from offline.engine import Game, Rules
    from offline.evaluate import board_hash, replay
    rules = Rules(gravity_frames=gravity)
    if mode == 'endurance':
        rules = replace(rules, target_lines=None)
    bot = make_bot(name, beam, depth)
    game = Game(seed, rules)
    inputs, locks = [], []
    started = time.perf_counter()
    reason = None
    while True:
        if game.won:
            reason = 'forty_lines'
            break
        if game.over:
            reason = 'top_out'
            break
        if cap_lines and game.lines >= cap_lines:
            reason = 'line_cap'
            break
        if time.perf_counter()-started > max_seconds:
            reason = 'wall_time'
            break
        command = bot.command(game)
        event = game.tick(command)
        if mode == 'sprint' and command != 'none':
            inputs.append(dict(frame=game.frames, action=command, accepted=event['accepted']))
        if event['locked']:
            if bot.expected_board is not None and board_hash(game.board) != board_hash(bot.expected_board):
                bot.path_mismatches += 1
            bot.expected_board = None
            if mode == 'sprint':
                locks.append(dict(frame=game.frames, lines=game.lines, board_sha256=board_hash(game.board)))
    row = dict(bot=name, seed=seed, mode=mode, gravity_frames=gravity, stop_reason=reason, lines=game.lines,
               placements=game.placements, frames=game.frames, simulation_seconds=game.frames/rules.fps,
               wall_seconds=time.perf_counter()-started, plans=bot.plans,
               ms_per_plan=1000*bot.planning_seconds/max(1, bot.plans), planning_overruns=bot.overruns,
               rejected_inputs=game.rejected_inputs, path_mismatches=bot.path_mismatches)
    if mode == 'sprint':
        row['passed'] = (reason == 'forty_lines' and not bot.overruns and not bot.path_mismatches
                         and not game.rejected_inputs)
        if reason == 'forty_lines':
            row['replay_verified'] = replay(dict(row, seed=seed, inputs=inputs, locks=locks,
                                                 final_board_sha256=board_hash(game.board)), rules)
    return row


def seeds(text):
    out = []
    for part in text.split(','):
        a, _, b = part.partition('-')
        out += list(range(int(a), int(b or a)+1))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=('sprint', 'endurance'), required=True)
    parser.add_argument('--seeds', type=seeds, required=True)
    parser.add_argument('--bots', nargs='+', default=['learned', 'classic'])
    parser.add_argument('--cap-lines', type=int, default=0)
    parser.add_argument('--learned-beam', type=int, default=4)
    parser.add_argument('--classic-beam', type=int, default=8)
    parser.add_argument('--classic-depth', type=int, default=3)
    parser.add_argument('--gravity-frames', type=int, default=30)
    parser.add_argument('--max-seconds', type=float, default=3600)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [(bot, seed, args.mode, args.cap_lines,
             args.learned_beam if bot == 'learned' else args.classic_beam, args.classic_depth,
             args.max_seconds, args.gravity_frames) for seed in args.seeds for bot in args.bots]
    with ProcessPoolExecutor(args.workers) as pool, (args.output/'games.jsonl').open('a') as f:
        for row in pool.map(play, jobs):
            f.write(json.dumps(row)+'\n')
            f.flush()
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
