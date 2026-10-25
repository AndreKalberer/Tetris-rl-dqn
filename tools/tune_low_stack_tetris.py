"""Tune tetris-building weights used only while the stack is low, at level-29+ speed.

Survival play (Dellacherie) handles every other position. Fitness is uncapped
points per line minus a heavy cost per top-out (a top-out ends a level-255
attempt). Simulation only (nes_original.sim): confirm in the original game.

  python tools/tune_low_stack_tetris.py --output runs/tune-lowstack-v1
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nes_original import sim  # noqa: E402


def _init(mask):
    if mask:
        from nes_original.verify_batch import limit_cpu
        limit_cpu(mask)


def _game(job):
    w, below, expect, seed, lines = job
    return sim.play_game(sim.DELLACHERIE, sim.DELLACHERIE, seed=seed, start_level=29, initial_score=sim.MAXED,
                         max_lines=lines, beam=3, well=-1, well_survive=-1, expect=expect,
                         alt_weights=w, alt_below=below, alt_well=9)


def fitness(rows, death_cost):
    lines = sum(r['lines'] for r in rows)
    points = sum(r['uncapped_score']-sim.MAXED for r in rows)
    deaths = sum(r['dead'] for r in rows)
    return points/max(1, lines)-death_cost*deaths/len(rows), deaths, points/max(1, lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--start', type=Path, default=ROOT/'runs/tune-score-v1/weights.json')
    p.add_argument('--below', type=int, default=4, help='use the tetris weights while the stack is at most this many rows')
    p.add_argument('--expect', type=int, default=0)
    p.add_argument('--generations', type=int, default=20)
    p.add_argument('--population', type=int, default=20)
    p.add_argument('--games', type=int, default=40)
    p.add_argument('--lines', type=int, default=720, help='lines per game (levels 174-246 span about 720)')
    p.add_argument('--death-cost', type=float, default=300000.)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--cpu-mask', type=lambda v: int(v, 0), default=0)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    mean = np.asarray(json.loads(args.start.read_text())['score'], dtype=float)
    sigma = np.maximum(np.abs(mean)*.3, .05)
    rng = np.random.default_rng(0)
    history, best = [], dict(fitness=-1e18)
    with mp.Pool(args.workers, initializer=_init, initargs=(args.cpu_mask,)) as pool:
        for gen in range(args.generations):
            t = time.perf_counter()
            seeds = [700000+gen*1000+i for i in range(args.games)]
            pop = [mean.copy()]+[mean+sigma*rng.standard_normal(len(mean)) for _ in range(args.population-1)]
            scored = []
            for w in pop:
                rows = pool.map(_game, [(w, args.below, args.expect, s, args.lines) for s in seeds])
                scored.append(fitness(rows, args.death_cost))
            order = np.argsort([f for f, _, _ in scored])[::-1]
            elite = np.asarray([pop[i] for i in order[:max(2, args.population//4)]])
            mean = .3*mean+.7*elite.mean(axis=0)
            sigma = np.maximum(.3*sigma+.7*elite.std(axis=0), .05)
            top = scored[order[0]]
            if top[0] > best['fitness']:
                best = dict(fitness=top[0], deaths=top[1], points_per_line=top[2], weights=pop[order[0]].tolist(), generation=gen)
            history.append(dict(generation=gen, seconds=time.perf_counter()-t, mean_candidate=scored[0], best=top))
            (args.output/'history.json').write_text(json.dumps(dict(args={k: str(v) for k, v in vars(args).items()},
                features=list(sim.FEATURES), history=history, best=best), indent=1))
            print(json.dumps(dict(generation=gen, mean_candidate=scored[0], best=top)), flush=True)
        check = [900000+i for i in range(120)]
        rows = pool.map(_game, [(mean, args.below, args.expect, s, args.lines) for s in check])
        f, deaths, ppl = fitness(rows, args.death_cost)
    result = dict(features=list(sim.FEATURES), alt_weights=mean.tolist(), alt_below=args.below, alt_well=9,
                  expect=args.expect, check=dict(games=120, deaths=deaths, points_per_line=ppl, fitness=f),
                  scope='Fast-simulator result only; level-29+ gravity from an empty board')
    (args.output/'weights.json').write_text(json.dumps(result, indent=1))
    print(json.dumps(result['check']), flush=True)


if __name__ == '__main__':
    main()
