"""Tune the no-singles zone weights (levels 156-158) on the crash-zone simulator.

Fitness = share of games passing levels 150-159 ('mid' stretch of tools/zone_sim.py)
with the real bot's selector, crash map and other zones. Simulation only.

  python nes-bot/tools/tune_zone.py --output runs/tune-zone-157-v1
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # nes-bot/
REPO = ROOT.parent  # repository root: runs/ and progress.md live here
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))
from nes_original import sim  # noqa: E402
import zone_sim  # noqa: E402

EXTRA = [[str(REPO/'runs/lowstack-246-249.json'), 246, 249],
         [str(REPO/'runs/dellacherie-zone.json'), 168, 172, 28, 2],
         [str(REPO/'runs/dellacherie-zone.json'), 173, 173, 4, 2]]


def config_for(path):
    return dict(beam=3, zone_weights=str(path), zone_levels=(156, 158), zone_ban=2, zone_expect=2,
                extra_zones=EXTRA, preempt=True, risk_budget=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--start', type=Path, default=REPO/'runs/tune-nosingles-v2/weights.json')
    p.add_argument('--generations', type=int, default=15)
    p.add_argument('--population', type=int, default=14)
    p.add_argument('--games', type=int, default=160)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = json.loads(args.start.read_text())
    mean = np.asarray(start[start.get('mode', 'survive')], dtype=float)
    sigma = np.maximum(np.abs(mean)*.25, .05)
    rng = np.random.default_rng(1)
    history, best = [], dict(pass_rate=-1)
    with mp.Pool(args.workers) as pool:
        for gen in range(args.generations):
            t = time.perf_counter()
            pop = [mean.copy()]+[mean+sigma*rng.standard_normal(len(mean)) for _ in range(args.population-1)]
            rates = []
            for k, w in enumerate(pop):
                path = args.output/f'cand-g{gen}-{k}.json'
                path.write_text(json.dumps(dict(features=list(sim.FEATURES), mode='survive', survive=w.tolist(),
                                                well_survive=-1)))
                rows = pool.map(zone_sim.play, [(50000+gen*1000+i, 'mid', config_for(path)) for i in range(args.games)],
                                chunksize=4)
                rates.append(sum(r['result'] == 'pass' for r in rows)/len(rows))
            order = np.argsort(rates)[::-1]
            elite = np.asarray([pop[i] for i in order[:4]])
            mean = .4*mean+.6*elite.mean(axis=0)
            sigma = np.maximum(.4*sigma+.6*elite.std(axis=0), .03)
            if rates[order[0]] > best['pass_rate']:
                best = dict(pass_rate=rates[order[0]], generation=gen, weights=pop[order[0]].tolist())
            history.append(dict(generation=gen, seconds=time.perf_counter()-t, mean_candidate=rates[0],
                                best=rates[order[0]], median=float(np.median(rates))))
            print(json.dumps(history[-1]), flush=True)
            final = args.output/'weights.json'
            final.write_text(json.dumps(dict(features=list(sim.FEATURES), mode='survive', survive=mean.tolist(),
                well_survive=-1, best=best, history=history,
                scope='Crash-zone simulator result (levels 150-159); confirm in the original game'), indent=1))
        rows = pool.map(zone_sim.play, [(90000+i, 'mid', config_for(final)) for i in range(600)], chunksize=4)
        rate = sum(r['result'] == 'pass' for r in rows)/len(rows)
        base = pool.map(zone_sim.play, [(90000+i, 'mid', config_for(args.start)) for i in range(600)], chunksize=4)
        base_rate = sum(r['result'] == 'pass' for r in base)/len(base)
    print(json.dumps(dict(check_games=600, tuned_pass=rate, start_pass=base_rate)), flush=True)


if __name__ == '__main__':
    main()
