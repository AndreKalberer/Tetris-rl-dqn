"""Tune or compare two-mode evaluator weights in the fast simulator.

Simulation only: approximate between-piece timing and piece sequences. Winners
must still be confirmed in the original game (Mesen) before any claim.

  tune --mode score    from level18 with the fixed survive mode taking over at
                       --switch-level; fitness = score at --max-lines (capped)
                       plus a bonus for reaching the cap sooner, minus a heavy
                       top-out penalty
  tune --mode survive  level29 gravity from an empty board after the cap;
                       fitness = simulated lines per top-out (censored at a cap)
  compare              fixed weight files on the same fresh seeds

Tuning uses the cross-entropy method with shared seeds per generation, then
re-evaluates the final mean on seeds disjoint from every tuning seed.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np

from nes_original import sim

TUNE_SEED_BASE = 100000   # generation g uses TUNE_SEED_BASE + g*1000 + game
CHECK_SEED_BASE = 900000  # disjoint seeds for final checks and comparisons
OBJECTIVE = dict(name='lines',death_cost=0.)  # survive mode; set by --objective
DEATH_PENALTY = 3.        # a top-out before the cap ends any level255 attempt
SPEED_BONUS = .25         # reaching the cap in fewer lines, relative to 600


def _game(job):
    w_score,w_survive,seed,options = job
    return sim.play_game(w_score,w_survive,seed=seed,**options)


def _init_worker():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def mode_options(mode,args):
    if mode=='score':
        return dict(start_level=18,max_lines=args.max_lines,beam=args.beam,well=args.well,
            switch_level=args.switch_level,tap_period=args.tap_period)
    return dict(start_level=29,max_lines=args.max_lines,beam=args.beam,well_survive=args.well,
        initial_score=sim.MAXED,tap_period=args.tap_period,ban=args.ban)


def summarize(rows):
    lines = np.array([r['lines'] for r in rows])
    scores = np.array([r['score'] for r in rows])
    deaths = int(sum(r['dead'] for r in rows))
    maxed = int(sum(r['maxout_lines']>=0 for r in rows))
    tetris_lines = 4*sum(r['tetrises'] for r in rows)
    maxout = [r['maxout_lines'] for r in rows if r['maxout_lines']>=0]
    return dict(games=len(rows),deaths=deaths,maxouts=maxed,maxout_rate=maxed/len(rows),
        maxout_rate_95ci=wilson(maxed,len(rows)),mean_score=float(scores.mean()),
        median_score=float(np.median(scores)),mean_lines=float(lines.mean()),
        median_lines=float(np.median(lines)),total_lines=int(lines.sum()),
        lines_per_topout=float(lines.sum()/max(1,deaths)),
        tetris_rate=tetris_lines/max(1,int(lines.sum())),
        mean_uncapped_score=float(np.mean([r['uncapped_score'] for r in rows])),
        uncapped_points_per_line=float(sum(r['uncapped_score'] for r in rows)/max(1,int(lines.sum()))),
        median_maxout_lines=float(np.median(maxout)) if maxout else None)


def wilson(k,n,z=1.96):
    if not n:
        return [0.,1.]
    p = k/n
    centre = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0.,centre-half),min(1.,centre+half)]


def fitness(mode,rows):
    if mode=='score':
        values = []
        for r in rows:
            value = r['score']/sim.MAXED-DEATH_PENALTY*r['dead']
            if r['maxout_lines']>=0:
                value += SPEED_BONUS*max(0.,1-r['maxout_lines']/600)
            values.append(value)
        return float(np.mean(values))
    total = sum(r['lines'] for r in rows)
    if OBJECTIVE['name']=='points':
        # Uncapped points per line, minus a heavy cost per top-out: a top-out
        # ends the level-255 attempt and every later point.
        points = sum(r['uncapped_score']-sim.MAXED for r in rows)
        deaths = sum(r['dead'] for r in rows)
        return float(points/max(1,total)-OBJECTIVE['death_cost']*deaths/len(rows))
    return float(total/(1+sum(r['dead'] for r in rows)))


def evaluate(pool,mode,w,fixed,seeds,options):
    """Play one game per seed; w is the tuned mode, fixed the other mode."""
    pair = (w,fixed) if mode=='score' else (fixed,w)
    return pool.map(_game,[(pair[0],pair[1],s,options) for s in seeds],chunksize=1)


def load_weights(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('features')!=list(sim.FEATURES):
        raise ValueError(f'{path} was written for a different feature list')
    return data


def named(name):
    base = dict(default_score=sim.DEFAULT_SCORE,default_survive=sim.DEFAULT_SURVIVE,
        retained_prior=sim.RETAINED_PRIOR,dellacherie=sim.DELLACHERIE)
    if name=='score_start':
        return dict(score=sim.SCORE_START.tolist(),survive=sim.DELLACHERIE.tolist(),well=9,well_survive=-1,
            switch_level=29)
    if name in base:
        return dict(score=base[name].tolist(),survive=base[name].tolist() if 'default' not in name
            else sim.DEFAULT_SURVIVE.tolist(),well=-1 if name in ('retained_prior','dellacherie') else 9,
            well_survive=-1)
    data = load_weights(name)
    return dict(score=data['score'],survive=data['survive'],well=data['well'],
        well_survive=data.get('well_survive',-1),switch_level=data.get('switch_level',999))


def tune(args):
    output = args.output
    output.mkdir(parents=True,exist_ok=False)
    start = load_weights(args.start) if args.start else None
    if start:
        mean = np.asarray(start[args.mode],dtype=float)
        fixed = np.asarray(start['survive' if args.mode=='score' else 'score'],dtype=float)
    else:
        mean = (sim.SCORE_START if args.mode=='score' else sim.DELLACHERIE).copy()
        fixed = (sim.DELLACHERIE if args.mode=='score' else sim.SCORE_START).copy()
    sigma = np.maximum(np.abs(mean)*args.relative_sigma,args.min_sigma)
    options = mode_options(args.mode,args)
    rng = np.random.default_rng(args.seed)
    history = []
    best = dict(fitness=-math.inf)
    elite_count = max(2,int(round(args.population*args.elite)))
    with mp.Pool(args.workers,initializer=_init_worker) as pool:
        for generation in range(args.generations):
            started = time.perf_counter()
            seeds = [TUNE_SEED_BASE+generation*1000+i for i in range(args.games)]
            population = [mean.copy()]+[mean+sigma*rng.standard_normal(len(mean))
                for _ in range(args.population-1)]
            scores = []
            for candidate in population:
                rows = evaluate(pool,args.mode,candidate,fixed,seeds,options)
                scores.append(fitness(args.mode,rows))
            order = np.argsort(scores)[::-1]
            elites = np.asarray([population[i] for i in order[:elite_count]])
            mean = (1-args.smoothing)*mean+args.smoothing*elites.mean(axis=0)
            sigma = np.maximum((1-args.smoothing)*sigma+args.smoothing*elites.std(axis=0),args.min_sigma)
            if scores[order[0]]>best['fitness']:
                best = dict(fitness=float(scores[order[0]]),generation=generation,
                    weights=population[order[0]].tolist())
            history.append(dict(generation=generation,seconds=time.perf_counter()-started,
                mean_candidate_fitness=float(scores[0]),best_fitness=float(scores[order[0]]),
                elite_fitness=float(np.mean([scores[i] for i in order[:elite_count]])),
                population_median=float(np.median(scores)),mean=mean.tolist(),sigma=sigma.tolist()))
            write(output/'history.json',dict(mode=args.mode,features=list(sim.FEATURES),
                options=options,args={k:str(v) for k,v in vars(args).items()},history=history,best=best))
            print(json.dumps({k:history[-1][k] for k in ('generation','seconds','mean_candidate_fitness',
                'best_fitness','elite_fitness')}),flush=True)
        check = [CHECK_SEED_BASE+i for i in range(args.check_games)]
        result = dict(features=list(sim.FEATURES),mode=args.mode,tuned='mean',
            well=args.well if args.mode=='score' else 9,well_survive=args.well if args.mode=='survive' else -1,
            switch_level=args.switch_level,
            score=(mean if args.mode=='score' else fixed).tolist(),
            survive=(fixed if args.mode=='score' else mean).tolist(),
            check=summarize(evaluate(pool,args.mode,mean,fixed,check,options)),
            check_seeds=[check[0],check[-1]],options=options,
            scope='Fast-simulator result only; approximate piece timing; not an original-game result')
    write(output/'weights.json',result)
    print(json.dumps(result['check']),flush=True)


def compare(args):
    rows = {}
    seeds = [CHECK_SEED_BASE+args.seed_offset+i for i in range(args.games)]
    with mp.Pool(args.workers,initializer=_init_worker) as pool:
        for name in args.weights:
            w = named(name)
            options = dict(start_level=args.start_level,max_lines=args.max_lines,beam=args.beam,
                well=w['well'],well_survive=w['well_survive'],tap_period=args.tap_period,
                initial_score=sim.MAXED if args.survive else 0,stop_at_maxout=args.stop_at_maxout,
            switch_level=w.get('switch_level',999) if args.switch_level is None else args.switch_level,
            expect=args.expect)
            results = pool.map(_game,[(np.asarray(w['score']),np.asarray(w['survive']),s,options)
                for s in seeds],chunksize=1)
            rows[name] = dict(options=options,summary=summarize(results))
            print(name,json.dumps(rows[name]['summary']),flush=True)
    if args.output:
        write(args.output,dict(seeds=[seeds[0],seeds[-1]],results=rows,
            scope='Fast-simulator comparison only; approximate piece timing'))


def write(path,data):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    os.replace(temporary,path)


def main():
    p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command',required=True)
    t = sub.add_parser('tune')
    t.add_argument('--mode',choices=('score','survive'),required=True)
    t.add_argument('--start',type=Path,help='weights.json to start from (keeps its other mode fixed)')
    t.add_argument('--generations',type=int,default=30)
    t.add_argument('--population',type=int,default=32)
    t.add_argument('--elite',type=float,default=.25)
    t.add_argument('--games',type=int,default=48)
    t.add_argument('--check-games',type=int,default=200)
    t.add_argument('--relative-sigma',type=float,default=.5)
    t.add_argument('--min-sigma',type=float,default=.05)
    t.add_argument('--smoothing',type=float,default=.7)
    t.add_argument('--seed',type=int,default=0)
    t.add_argument('--output',type=Path,required=True)
    c = sub.add_parser('compare')
    c.add_argument('weights',nargs='+',help='weights.json paths or built-ins: retained_prior dellacherie default_score')
    c.add_argument('--games',type=int,default=200)
    c.add_argument('--seed-offset',type=int,default=0)
    c.add_argument('--start-level',type=int,default=18)
    c.add_argument('--survive',action='store_true',help='Start already capped (survive mode)')
    c.add_argument('--stop-at-maxout',action='store_true')
    c.add_argument('--output',type=Path)
    c.add_argument('--expect',type=int,default=0,help='Also average over the unknown piece after next')
    for q in (t,c):
        q.add_argument('--beam',type=int,default=3)
        q.add_argument('--max-lines',type=int,default=500)
        q.add_argument('--tap-period',type=int,default=2)
        q.add_argument('--workers',type=int,default=max(1,(os.cpu_count() or 2)-4))
    t.add_argument('--well',type=int,default=None,help='Tuned mode well column (default 9 score, -1 survive)')
    t.add_argument('--switch-level',type=int,default=29)
    t.add_argument('--ban',type=int,default=0,help='survive mode: bitmask of banned clear sizes (2 = singles)')
    t.add_argument('--objective',choices=('lines','points'),default='lines',
                   help='survive mode: lines per top-out, or uncapped points per line minus --death-cost per top-out')
    t.add_argument('--death-cost',type=float,default=200000.,help='points-per-line cost of one top-out per game')
    c.add_argument('--switch-level',type=int,default=None,help='Override each weight set switch level')
    args = p.parse_args()
    if args.command=='tune' and args.well is None:
        args.well = 9 if args.mode=='score' else -1
    if args.command=='tune':
        OBJECTIVE.update(name=args.objective,death_cost=args.death_cost)
    tune(args) if args.command=='tune' else compare(args)


if __name__=='__main__':
    main()
