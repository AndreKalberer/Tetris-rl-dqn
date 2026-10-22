"""Freeze a fresh-start original-game comparison of selected training seeds."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from nes_original.learn import Config,cpu_limits,evaluate,load_policy,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job',type=Path,required=True,help='Finished training directory containing seedN')
    p.add_argument('--additional-job',type=Path,action='append',default=[],help='Completed continuation directories')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[1601,1613,1627])
    p.add_argument('--cpu-mask',type=int,default=3)
    p.add_argument('--jit-deps',type=Path)
    p.add_argument('--mesen',type=Path,help='Use an existing portable reference build for this comparison')
    p.add_argument('--workers',type=int,default=1,help='Independent games at once; default preserves serial evaluation')
    p.add_argument('--worker-cpu-masks',type=lambda value:int(value,0),nargs='+',default=[1,256])
    p.add_argument('--fixed-reaction',action='store_true',help='Logical-timing diagnostic; default charges runtime latency')
    args=p.parse_args()
    if args.jit_deps:
        sys.path.insert(0,str(args.jit_deps.resolve()))
    cpu_limits(args.cpu_mask)
    if args.output.exists():
        p.error('Use a new output directory; comparisons must remain frozen')
    if not 1<=args.workers<=4 or len(args.worker_cpu_masks)<args.workers:
        p.error('Use1..4workers with an explicit CPU mask for each')
    active_masks=args.worker_cpu_masks[:args.workers]
    if any(mask<=0 for mask in active_masks) or any(a&b for i,a in enumerate(active_masks) for b in active_masks[i+1:]):
        p.error('Worker CPU masks must be positive and disjoint')
    jobs=[args.job,*args.additional_job]
    if any((job/'RUNNING.lock').exists() for job in jobs):
        p.error('Finish/checkpoint the study before freezing a final comparison')
    models=[]
    for folder in sorted(folder for job in jobs for folder in job.glob('seed*')):
        policy=folder/'best.pt'
        if policy.is_file():
            _,metadata=load_policy(policy)
            models.append(dict(seed=metadata['config']['seed'],path=str(policy.resolve()),
                sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
                decisions=metadata['decisions'],config=metadata['config']))
    if not models:
        p.error('No selected policies found')
    if len({row['seed'] for row in models})!=len(models):
        p.error('Duplicate training seeds across supplied directories')
    profile={(row['config'].get('prior','heuristic'),row['config'].get('tap_period',0)) for row in models}
    if len(profile)!=1:
        p.error('Policies do not share the same prior/controller profile')
    args.output.mkdir(parents=True)
    from nes_original.env import OriginalGame
    game_factory=OriginalGame
    if args.mesen:
        from nes_original.mesen import MesenGame
        game_factory=lambda:MesenGame(executable=args.mesen,session_root=args.output/'reference-sessions')
    # Written before the first tested start. No post-result model reselection.
    write_json(args.output/'protocol.json',dict(models=models,test_menu_wait_seeds=args.seeds,
        starting_level=18,target_level=30,runtime_latency_charged=not args.fixed_reaction,
        selected_only_on_prior_validation=[1001,1013],cpu_mask=args.cpu_mask,
        workers=args.workers,worker_cpu_masks=active_masks if args.workers>1 else None,
        backend='MesenGame' if args.mesen else 'OriginalGame/nes-py',
        scope='Small fresh-start original-game pilot, not20-game acceptance or hardware-fidelity certification',
        independence='Independent training seeds; paired initial starts, timing-dependent later piece sequences'))
    def run_cases(policy,folder,cfg):
        if args.workers>1:
            from nes_original.parallel_evaluate import evaluate_parallel
            return evaluate_parallel(policy,folder,seeds=args.seeds,config=cfg,
                workers=args.workers,masks=active_masks,mesen=args.mesen,jit_deps=args.jit_deps,runtime=not args.fixed_reaction)
        model=load_policy(policy)[0] if policy else None
        return evaluate(model,folder,seeds=args.seeds,runtime=not args.fixed_reaction,config=Config(**cfg),game_factory=game_factory)
    result=dict(protocol='protocol.json',baseline=run_cases(None,args.output/'baseline',models[0]['config']),models=[])
    write_json(args.output/'results.json',result)
    for frozen in models:
        path=Path(frozen['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=frozen['sha256']:
            raise RuntimeError('Selected model changed after protocol freeze')
        rows=run_cases(path,args.output/f"training-seed{frozen['seed']}",frozen['config'])
        result['models'].append(dict(training_seed=frozen['seed'],selected_decisions=frozen['decisions'],games=rows))
        write_json(args.output/'results.json',result)
        print(json.dumps(dict(seed=frozen['seed'],selected=frozen['decisions'],lines=[r['lines'] for r in rows],
            successes=sum(r['success'] for r in rows),errors=[r['error'] for r in rows])),flush=True)


if __name__=='__main__':
    main()
