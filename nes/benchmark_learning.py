"""Select NES snapshots on development starts, then run fresh recorded comparisons."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from nes.agent import CandidateController,record_episode
from nes.learn import EVAL_LEVELS,VALIDATION_SEEDS,load_policy,publish_progress,write_json


def evaluate(config,model,policy,seeds,output,account_latency=True):
    games=[]
    for level in EVAL_LEVELS:
        for seed in seeds:
            controller=CandidateController(config,model if policy=='learned' else None,policy,seed,account_latency)
            row,_=record_episode(controller,seed,level)
            write_json(output/f'level{level}-seed{seed}.json',row)
            games.append({k:v for k,v in row.items() if k not in ('inputs','locks')})
    result=dict(policy=policy,games=games,mean_observed_lines=float(np.mean([row['lines'] for row in games])),
        completed_games=sum(row['completed'] for row in games),scope='Small seeded-start sample; censored outcomes remain labeled')
    write_json(output/'summary.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policies',type=Path,nargs='+',required=True)
    parser.add_argument('--warm-policy',type=Path,required=True)
    parser.add_argument('--reserved-seeds',type=int,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    publish_progress(dict(state='running',pid=os.getpid(),phase='latency-accounted-model-selection',
        rl_decisions=50000,teacher_decisions=1000,loss=None))
    protocol=dict(validation_seeds=VALIDATION_SEEDS,reserved_seeds=args.reserved_seeds,levels=EVAL_LEVELS,
        selection='Highest mean observed lines on six fixed-logical-reaction validation cases only; CPU jitter does not change validation trajectories; all candidates retained',
        latency='Three-frame minimum reaction; late plans discarded, their full elapsed frames charged as idle, then replan',
        policies=[dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in args.policies],
        source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
            ('nes/engine.py','nes/actions.py','nes/agent.py','nes/benchmark_learning.py')},
        scope='One training seed, provisional NES simulator, no ROM/human/world-record comparison')
    write_json(args.output/'protocol.json',protocol)
    selections=[]
    for index,path in enumerate(args.policies):
        config,model,metadata=load_policy(path)
        result=evaluate(config,model,'learned',VALIDATION_SEEDS,args.output/'validation'/f'candidate{index}',False)
        selections.append(dict(path=str(path),metadata={k:v for k,v in metadata.items() if k not in ('online','config')},result=result))
        write_json(args.output/'selection.json',selections)
        print(json.dumps(dict(stage='validation',path=str(path),mean_observed_lines=result['mean_observed_lines'])),flush=True)
        publish_progress(dict(state='running',pid=os.getpid(),phase=f'benchmark-validation-{index+1}',
            rl_decisions=50000,teacher_decisions=1000,loss=None))
    selected=max(selections,key=lambda row:row['result']['mean_observed_lines'])
    config,model,metadata=load_policy(selected['path'])
    import shutil
    shutil.copy2(selected['path'],args.output/'policy-best.pt')
    comparisons={}
    for policy in ('random','heuristic','learned'):
        comparisons[policy]=evaluate(config,model,policy,args.reserved_seeds,args.output/'reserved'/policy)
        print(json.dumps(dict(stage='reserved',policy=policy,mean_observed_lines=comparisons[policy]['mean_observed_lines'])),flush=True)
    warm_config,warm_model,_=load_policy(args.warm_policy)
    if asdict(warm_config)!=asdict(config): raise ValueError('Warm model configuration differs')
    comparisons['imitation_only']=evaluate(config,warm_model,'learned',args.reserved_seeds,args.output/'reserved'/'imitation_only')
    write_json(args.output/'results.json',dict(protocol=protocol,selected_path=selected['path'],
        selected_metadata={k:v for k,v in metadata.items() if k not in ('online','config')},
        validation=selections,reserved=comparisons))
    publish_progress(dict(state='finished',pid=os.getpid(),phase='latency-accounted-comparison-complete',
        rl_decisions=50000,teacher_decisions=1000,loss=None))


if __name__=='__main__': main()
