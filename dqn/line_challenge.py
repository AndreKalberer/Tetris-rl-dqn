"""Evaluate an explicit line-clear goal; resource cutoffs stay inconclusive."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

from dqn.candidates import enumerate_candidates
from dqn.lookahead import choose_lookahead
from dqn.robust_train import write_json
from dqn.train_candidates import load_policy, make_env


def stop_reason(*, lines, target, terminated, truncated, placements, max_placements, seconds, max_seconds):
    if lines >= target:
        return 'line_target_reached'
    if terminated:
        return 'top_out'
    if truncated:
        return 'environment_truncated'
    if placements >= max_placements:
        return 'placement_budget'
    if seconds >= max_seconds:
        return 'time_budget'
    return None


def run(checkpoint, seed, target, beam, max_placements, max_seconds, output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'status.json').exists():
        raise ValueError('Choose a new output directory to preserve previous evidence')
    config,model=load_policy(checkpoint)
    env=make_env(config)
    env.reset(seed=seed)
    status=dict(pid=os.getpid(),state='running',seed=seed,target_lines=target,beam=beam,
        checkpoint=str(checkpoint),policy_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        config=asdict(config),max_placements=max_placements,max_seconds=max_seconds,
        lines=0,placements=0,seconds=0)
    write_json(output/'status.json',status)
    started=time.perf_counter()
    lines=0
    try:
        for step in range(1,max_placements+1):
            cs=enumerate_candidates(env,board_input=config.board_input)
            i,_=choose_lookahead(env,cs,model,gamma=config.gamma,beam=beam,board_input=config.board_input)
            _,_,terminated,truncated,info=env.step(int(cs.actions[i]))
            assert not info['invalid_macro_action']
            lines+=int(info['lines_cleared'])
            elapsed=time.perf_counter()-started
            reason=stop_reason(lines=lines,target=target,terminated=terminated,truncated=truncated,
                placements=step,max_placements=max_placements,seconds=elapsed,max_seconds=max_seconds)
            if not reason and step % 50 == 0 and (output/'STOP').exists():
                reason='user_stop'
            if step % 1000 == 0 or reason:
                status.update(lines=lines,placements=step,seconds=elapsed,
                    placements_per_second=step/elapsed,terminated=bool(terminated),truncated=bool(truncated))
                write_json(output/'status.json',status)
                with (output/'trajectory.jsonl').open('a') as f:
                    f.write(json.dumps(status)+'\n')
                print(json.dumps({k:status[k] for k in ['seed','lines','placements','seconds']}),flush=True)
            if reason:
                status.update(state='complete',stop_reason=reason,
                    outcome='passed' if reason=='line_target_reached' else ('failed' if reason=='top_out' else 'inconclusive'),
                    alive_at_stop=not bool(terminated),
                    scope='Exploratory line-goal challenge in the placement environment; not offline-controls certification')
                write_json(output/'status.json',status)
                write_json(output/'results.json',status)
                print(json.dumps(status),flush=True)
                return status
    except BaseException as error:
        status.update(state='failed',error_type=type(error).__name__,error=str(error))
        write_json(output/'status.json',status)
        raise
    finally:
        env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--target-lines',type=int,default=10000)
    parser.add_argument('--beam',type=int,default=4)
    parser.add_argument('--max-placements',type=int,default=100000)
    parser.add_argument('--max-seconds',type=float,default=900)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if min(args.target_lines,args.beam,args.max_placements,args.max_seconds)<=0:
        parser.error('Targets, beam, and resource budgets must be positive')
    run(args.checkpoint,args.seed,args.target_lines,args.beam,args.max_placements,args.max_seconds,args.output)


if __name__=='__main__':
    main()
