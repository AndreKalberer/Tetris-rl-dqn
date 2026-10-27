"""Frozen-rule forty-line acceptance using timed legal inputs and replay audits."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from dqn.robust_train import write_json
from dqn.train_candidates import load_policy
from offline.engine import Game,Rules
from offline.bot import Bot


def checkpoint_default():
    packaged=Path(__file__).resolve().parents[2]/'old-dqn/models/seed1-100000.pt'
    return packaged if packaged.exists() else Path('runs/robustness-v1/seed1/policy-100000.pt')


def board_hash(board):
    return hashlib.sha256(board.tobytes()).hexdigest()


def episode(config,model,seed,rules,beam=4,max_frames=36000,max_seconds=120,frame_callback=None):
    game=Game(seed,rules)
    bot=Bot(config,model,beam)
    inputs=[]
    locks=[]
    started=time.perf_counter()
    outcome=reason=None
    while game.frames<max_frames:
        command=bot.command(game)
        event=game.tick(command)
        if command!='none':
            inputs.append(dict(frame=game.frames,action=command,accepted=event['accepted']))
        if event['locked']:
            if bot.expected_board is not None and board_hash(game.board)!=board_hash(bot.expected_board):
                bot.path_mismatches+=1
            bot.expected_board=None
            locks.append(dict(frame=game.frames,lines=game.lines,board_sha256=board_hash(game.board)))
        if frame_callback:
            frame_callback(game)
        if game.won:
            outcome,reason='passed','forty_lines'
            break
        if game.over:
            outcome,reason='failed','top_out'
            break
        if time.perf_counter()-started>max_seconds:
            outcome,reason='inconclusive','wall_time_budget'
            break
    if outcome is None:
        outcome,reason='inconclusive','simulation_frame_budget'
    if bot.overruns or bot.path_mismatches or game.rejected_inputs:
        outcome,reason='inconclusive','controller_fidelity_or_planning_budget'
    row=dict(seed=seed,outcome=outcome,stop_reason=reason,lines=game.lines,placements=game.placements,
        frames=game.frames,simulation_seconds=game.frames/rules.fps,
        wall_seconds=time.perf_counter()-started,rejected_inputs=game.rejected_inputs,
        path_mismatches=bot.path_mismatches,planning_overruns=bot.overruns,
        plans=bot.plans,planning_seconds=bot.planning_seconds,
        inputs=inputs,locks=locks,final_board_sha256=board_hash(game.board))
    return row


def replay(row,rules):
    """Verify the outcome by replaying only frame-numbered primitive inputs."""
    game=Game(row['seed'],rules)
    inputs={entry['frame']:entry for entry in row['inputs']}
    locks={entry['frame']:entry for entry in row['locks']}
    for frame in range(1,row['frames']+1):
        entry=inputs.get(frame)
        event=game.tick(entry['action'] if entry else 'none')
        if entry:
            assert event['accepted']==entry['accepted']
        if event['locked']:
            assert frame in locks and locks[frame]['board_sha256']==board_hash(game.board)
            assert locks[frame]['lines']==game.lines
    assert game.lines==row['lines'] and board_hash(game.board)==row['final_board_sha256']
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=None)
    parser.add_argument('--seeds',type=int,nargs='+',default=[1000,1001,1002])
    parser.add_argument('--beam',type=int,default=4)
    parser.add_argument('--output',type=Path,default=Path('runs/offline-development'))
    parser.add_argument('--max-seconds',type=float,default=120)
    args=parser.parse_args()
    checkpoint=args.checkpoint or checkpoint_default()
    config,model=load_policy(checkpoint)
    if config.next_n!=5 or not config.hold:
        parser.error('Offline adapter requires a next-5, hold-enabled candidate policy')
    rules=Rules()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'protocol.json').exists():
        parser.error('Choose a new output directory to preserve evidence')
    write_json(args.output/'protocol.json',dict(rules=asdict(rules),seeds=args.seeds,beam=args.beam,
        checkpoint=str(checkpoint),policy_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        source_sha256={name:hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest()
            for name in ['offline/engine.py','offline/bot.py','offline/evaluate.py','offline/RULES.md']},
        max_seconds=args.max_seconds,required_successes='18 of 20 in the reserved acceptance set'))
    rows=[]
    for seed in args.seeds:
        row=episode(config,model,seed,rules,args.beam,max_seconds=args.max_seconds)
        row['input_replay_verified']=replay(row,rules)
        rows.append(row)
        write_json(args.output/f'game-{seed}.json',row)
        compact={k:v for k,v in row.items() if k not in ['inputs','locks']}
        print(json.dumps(compact),flush=True)
        write_json(args.output/'status.json',dict(completed=len(rows),total=len(args.seeds),latest=compact))
    result=dict(rules=asdict(rules),beam=args.beam,checkpoint=str(checkpoint),
        passed=sum(r['outcome']=='passed' for r in rows),failed=sum(r['outcome']=='failed' for r in rows),
        inconclusive=sum(r['outcome']=='inconclusive' for r in rows),games=rows)
    write_json(args.output/'results.json',result)


if __name__=='__main__':
    main()
