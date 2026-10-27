"""One offline game to natural top-out, with no placement/line/frame limit.

The wall-time watchdog is a resource limit; hitting it is unfinished/censored.
Every non-none input and locked board is streamed for an independent replay.
Own-process state is saved on exit so an unfinished game can be continued.
"""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

from dqn.robust_train import write_json
from dqn.train_candidates import load_policy
from offline.bot import Bot
from offline.engine import Game,Rules
from offline.evaluate import board_hash,checkpoint_default


def report(output,status):
    write_json(output/'status.json',status)
    root=Path(__file__).resolve().parents[2]  # repository root
    task=root.parents[1]
    if (task/'outputs/progress.md').exists():
        p=task/'outputs/progress.md'
        s=p.read_text(encoding='utf-8')
        a=s.index('Active work:')
        b=s.index('\n\n',a)
        text=(f"Offline endurance experiment: {status['state']}; seed {status['seed']}; "
              f"{status['placements']:,} pieces placed, {status['lines']:,} lines cleared; "
              f"{status['wall_seconds']:.1f}s elapsed wall time. No line/piece/frame cutoff. "
              f"Resource watchdog: {status['wall_budget_seconds']:.0f}s; a cutoff is unfinished, not a loss.")
        s=s[:a]+'Active work: '+text+s[b:]
        for target in (p,root/'progress.md'):
            temporary=target.with_suffix('.md.tmp')
            temporary.write_text(s,encoding='utf-8')
            temporary.replace(target)
        write_json(task/'outputs/endurance-status.json',status)


def run(checkpoint,seed,beam,wall_seconds,output,resume=None):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'protocol.json').exists():
        raise ValueError('Choose a fresh output directory to preserve evidence')
    if resume:
        # Only load trusted local state produced by this script.
        with Path(resume).open('rb') as f:
            saved=pickle.load(f)
        game,bot=saved['game'],saved['bot']
        if game.over or game.won:
            raise ValueError('This game has already ended')
        import torch
        torch.set_num_threads(bot.config.threads)
        # Continue relative to the supplied state, including after a backup move.
        parent=str(Path(resume).resolve().parent)
        checkpoint=saved['checkpoint']
        seed=game.seed
        beam=bot.beam
    else:
        config,model=load_policy(checkpoint)
        if config.next_n!=5 or not config.hold:
            raise ValueError('Endurance adapter requires a next-5, hold-enabled policy')
        game=Game(seed,replace(Rules(),target_lines=None))
        bot=Bot(config,model,beam)
        parent=None
    protocol=dict(seed=seed,beam=beam,checkpoint=str(checkpoint),rules=asdict(game.rules),
        policy_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        wall_budget_seconds=wall_seconds,parent_segment=parent,starting_frame=game.frames,
        line_limit=None,placement_limit=None,simulation_frame_limit=None,
        source_sha256={name:hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest()
            for name in ['offline/engine.py','offline/bot.py','offline/endurance.py']},
        scope='One preselected piece sequence; natural top-out measures this game, not average strength')
    write_json(output/'protocol.json',protocol)
    wall_start=time.perf_counter()
    cpu_start=time.process_time()
    start_frames=game.frames
    last_report=-1
    reason=None
    status={}
    try:
        with (output/'inputs.jsonl').open('w',encoding='utf-8') as inputs, (output/'locks.jsonl').open('w',encoding='utf-8') as locks:
            while not game.over:
                elapsed=time.perf_counter()-wall_start
                if elapsed>=wall_seconds:
                    reason='wall_time_budget'
                    break
                if game.frames%60==0 and (output/'STOP').exists():
                    reason='user_stop'
                    break
                command=bot.command(game)
                event=game.tick(command)
                if command!='none':
                    inputs.write(json.dumps(dict(frame=game.frames,action=command,accepted=event['accepted']))+'\n')
                if event['locked']:
                    if bot.expected_board is not None and board_hash(game.board)!=board_hash(bot.expected_board):
                        bot.path_mismatches+=1
                    bot.expected_board=None
                    locks.write(json.dumps(dict(frame=game.frames,lines=game.lines,board_sha256=board_hash(game.board)))+'\n')
                if game.rejected_inputs or bot.path_mismatches:
                    reason='controller_fidelity_error'
                    break
                if game.placements!=last_report and (game.placements%100==0 or game.over):
                    elapsed=time.perf_counter()-wall_start
                    status=dict(state='running',pid=os.getpid(),seed=seed,beam=beam,
                        placements=game.placements,lines=game.lines,frames=game.frames,
                        simulation_seconds=game.frames/game.rules.fps,wall_seconds=elapsed,
                        process_cpu_seconds=time.process_time()-cpu_start,wall_budget_seconds=wall_seconds,
                        planning_overruns=bot.overruns,rejected_inputs=game.rejected_inputs,path_mismatches=bot.path_mismatches)
                    report(output,status)
                    with (output/'milestones.jsonl').open('a',encoding='utf-8') as f:
                        f.write(json.dumps(status)+'\n')
                    print(json.dumps(status),flush=True)
                    inputs.flush(); locks.flush()
                    last_report=game.placements
            if game.over:
                reason='natural_top_out'
    except KeyboardInterrupt:
        reason='user_interrupt'
    except Exception as error:
        reason='execution_error'
        status.update(error_type=type(error).__name__,error=str(error))
        raise
    finally:
        elapsed=time.perf_counter()-wall_start
        status.update(state='finished' if reason=='natural_top_out' else 'unfinished',
            pid=os.getpid(),seed=seed,beam=beam,stop_reason=reason,placements=game.placements,
            lines=game.lines,frames=game.frames,simulation_seconds=game.frames/game.rules.fps,
            segment_frames=game.frames-start_frames,wall_seconds=elapsed,
            process_cpu_seconds=time.process_time()-cpu_start,wall_budget_seconds=wall_seconds,
            planning_overruns=bot.overruns,rejected_inputs=game.rejected_inputs,path_mismatches=bot.path_mismatches,
            final_board_sha256=board_hash(game.board),alive_at_stop=not game.over,
            counts_valid=not bool(game.rejected_inputs or bot.path_mismatches),parent_segment=parent)
        temp=output/'state.pkl.tmp'
        with temp.open('wb') as f:
            pickle.dump(dict(game=game,bot=bot,output=str(output.resolve()),checkpoint=str(checkpoint)),f,protocol=pickle.HIGHEST_PROTOCOL)
        temp.replace(output/'state.pkl')
        write_json(output/'results.json',status)
        report(output,status)
        print(json.dumps(status),flush=True)
    return status


def replay_segment(output):
    output=Path(output)
    protocol=json.loads((output/'protocol.json').read_text())
    result=json.loads((output/'results.json').read_text())
    if protocol['parent_segment']:
        # Prefer the adjacent preserved segment over its original machine path.
        adjacent=output.parent/Path(protocol['parent_segment']).name
        parent=adjacent if (adjacent/'protocol.json').exists() else Path(protocol['parent_segment'])
        game=replay_segment(parent)
    else:
        game=Game(protocol['seed'],Rules(**protocol['rules']))
    with (output/'inputs.jsonl').open(encoding='utf-8') as f:
        entries=iter(json.loads(line) for line in f)
        pending=next(entries,None)
        with (output/'locks.jsonl').open(encoding='utf-8') as lf:
            lock_entries=iter(json.loads(line) for line in lf)
            next_lock=next(lock_entries,None)
            for frame in range(game.frames+1,result['frames']+1):
                entry=pending if pending and pending['frame']==frame else None
                event=game.tick(entry['action'] if entry else 'none')
                if entry:
                    assert event['accepted']==entry['accepted']
                    pending=next(entries,None)
                if event['locked']:
                    assert next_lock and next_lock['frame']==frame
                    assert next_lock['lines']==game.lines and next_lock['board_sha256']==board_hash(game.board)
                    next_lock=next(lock_entries,None)
            assert pending is None and next_lock is None
    assert game.lines==result['lines'] and game.placements==result['placements']
    assert board_hash(game.board)==result['final_board_sha256']
    return game


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=None)
    parser.add_argument('--seed',type=int,default=92000)
    parser.add_argument('--beam',type=int,default=4)
    parser.add_argument('--wall-seconds',type=float,default=1800)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',type=Path,help='Trusted local state.pkl; use a new output directory')
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if args.verify_only:
        game=replay_segment(args.output)
        print(json.dumps(dict(replay_verified=True,placements=game.placements,lines=game.lines)))
        return
    if args.wall_seconds<=0 or args.beam<1:
        parser.error('Wall budget and beam must be positive')
    run(args.checkpoint or checkpoint_default(),args.seed,args.beam,args.wall_seconds,args.output,args.resume)


if __name__=='__main__':
    main()
