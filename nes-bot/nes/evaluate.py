"""Small legal-input engine probes with deterministic, full-state replay."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from nes.controller import StackingProbe
from nes.engine import Game,Rules


def source_hashes():
    root=Path(__file__).resolve().parents[1]
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in ('nes/engine.py','nes/controller.py','nes/evaluate.py')}


def replay(row):
    game=Game(row['seed'],Rules(**row['rules']))
    inputs=iter(row['inputs'])
    pending=next(inputs,None)
    locks=iter(row['locks'])
    next_lock=next(locks,None)
    buttons=0
    for frame in range(1,row['frames']+1):
        if pending and pending['frame']==frame:
            buttons=pending['buttons']
            pending=next(inputs,None)
        event=game.tick(buttons)
        if event['locked']:
            assert next_lock and next_lock['frame']==frame
            assert game.signature()==next_lock['state_sha256']
            next_lock=next(locks,None)
    assert pending is None and next_lock is None
    assert game.signature()==row['final_state_sha256']
    assert game.lines==row['lines'] and game.placements==row['placements'] and game.score==row['score']
    return game


def episode(seed,start_level=0,max_frames=120000,wall_seconds=30,callback=None):
    game=Game(seed,Rules(start_level=start_level))
    bot=StackingProbe()
    inputs,locks=[],[]
    previous=-1
    started=time.perf_counter()
    while not game.over and game.frames<max_frames and time.perf_counter()-started<wall_seconds:
        buttons=bot.command(game.observe())
        assert not (buttons&3==3), 'Probe must not press both horizontal directions'
        if buttons!=previous:
            inputs.append(dict(frame=game.frames+1,buttons=buttons))
            previous=buttons
        event=game.tick(buttons)
        if event['locked']:
            locks.append(dict(frame=game.frames,state_sha256=game.signature()))
        if callback:
            callback(game)
    reason='natural_top_out' if game.over else 'simulation_frame_limit' if game.frames>=max_frames else 'wall_time_limit'
    row=dict(seed=seed,rules=asdict(game.rules),controller=bot.name,placements=game.placements,lines=game.lines,
        score=game.score,level=game.level,frames=game.frames,simulated_seconds=game.frames/60.0988,
        wall_seconds=time.perf_counter()-started,planning_seconds=bot.planning_seconds,plans=bot.plans,
        stopped_by=reason,completed=game.over,alive_at_stop=not game.over,
        final_state_sha256=game.signature(),inputs=inputs,locks=locks,
        scope='Development engine probe; untrained heuristic, uncharged planning latency, ROM parity pending')
    assert replay(row).signature()==game.signature()
    row['replay_verified']=True
    return row,game


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--start-level',type=int,default=0)
    parser.add_argument('--max-frames',type=int,default=120000)
    parser.add_argument('--wall-seconds',type=float,default=30)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.max_frames<1 or args.wall_seconds<=0:
        parser.error('Limits must be positive')
    Rules(start_level=args.start_level)
    args.output.mkdir(parents=True,exist_ok=False)
    protocol=dict(rules=asdict(Rules(start_level=args.start_level)),seeds=args.seeds,
        max_frames=args.max_frames,wall_seconds=args.wall_seconds,source_sha256=source_hashes(),
        scope='Development sample only; not a trained policy or emulator-parity result')
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    rows=[]
    for seed in args.seeds:
        row,game=episode(seed,args.start_level,args.max_frames,args.wall_seconds)
        rows.append(row)
        (args.output/f'seed{seed}.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
        (args.output/f'seed{seed}-state.json').write_text(json.dumps(game.get_state(),indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in row.items() if k not in ('inputs','locks')}),flush=True)
    summary=dict(protocol=protocol,games=[{k:v for k,v in row.items() if k not in ('inputs','locks')} for row in rows])
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
