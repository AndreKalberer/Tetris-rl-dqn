"""Prepared background endurance worker. Run only after Andre authorizes launch."""
import argparse
import ctypes
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time

JOB=Path(__file__).resolve().parent
SOURCE=JOB/'source'
sys.path.insert(0,str(SOURCE))
SIMULATION_FPS=60
FRAMES_PER_SIMULATED_HOUR=SIMULATION_FPS*60*60
DEFAULT_MAX_SIMULATED_HOURS=100


def simulated_frame_limit(*,hours=None,frames=None):
    """Absolute lifetime limit; hours must resolve to exact whole 60 Hz frames."""
    if hours is not None and frames is not None:
        raise ValueError('Choose simulated hours or frames, not both')
    if frames is not None:
        if isinstance(frames,bool) or not isinstance(frames,int) or frames<=0:
            raise ValueError('Maximum simulated frames must be a positive integer')
        return frames
    try:
        hours=Decimal(str(DEFAULT_MAX_SIMULATED_HOURS if hours is None else hours))
    except InvalidOperation as error:
        raise ValueError('Maximum simulated hours must be a finite positive number') from error
    if not hours.is_finite() or hours<=0:
        raise ValueError('Maximum simulated hours must be a finite positive number')
    value=hours*FRAMES_PER_SIMULATED_HOUR
    if value!=value.to_integral_value():
        raise ValueError('Maximum simulated hours must represent whole 60 Hz frames')
    return int(value)


def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2),encoding='utf-8')
    temporary.replace(path)


def process_limits(cpu_mask):
    details=dict(logical_processors=os.cpu_count(),priority='unchanged',affinity_mask=None)
    if os.name=='nt':
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetCurrentProcess.restype=ctypes.c_void_p
        handle=kernel.GetCurrentProcess()
        kernel.SetPriorityClass.argtypes=(ctypes.c_void_p,ctypes.c_ulong)
        if not kernel.SetPriorityClass(handle,0x20): raise ctypes.WinError(ctypes.get_last_error())
        details['priority']='Normal'
        available=ctypes.c_size_t()
        system=ctypes.c_size_t()
        kernel.GetProcessAffinityMask.argtypes=(ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t))
        if not kernel.GetProcessAffinityMask(handle,ctypes.byref(available),ctypes.byref(system)):
            raise ctypes.WinError(ctypes.get_last_error())
        if cpu_mask<=0 or cpu_mask & ~available.value:
            raise ValueError('CPU mask must select available logical processors')
        selected=[index for index in range(64) if cpu_mask & (1<<index)]
        mask=cpu_mask
        kernel.SetProcessAffinityMask.argtypes=(ctypes.c_void_p,ctypes.c_size_t)
        if not kernel.SetProcessAffinityMask(handle,mask): raise ctypes.WinError(ctypes.get_last_error())
        details['affinity_mask']=mask
        details['logical_processors_allowed']=selected
    return details


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-cores',type=float,default=4.0,help='Average process CPU seconds per wall second')
    parser.add_argument('--cpu-mask',type=lambda text:int(text,0),default=1108,
        help='Allowed logical CPU bitmask, decimal or 0x notation; default1108 selects2/4/6/10')
    parser.add_argument('--resume',type=Path,required=True,help='Explicit trusted continuation state; never reset to the initial preparation')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--checkpoint-seconds',type=float,default=60)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--status-path',type=Path,default=JOB.parents[1]/'outputs/overnight-modern-status.json',
        help='Shared status target; isolated tests must redirect this outside the live outputs')
    cap=parser.add_mutually_exclusive_group()
    cap.add_argument('--max-simulated-hours',help='Absolute lifetime game hours at60FPS; default100 (not100 additional hours)')
    cap.add_argument('--max-simulated-frames',type=int,help='Absolute lifetime frame cap, for exact bounded continuation tests')
    args=parser.parse_args()
    if (not math.isfinite(args.cpu_cores) or args.cpu_cores<=0 or args.threads<1
            or not math.isfinite(args.checkpoint_seconds) or args.checkpoint_seconds<=0
            or args.cpu_mask<=0): parser.error('Finite positive CPU/thread/checkpoint/mask settings required')
    try:
        frame_limit=simulated_frame_limit(hours=args.max_simulated_hours,frames=args.max_simulated_frames)
        limits=process_limits(args.cpu_mask)
    except ValueError as error:
        parser.error(str(error))
    simulation_limit=dict(max_simulated_frames=frame_limit,
        max_simulated_hours=float(Decimal(frame_limit)/FRAMES_PER_SIMULATED_HOUR),
        simulation_fps=SIMULATION_FPS,simulated_time_limit_is_absolute=True)
    for name,expected in json.loads((JOB/'source-manifest.json').read_text()).items():
        if hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen modern source/model hash differs: '+name)
    import torch
    from offline.evaluate import board_hash
    with args.resume.open('rb') as stream:
        saved=pickle.load(stream)
    game,bot=saved['game'],saved['bot']
    if game.over or game.won: raise ValueError('Saved game is already finished')
    if game.rules.fps!=SIMULATION_FPS: raise ValueError('This cap protocol requires the frozen60FPS rules')
    torch.set_num_threads(args.threads)
    limits['torch_threads']=args.threads
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    cpu_started=time.process_time()
    starting=dict(placements=game.placements,lines=game.lines,frames=game.frames)
    parent=saved.get('output')
    checkpoint=str(SOURCE/'models/seed1-100000.pt')
    status_path=args.status_path.resolve()
    protocol=dict(starting=starting,seed=game.seed,beam=bot.beam,cpu_cores=args.cpu_cores,
        **simulation_limit,cpu_mask=args.cpu_mask,
        starting_simulated_hours=game.frames/FRAMES_PER_SIMULATED_HOUR,
        started_at_or_above_limit=game.frames>=frame_limit,status_path=str(status_path),
        worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rules=asdict(game.rules),
        parent_segment=parent,resume_file=str(args.resume.resolve()),
        resume_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest(),
        recovery=saved.get('recovery'),torch_threads=args.threads,
        approximate_total_cpu_percent=args.cpu_cores/(os.cpu_count() or 1)*100,limits=limits,
        line_limit=None,piece_limit=None,wall_time_limit=None,
        policy_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        source_manifest=json.loads((JOB/'source-manifest.json').read_text(encoding='utf-8')),
        original_segment=json.loads((JOB/'parent-segment.json').read_text(encoding='utf-8')),
        scope='Continuation of the original modern offline game; absolute lifetime simulated-time resource cap; no automatic restart after top-out')
    write_json(output/'protocol.json',protocol)
    reason=None
    last_report=-1
    last_report_time=0.0
    last_save=0.0
    def outcome():
        return dict(stop_reason=reason,censored=reason=='simulated_time_limit',
            termination_kind=('resource_limit' if reason=='simulated_time_limit' else
                              'natural_top_out' if reason=='natural_top_out' else reason),
            natural_top_out=game.over,won=game.won,
            simulated_hours=game.frames/FRAMES_PER_SIMULATED_HOUR,**simulation_limit)
    def save():
        temp=output/'state.pkl.tmp'
        with temp.open('wb') as stream:
            pickle.dump(dict(game=game,bot=bot,output=str(output),checkpoint=checkpoint,
                **outcome()),stream,pickle.HIGHEST_PROTOCOL)
        temp.replace(output/'state.pkl')
        write_json(output/'checkpoint-status.json',dict(placements=game.placements,lines=game.lines,
            frames=game.frames,alive=not game.over,**outcome()))
    def report(state):
        elapsed=time.perf_counter()-started
        cpu=time.process_time()-cpu_started
        row=dict(state=state,pid=os.getpid(),seed=game.seed,beam=bot.beam,
            placements=game.placements,lines=game.lines,frames=game.frames,
            additional_placements=game.placements-starting['placements'],additional_lines=game.lines-starting['lines'],
            wall_seconds=elapsed,process_cpu_seconds=cpu,cpu_core_budget=args.cpu_cores,
            average_total_cpu_percent=(cpu/max(elapsed,0.001))/(os.cpu_count() or 1)*100,
            limits=limits,alive=not game.over,output=str(output),**outcome(),
            planning_overruns=bot.overruns,rejected_inputs=game.rejected_inputs,path_mismatches=bot.path_mismatches,
            game_scope='Original modern offline rules, no line/piece cap; absolute simulated-time resource cap',
            final_board_sha256=board_hash(game.board))
        write_json(output/'status.json',row)
        write_json(status_path,row)
        print(json.dumps(row),flush=True)
        return row
    save()
    report('running')
    try:
        with (output/'inputs.jsonl').open('w',encoding='utf-8') as inputs,(output/'locks.jsonl').open('w',encoding='utf-8') as locks:
            while not game.over:
                # Check before bot.command as well as tick: a capped checkpoint
                # must preserve both game and pending controller state exactly.
                if game.frames>=frame_limit:
                    reason='simulated_time_limit'
                    break
                if (JOB/'STOP').exists() or (output/'STOP').exists():
                    reason='requested_stop'
                    break
                command=bot.command(game)
                event=game.tick(command)
                if command!='none': inputs.write(json.dumps(dict(frame=game.frames,action=command,accepted=event['accepted']))+'\n')
                if event['locked']:
                    if bot.expected_board is not None and board_hash(game.board)!=board_hash(bot.expected_board):
                        bot.path_mismatches+=1
                    bot.expected_board=None
                    locks.write(json.dumps(dict(frame=game.frames,lines=game.lines,board_sha256=board_hash(game.board)))+'\n')
                if game.rejected_inputs or bot.path_mismatches:
                    reason='controller_fidelity_error'
                    break
                elapsed=time.perf_counter()-started
                if game.placements!=last_report and game.placements%100==0 or elapsed-last_report_time>=30:
                    inputs.flush(); locks.flush()
                    report('running')
                    last_report=game.placements
                    last_report_time=elapsed
                if elapsed-last_save>=args.checkpoint_seconds:
                    inputs.flush(); locks.flush()
                    save()
                    last_save=elapsed
                # Sleep only between simulation ticks, outside the planner's
                # own timing window. This limits average CPU without changing
                # model weights or the game's fixed simulation-frame rules.
                debt=(time.process_time()-cpu_started)/args.cpu_cores-(time.perf_counter()-started)
                if debt>0: time.sleep(min(debt,1.0))
            if game.over: reason='natural_top_out'
    except KeyboardInterrupt:
        reason='user_interrupt'
    except BaseException:
        reason='execution_error'
        raise
    finally:
        save()
        write_json(output/'results.json',report('finished' if game.over else 'unfinished'))


if __name__=='__main__': main()
