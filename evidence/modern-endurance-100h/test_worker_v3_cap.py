"""Isolated genuine continuation tests; never modify live worker/state/source.

Only new worker_v3 and validation files are written. No game/RAM/counter
injections: short limits refer to the immutable recovered game's actual frames.
"""
import ctypes
from dataclasses import asdict,is_dataclass
from collections import deque
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
sys.dont_write_bytecode=True

HERE=Path(__file__).resolve().parent
JOB=HERE.parent
SOURCE=JOB/'source'
WORKER=JOB/'worker_v3.py'
RECOVERED=JOB/'recovered-last-log-20261022.pkl'
sys.path.insert(0,str(SOURCE))


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin_validation():
    if os.name=='nt':
        k=ctypes.WinDLL('kernel32',use_last_error=True)
        k.GetCurrentProcess.restype=ctypes.c_void_p
        k.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
        if not k.SetProcessAffinityMask(k.GetCurrentProcess(),64): raise ctypes.WinError(ctypes.get_last_error())


def canonical(value):
    import numpy as np
    import torch
    if is_dataclass(value): return canonical(asdict(value))
    if isinstance(value,np.ndarray):
        return dict(dtype=str(value.dtype),shape=list(value.shape),sha256=hashlib.sha256(value.tobytes()).hexdigest())
    if isinstance(value,torch.Tensor): return canonical(value.detach().cpu().numpy())
    if isinstance(value,dict): return {str(k):canonical(v) for k,v in sorted(value.items(),key=lambda item:str(item[0]))}
    if isinstance(value,(list,tuple,deque)): return [canonical(item) for item in value]
    if isinstance(value,(str,int,float,bool)) or value is None: return value
    if hasattr(value,'getstate'): return canonical(value.getstate())
    if hasattr(value,'__dict__'): return canonical(vars(value))
    raise TypeError(type(value))


def game_signature(game): return canonical(vars(game))


def bot_signature(bot,*,include_wall_timing=False):
    result={key:canonical(value) for key,value in vars(bot).items()
        if key not in ('model','planning_seconds')}
    result['model_state_dict']=canonical(bot.model.state_dict())
    result['model_training']=bot.model.training
    if include_wall_timing: result['planning_seconds']=bot.planning_seconds
    return result


def load(path):
    with Path(path).open('rb') as stream: return pickle.load(stream)


def run_segment(root,name,resume,limit):
    output=root/name
    command=[sys.executable,str(WORKER),'--resume',str(resume),'--output',str(output),
        '--max-simulated-frames',str(limit),'--cpu-mask','64','--cpu-cores','1',
        '--threads','2','--checkpoint-seconds','60','--status-path',str(output/'isolated-public-status.json')]
    result=subprocess.run(command,cwd=SOURCE,capture_output=True,text=True,timeout=120,
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    (root/(name+'.stdout.log')).write_text(result.stdout,encoding='utf-8')
    (root/(name+'.stderr.log')).write_text(result.stderr,encoding='utf-8')
    assert result.returncode==0,(name,result.returncode,result.stderr)
    saved=load(output/'state.pkl')
    status=json.loads((output/'results.json').read_text())
    checkpoint_status=json.loads((output/'checkpoint-status.json').read_text())
    protocol=json.loads((output/'protocol.json').read_text())
    game,bot=saved['game'],saved['bot']
    assert game.frames==limit,(name,game.frames,limit)
    for data in (status,checkpoint_status,saved):
        assert data['stop_reason']=='simulated_time_limit',data
        assert data['termination_kind']=='resource_limit' and data['censored'] is True
        assert data['natural_top_out'] is False and data['won'] is False
        assert data['max_simulated_frames']==limit and data['simulated_time_limit_is_absolute'] is True
        assert data['simulated_hours']==limit/216000
        assert data['simulation_fps']==60
    assert status['alive'] is True and not game.over and not game.won
    assert status['rejected_inputs']==game.rejected_inputs==0
    assert status['path_mismatches']==bot.path_mismatches==0
    assert status['limits']['affinity_mask']==64 and status['limits']['torch_threads']==2
    assert status['cpu_core_budget']==1
    assert protocol['max_simulated_frames']==limit and protocol['simulation_fps']==60
    assert Path(protocol['status_path']).is_relative_to(HERE)
    assert protocol['resume_sha256']==sha(resume)
    assert status['frames']==game.frames
    return saved,status,protocol,output


def replay_physics(before,after,output):
    """Verify full saved game continuation from actual primitive input log."""
    game=pickle.loads(pickle.dumps(before['game'],pickle.HIGHEST_PROTOCOL))
    actions={row['frame']:row['action'] for line in (output/'inputs.jsonl').read_text().splitlines()
             if (row:=json.loads(line))}
    for frame in range(game.frames+1,after['game'].frames+1): game.tick(actions.get(frame,'none'))
    assert game_signature(game)==game_signature(after['game'])


def main():
    pin_validation()
    before_immutable={str(RECOVERED):sha(RECOVERED),str(JOB/'worker_v2.py'):sha(JOB/'worker_v2.py')}
    manifest=json.loads((JOB/'source-manifest.json').read_text())
    for name,expected in manifest.items(): assert sha(SOURCE/name)==expected,name
    spec=importlib.util.spec_from_file_location('isolated_worker_v3',WORKER)
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    assert worker.simulated_frame_limit()==21_600_000
    assert worker.simulated_frame_limit(hours='100')==21_600_000
    assert worker.simulated_frame_limit(hours='0.5')==108_000
    for invalid in ('0','-1','NaN','Infinity','-Infinity','garbage','0.000001'):
        try: worker.simulated_frame_limit(hours=invalid)
        except ValueError: pass
        else: raise AssertionError(('accepted invalid hours',invalid))
    for invalid in (0,-1,1.5,True):
        try: worker.simulated_frame_limit(frames=invalid)
        except ValueError: pass
        else: raise AssertionError(('accepted invalid frames',invalid))
    import torch
    torch.set_num_threads(2)
    recovered=load(RECOVERED)
    original_frames=recovered['game'].frames
    assert (original_frames,recovered['game'].lines)==(7264323,70318)
    root=HERE/('cap-'+time.strftime('%Y%m%d-%H%M%S'))
    root.mkdir()
    first,first_status,first_protocol,first_path=run_segment(root,'plus30',RECOVERED,original_frames+30)
    second,second_status,second_protocol,second_path=run_segment(root,'plus60',first_path/'state.pkl',original_frames+60)
    same,same_status,same_protocol,same_path=run_segment(root,'already-at-cap',second_path/'state.pkl',original_frames+60)
    replay_physics(recovered,first,first_path)
    replay_physics(first,second,second_path)
    assert first_protocol['starting']['frames']==original_frames
    assert second_protocol['starting']==dict(placements=first['game'].placements,
        lines=first['game'].lines,frames=first['game'].frames)
    # No extra bot.command or tick at a checkpoint already on the absolute cap.
    assert game_signature(same['game'])==game_signature(second['game'])
    assert bot_signature(same['bot'],include_wall_timing=True)==bot_signature(second['bot'],include_wall_timing=True)
    assert (same_path/'inputs.jsonl').read_text()=='' and (same_path/'locks.jsonl').read_text()==''
    # Independently replay the unsegmented frozen controller for 60 genuine ticks.
    reference=load(RECOVERED)
    from offline.evaluate import board_hash
    direct_actions=[]
    while reference['game'].frames<original_frames+60:
        game,bot=reference['game'],reference['bot']
        command=bot.command(game);event=game.tick(command)
        direct_actions.append((game.frames,command))
        if event['locked']:
            if bot.expected_board is not None and board_hash(game.board)!=board_hash(bot.expected_board): bot.path_mismatches+=1
            bot.expected_board=None
    assert game_signature(reference['game'])==game_signature(second['game'])
    assert bot_signature(reference['bot'])==bot_signature(second['bot'])
    invalid_cases=[['--max-simulated-hours',value] for value in ('0','-1','NaN','Infinity','garbage','0.000001')]
    invalid_cases += [['--max-simulated-frames',value] for value in ('0','-1','1.5')]
    invalid_cases += [['--max-simulated-hours','100','--max-simulated-frames','10'],
        ['--cpu-mask','0'],['--cpu-cores','NaN'],['--threads','0']]
    for index,extra in enumerate(invalid_cases):
        output=root/('invalid-'+str(index))
        result=subprocess.run([sys.executable,str(WORKER),'--resume',str(RECOVERED),'--output',str(output),
            '--status-path',str(root/'unused-status.json'),*extra],capture_output=True,text=True,timeout=15,
            env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        assert result.returncode==2,(extra,result.returncode,result.stderr)
        assert not output.exists(),extra
    for path,expected in before_immutable.items(): assert sha(path)==expected,path
    for name,expected in manifest.items(): assert sha(SOURCE/name)==expected,name
    summary=dict(passed=True,scope='isolated continuation of immutable recovered modern game; no source/policy/physics/counter modifications',
        recovered_sha256=before_immutable[str(RECOVERED)],worker_v3_sha256=sha(WORKER),
        original_frames=original_frames,original_lines=recovered['game'].lines,
        default_hours=100,default_absolute_frame_limit=worker.simulated_frame_limit(),
        remaining_frames_at_default_limit=worker.simulated_frame_limit()-original_frames,
        checkpoints=[dict(path=str(first_path),**first_status),dict(path=str(second_path),**second_status)],
        resume_already_at_cap_preserves_full_game_and_bot=True,
        full_game_physics_log_replay_exact=True,unsegmented_60_frame_game_and_controller_exact=True,
        controller_wall_planning_seconds_excluded_from_unsegmented_comparison=True,
        immutable_recovered_v2_and_frozen_source_unchanged=True,invalid_cli_cases=len(invalid_cases),
        validation_affinity_mask=64,torch_threads=2,shared_live_status_written=False)
    (root/'validation.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    (HERE/'latest-cap-validation.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='checkpoints'},indent=2),flush=True)


if __name__=='__main__': main()
