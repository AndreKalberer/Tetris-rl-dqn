"""Bounded parallel verification of independent original-game input proofs.

Does not train, resume or change a live game. Every worker reconstructs each
proof in a fresh emulator and preserves exact RAM/RGB checks. Two spare logical
processors on the development workstation are defaults; adjust masks elsewhere.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import ctypes
import json
import multiprocessing
import os
from pathlib import Path
import time


def limit_cpu(mask):
    if mask<=0: raise ValueError('CPU masks must be positive')
    if os.name=='nt':
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetCurrentProcess.restype=ctypes.c_void_p
        kernel.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
        kernel.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
        handle=kernel.GetCurrentProcess()
        if not kernel.SetProcessAffinityMask(handle,mask) or not kernel.SetPriorityClass(handle,0x4000):
            raise ctypes.WinError(ctypes.get_last_error())
    elif hasattr(os,'sched_setaffinity'):
        os.sched_setaffinity(0,{i for i in range(mask.bit_length()) if mask&(1<<i)})
        os.nice(10)


def verify_group(paths,mask,mesen,session_root):
    limit_cpu(mask)
    from nes_original.env import OriginalGame
    rows=[]
    for index,path in paths:
        started=time.perf_counter()
        backend=None
        try:
            record=json.loads(Path(path).read_text(encoding='utf-8-sig'))
            if 'mesence' in record.get('protocol',''):
                if not mesen: raise ValueError('Reference proof requires --mesen')
                from nes_original.mesen import MesenGame
                game=MesenGame(executable=mesen,session_root=session_root,affinity=mask)
            else:
                game=OriginalGame()
            backend=type(game).__name__
            with game:
                game.load_replay(record)
            row=dict(index=index,path=path,verified=True,error=None,backend=backend,
                result=record['result'])
        except Exception as error:
            row=dict(index=index,path=path,verified=False,error=repr(error),backend=backend)
        row.update(wall_seconds=time.perf_counter()-started,pid=os.getpid(),cpu_mask=mask)
        rows.append(row)
    return rows


def verify_many(paths, *, workers=2,masks=(1,256),mesen=None,session_root=None):
    paths=[str(Path(path).resolve()) for path in paths]
    if not paths or not 1<=workers<=4 or len(masks)<workers:
        raise ValueError('Pass proofs and1..4workers, each with an explicit CPU mask')
    selected=list(masks[:workers])
    if any(mask<=0 for mask in selected) or any(a&b for i,a in enumerate(selected) for b in selected[i+1:]):
        raise ValueError('Workers require positive disjoint CPU masks')
    started=time.perf_counter()
    groups=[list(enumerate(paths))[i::workers] for i in range(workers)]
    rows=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs=[pool.submit(verify_group,group,mask,str(Path(mesen).resolve()) if mesen else None,session_root)
              for group,mask in zip(groups,selected) if group]
        for job in jobs: rows.extend(job.result())
    return dict(workers=workers,wall_seconds=time.perf_counter()-started,
        verified=all(row['verified'] for row in rows),rows=sorted(rows,key=lambda row:row['index']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('proofs',type=Path,nargs='+')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--cpu-masks',type=lambda value:int(value,0),nargs='+',default=[1,256])
    p.add_argument('--mesen',type=Path)
    p.add_argument('--benchmark',action='store_true',help='Compare serial and parallel verification of the same proofs')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): p.error('Use a fresh report path')
    if not all(path.is_file() for path in args.proofs): p.error('All proof paths must exist')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    options=dict(masks=args.cpu_masks,mesen=args.mesen,
        session_root=str(args.output.parent/'reference-verification-sessions'))
    result=dict(scope='Verification throughput only; not a playing-strength or live-training speed claim')
    if args.benchmark:
        result['serial']=verify_many(args.proofs,workers=1,**options)
    result['parallel']=verify_many(args.proofs,workers=args.workers,**options)
    result['verified']=result['parallel']['verified'] and result.get('serial',{}).get('verified',True)
    if args.benchmark and result['verified']:
        result['speedup']=result['serial']['wall_seconds']/result['parallel']['wall_seconds']
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)
    if not result['verified']: raise SystemExit(1)


if __name__=='__main__': main()
