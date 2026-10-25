"""Independent original-game evaluation cases on bounded, separate CPUs.

Every case still plays and reconstructs its entire input proof. Logical policy
timing is unchanged; runtime evaluations record actual charged planning delays.
Training and the caller's model/RNG state are never shared with workers.
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

_INITIALIZED=False


def _group(index,seeds,policy,expected_hash,config,output,mask,mesen,jit_deps,runtime,max_frames,target_level):
    global _INITIALIZED
    if jit_deps: sys.path.insert(0,jit_deps)
    from nes_original.verify_batch import limit_cpu
    limit_cpu(mask)
    import torch
    if not _INITIALIZED:
        torch.set_num_interop_threads(1)
        _INITIALIZED=True
    from nes_original.learn import Config,evaluate,load_policy
    from nes_original.env import OriginalGame
    cfg=Config(**config)
    torch.set_num_threads(cfg.threads)
    model=None
    if policy:
        if hashlib.sha256(Path(policy).read_bytes()).hexdigest()!=expected_hash:
            raise ValueError('Selected policy changed after parallel protocol freeze')
        model,_=load_policy(policy)
    folder=Path(output)/f'worker{index}'
    factory=OriginalGame
    if mesen:
        from nes_original.mesen import MesenGame
        factory=lambda:MesenGame(executable=mesen,session_root=folder/'reference-sessions',affinity=mask)
    rows=evaluate(model,folder,seeds=seeds,runtime=runtime,max_frames=max_frames,
        config=cfg,game_factory=factory,target_level=target_level)
    for row in rows:
        row.update(evaluation_worker=index,worker_pid=os.getpid(),worker_cpu_mask=mask,
            proof=f'worker{index}/seed-{row["seed"]}.json')
    return rows


def evaluate_parallel(policy,output, *, seeds,config,workers=2,masks=(1,256),
                      mesen=None,jit_deps=None,runtime=True,max_frames=120000,target_level=30):
    if isinstance(target_level,bool) or not isinstance(target_level,int) or not 19<=target_level<=255:
        raise ValueError('target_level must be an integer from19 through255 for a level18 start')
    if isinstance(max_frames,bool) or not isinstance(max_frames,int) or max_frames<1:
        raise ValueError('max_frames must be a positive integer gameplay-frame budget')
    seeds=[int(seed) for seed in seeds]
    if not seeds or len(set(seeds))!=len(seeds):
        raise ValueError('Pass distinct menu-wait seeds')
    if not 1<=workers<=4 or len(masks)<workers:
        raise ValueError('Pass1..4workers and a CPU mask for each')
    selected=list(masks[:workers])
    if any(mask<=0 for mask in selected) or any(a&b for i,a in enumerate(selected) for b in selected[i+1:]):
        raise ValueError('Workers require positive disjoint CPU masks')
    output=Path(output)
    if output.exists(): raise ValueError('Use a new parallel evaluation directory')
    output.mkdir(parents=True)
    policy=str(Path(policy).resolve()) if policy else None
    expected_hash=hashlib.sha256(Path(policy).read_bytes()).hexdigest() if policy else None
    groups=[seeds[i::workers] for i in range(workers)]
    plan=dict(policy=policy,policy_sha256=expected_hash,config=config,seeds=seeds,
        requested_workers=workers,cpu_masks=selected,seed_groups=groups,
        backend='MesenGame' if mesen else 'OriginalGame/nes-py',runtime_latency_charged=runtime,
        start_level=18,target_level=target_level,max_frames=max_frames,
        proofs='Every game reconstructed in a fresh emulator, no deferred checks')
    (output/'parallel-protocol.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    started=time.perf_counter()
    rows=[]
    try:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            tasks=[pool.submit(_group,i,group,policy,expected_hash,config,str(output.resolve()),mask,
                str(Path(mesen).resolve()) if mesen else None,
                str(Path(jit_deps).resolve()) if jit_deps else None,runtime,max_frames,target_level)
                for i,(group,mask) in enumerate(zip(groups,selected)) if group]
            for task in tasks: rows.extend(task.result())
    except Exception as error:
        (output/'parallel-failure.json').write_text(json.dumps(dict(error=repr(error),
            completed_rows=rows,wall_seconds=time.perf_counter()-started),indent=2)+'\n',encoding='utf-8')
        raise
    by_seed={row['seed']:row for row in rows}
    result=[by_seed[seed] for seed in seeds]
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (output/'parallel-summary.json').write_text(json.dumps(dict(wall_seconds=time.perf_counter()-started,
        games=len(result),workers=workers,all_replays_verified=all(row['exact_ram_rgb_replay'] for row in result)),indent=2)+'\n',encoding='utf-8')
    return result
