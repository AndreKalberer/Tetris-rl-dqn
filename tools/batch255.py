"""Run a fixed list of level-255 attempts, a few at a time, with one frozen launcher.

Each attempt is `sh LAUNCHER seed planner_mask emulator_mask` (an endurance.py run that
plays, then fresh-replay verifies). Slots are disjoint CPU-mask pairs; a new seed starts
whenever a slot frees up. Progress is written to BATCH_DIR/batch-status.json.

  python tools/batch255.py --launcher runs/nes-255-v7/launch.sh --dir runs/nes-255-v7-test25 --seeds 1001 1777 ...
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

BASH = r'C:\Program Files\Git\bin\bash.exe'
SLOTS = [(4, 8), (16, 32), (64, 128), (256, 512), (1024, 2048), (4096, 8192), (16384, 32768), (1, 2)]


def case(run_dir, seed):
    path = Path(run_dir)/f'seed{seed}'/'cases'/f'seed-{seed}-case.json'
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    return {k: data.get(k) for k in ('level', 'lines', 'termination_kind', 'case_state', 'exact_ram_rgb_replay')}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--launcher', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True, help='folder the launcher writes seed<N> into')
    p.add_argument('--dir', type=Path, required=True, help='batch status folder')
    p.add_argument('--seeds', type=int, nargs='+', required=True)
    p.add_argument('--slots', type=int, default=6)
    args = p.parse_args()
    args.dir.mkdir(parents=True, exist_ok=True)
    pending = list(args.seeds)
    running = {}   # slot index -> (seed, Popen)
    done = {}
    started = datetime.now(timezone.utc).isoformat()

    def write():
        (args.dir/'batch-status.json').write_text(json.dumps(dict(
            started_utc=started, updated_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
            launcher=str(args.launcher), seeds=args.seeds, pending=pending,
            running={str(s): seed for s, (seed, _) in running.items()},
            finished={str(seed): result for seed, result in done.items()},
            reached_255=sum(1 for r in done.values() if r and r.get('level') == 255),
            verified_255=sum(1 for r in done.values() if r and r.get('level') == 255 and r.get('case_state') == 'verified')),
            indent=1), encoding='utf-8')

    while pending or running:
        for slot in range(args.slots):
            if slot not in running and pending:
                seed = pending.pop(0)
                planner, emulator = SLOTS[slot]
                proc = subprocess.Popen([BASH, str(args.launcher), str(seed), str(planner), str(emulator)],
                                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                running[slot] = (seed, proc)
        for slot, (seed, proc) in list(running.items()):
            if proc.poll() is not None:
                done[seed] = case(args.run_dir, seed)
                del running[slot]
        write()
        time.sleep(20)
    write()


if __name__ == '__main__':
    main()
