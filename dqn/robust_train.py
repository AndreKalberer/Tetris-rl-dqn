"""Independent-seed training worker with retained 60k/100k policy snapshots."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

from dqn.train_candidates import Config, Trainer


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2)+'\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--retain', type=int, nargs='+', default=[60000, 100000])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or any(s < 1 or s > args.steps for s in args.retain):
        parser.error('Retained steps must be inside the training budget')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'status.json').exists():
        parser.error('Choose a new output directory to preserve prior evidence')
    config = Config(seed=args.seed)
    write_json(args.output/'config.json', asdict(config))
    trainer = Trainer(config)
    started = time.perf_counter()
    status = dict(pid=os.getpid(), seed=args.seed, target_steps=args.steps, state='running', step=0)
    write_json(args.output/'status.json', status)
    try:
        while trainer.step < args.steps:
            trainer.advance()
            if trainer.step in args.retain:
                trainer.save(args.output/f'checkpoint-{trainer.step}.pt')
                trainer.export_policy(args.output/f'policy-{trainer.step}.pt')
            if trainer.step % 10000 == 0:
                trainer.save(args.output/'latest.pt')
            if trainer.step % 5000 == 0:
                seconds = time.perf_counter()-started
                status.update(step=trainer.step, seconds=seconds,
                    placements_per_second=trainer.step/seconds,
                    completed_episodes=trainer.episodes, replay_mib=trainer.replay.bytes/2**20,
                    last_loss=trainer.last_loss)
                write_json(args.output/'status.json', status)
                with (args.output/'training.jsonl').open('a') as f:
                    f.write(json.dumps(status)+'\n')
                print(json.dumps(status), flush=True)
            if trainer.step % 1000 == 0 and (args.output/'STOP').exists():
                trainer.save(args.output/'interrupted.pt')
                status.update(state='stopped', step=trainer.step)
                write_json(args.output/'status.json', status)
                return
        trainer.save(args.output/'latest.pt')
        status.update(state='complete', step=trainer.step, seconds=time.perf_counter()-started)
        write_json(args.output/'status.json', status)
    except BaseException as error:
        status.update(state='failed', step=trainer.step, error_type=type(error).__name__, error=str(error))
        write_json(args.output/'status.json', status)
        raise
    finally:
        trainer.env.close()


if __name__ == '__main__':
    main()
