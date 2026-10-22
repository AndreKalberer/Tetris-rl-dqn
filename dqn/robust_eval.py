"""Run the predeclared multi-seed robustness protocol, retaining all outcomes."""
import hashlib
import json
import os
from pathlib import Path
import time

from dqn.benchmark_lookahead import benchmark
from dqn.robust_train import write_json


def main():
    root = Path('runs/robustness-v1')
    protocol = json.loads((root/'protocol.json').read_text())
    models = [('seed0-60000', Path('runs/candidates-60k/policy.pt'), None),
              ('seed1-60000', root/'seed1/policy-60000.pt', root/'seed1/status.json'),
              ('seed1-100000', root/'seed1/policy-100000.pt', root/'seed1/status.json'),
              ('seed2-60000', root/'seed2/policy-60000.pt', root/'seed2/status.json'),
              ('seed2-100000', root/'seed2/policy-100000.pt', root/'seed2/status.json')]
    assert [m[0] for m in models] == protocol['models']
    status_path = root/'evaluation-status.json'
    if status_path.exists():
        raise RuntimeError('Preserve existing evaluation evidence; use a new protocol directory for a new run')
    status = dict(pid=os.getpid(), state='starting', completed_models=0, planned_models=len(models))
    write_json(status_path, status)
    records = []
    try:
        for name, path, worker_status in models:
            status.update(state='waiting_for_policy', model=name)
            write_json(status_path, status)
            while not path.exists():
                if (root/'STOP').exists():
                    raise KeyboardInterrupt('Stop requested')
                if worker_status and worker_status.exists():
                    worker = json.loads(worker_status.read_text())
                    if worker['state'] in {'failed', 'stopped'}:
                        raise RuntimeError(f'Training {name} ended before producing its snapshot')
                time.sleep(1)
            # Workers publish their retained policy before updating the matching status.
            if worker_status:
                required_step = int(name.rsplit('-',1)[1])
                while json.loads(worker_status.read_text()).get('step',0) < required_step:
                    time.sleep(1)
            status.update(state='running', model=name)
            write_json(status_path, status)
            def progress(row):
                status['current_game'] = row
                write_json(status_path, status)
            result = benchmark(path, protocol['evaluation_seeds'], protocol['placement_cap'],
                protocol['beams'], root/'evaluation'/name, on_progress=progress, stop_file=root/'STOP')
            result['model'] = name
            result['policy_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            records.append(result)
            status['completed_models'] = len(records)
            write_json(root/'partial-results.json', dict(protocol=protocol, models=records))
        write_json(root/'results.json', dict(protocol=protocol, models=records))
        status.update(state='complete', completed_models=len(records))
        write_json(status_path, status)
    except KeyboardInterrupt:
        status['state'] = 'stopped'
        write_json(status_path, status)
    except BaseException as error:
        status.update(state='failed', error_type=type(error).__name__, error=str(error))
        write_json(status_path, status)
        raise


if __name__ == '__main__':
    main()
