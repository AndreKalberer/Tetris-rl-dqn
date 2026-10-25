"""Validate and merge all predeclared cases after parallel evaluation finishes."""
import hashlib
import json
from pathlib import Path

import torch

from dqn.robust_train import write_json
from dqn.summarize_robustness import summarize


def main():
    root=Path('runs/robustness-v1')
    protocol=json.loads((root/'protocol.json').read_text())
    scheduling=json.loads((root/'scheduling.json').read_text())
    expected={(seed,beam) for seed in protocol['evaluation_seeds'] for beam in protocol['beams']}
    source_hashes={name.replace('\\','/'):digest for name,digest in protocol['source_sha256'].items()}
    for source in ['tetris_macro_env.py','dqn/candidates.py','dqn/lookahead.py','dqn/train_candidates.py','dqn/robust_train.py']:
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==source_hashes[source], source
    models=[]
    for name in protocol['models']:
        origin=scheduling['selected_origins'][name]
        result=json.loads((root/origin/name/'results.json').read_text())
        rows=result['episodes']
        assert len(rows)==len(expected) and {(r['seed'],r['beam']) for r in rows}==expected
        assert result['placement_cap']==protocol['placement_cap']
        training_seed,steps=map(int,name.removeprefix('seed').split('-'))
        path=Path(result['checkpoint'])
        state=torch.load(path,map_location='cpu',weights_only=True)
        assert state['training_steps']==steps and state['config']['seed']==training_seed
        assert result['config']==state['config']
        for row in rows:
            assert 1<=row['placements']<=protocol['placement_cap']
            assert row['completed'] != row['capped']
            assert not row['capped'] or row['placements']==protocol['placement_cap']
            assert 0<=row['lines']<=row['placements']*4//10
        result.update(model=name,policy_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),origin=origin)
        models.append(result)
    result=dict(protocol=protocol,scheduling=scheduling,models=models,
                training=[json.loads((root/f'seed{s}/status.json').read_text()) for s in [1,2]])
    assert all(t['state']=='complete' and t['step']==100000 for t in result['training'])
    write_json(root/'results.json',result)
    summary=summarize(result)
    write_json(root/'summary.json',summary)
    outputs=Path('../../outputs')
    write_json(outputs/'robustness-results.json',dict(summary=summary,raw=result))
    write_json(root/'evaluation-status.json',dict(state='complete',completed_models=5,planned_models=5,
        evaluated_games=sum(len(m['episodes']) for m in models),scheduling='parallel after first two complete models'))
    for model in summary['models']:
        print(json.dumps(dict(model=model['model'],
            means={b:s['mean_lines_before_death_or_cap'] for b,s in model['by_beam'].items()},
            survivors={b:s['still_alive_at_cap'] for b,s in model['by_beam'].items()})),flush=True)


if __name__=='__main__':
    main()
