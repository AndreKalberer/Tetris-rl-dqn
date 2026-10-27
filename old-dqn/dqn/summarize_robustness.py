"""Summarize the predeclared study, keeping matched budgets and censoring clear."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(result):
    summaries = []
    for model in result['models']:
        seed, steps = model['model'].split('-')
        by_beam = {}
        for beam in result['protocol']['beams']:
            rows = [r for r in model['episodes'] if r['beam'] == beam]
            complete = [r for r in rows if r['completed']]
            by_beam[str(beam)] = dict(
                mean_lines_before_death_or_cap=statistics.mean(r['lines'] for r in rows),
                median_lines_before_death_or_cap=statistics.median(r['lines'] for r in rows),
                min_lines=min(r['lines'] for r in rows), max_lines=max(r['lines'] for r in rows),
                naturally_completed=len(complete), still_alive_at_cap=sum(r['capped'] for r in rows),
                conditional_mean_completed_lines=statistics.mean(r['lines'] for r in complete) if complete else None,
                aggregate_placements_per_second=sum(r['placements'] for r in rows)/sum(r['seconds'] for r in rows),
                lines_by_evaluation_seed={str(r['seed']):r['lines'] for r in rows})
        paired = []
        for evaluation_seed in result['protocol']['evaluation_seeds']:
            before = next(r for r in model['episodes'] if r['seed']==evaluation_seed and r['beam']==1)
            after = next(r for r in model['episodes'] if r['seed']==evaluation_seed and r['beam']==4)
            paired.append(dict(seed=evaluation_seed, bounded_line_difference=after['lines']-before['lines']))
        summaries.append(dict(model=model['model'], training_seed=int(seed.removeprefix('seed')),
                              training_steps=int(steps), by_beam=by_beam, paired=paired,
                              policy_sha256=model['policy_sha256']))
    matched = [m for m in summaries if m['training_steps']==60000]
    extended = [m for m in summaries if m['training_steps']==100000]
    assert len(matched)==3 and len(extended)==2
    return dict(scope='Three training seeds; five paired evaluation streams; five retained snapshots',
        placement_cap=result['protocol']['placement_cap'], evaluation_seeds=result['protocol']['evaluation_seeds'],
        models=summaries,
        matched_60k=dict(independent_training_seeds=[0,1,2],
            bounded_mean_by_beam={str(b):statistics.mean(m['by_beam'][str(b)]['mean_lines_before_death_or_cap'] for m in matched) for b in [1,4]},
            survivors_by_beam={str(b):sum(m['by_beam'][str(b)]['still_alive_at_cap'] for m in matched) for b in [1,4]},
            model_stream_pairs=15),
        limits=['Capped games are censored; bounded means are not full-game survival means.',
                'The same five piece streams are reused across agents, so 15 pairs are not 15 independent streams.',
                '60k and 100k snapshots of the same seed are correlated; only three independent training runs exist.',
                'No hyperparameter tuning used this evaluation set; no comparison to the unavailable historical checkpoint.',
                'Concurrent CPU jobs affect timing; runtime numbers describe these runs, not a controlled hardware benchmark.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('runs/robustness-v1/results.json'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=summarize(json.loads(args.input.read_text()))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for model in result['models']:
        print(json.dumps(dict(model=model['model'],by_beam=model['by_beam'])))


if __name__=='__main__':
    main()
