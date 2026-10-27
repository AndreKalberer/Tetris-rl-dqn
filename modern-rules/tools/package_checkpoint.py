"""Package a committed source snapshot, trained states and retained evidence."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import zipfile


ROOT=Path(__file__).resolve().parents[1]
PREFIX='Tetris-rl-dqn-checkpoint/'


def sha_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--tag',default='checkpoint/modern-offline-2026-10-21')
    parser.add_argument('--artifacts-dir',type=Path,default=ROOT.parents[1]/'outputs')
    args=parser.parse_args()
    def git(*arguments):
        return subprocess.check_output(['git','-C',str(ROOT),*arguments])
    commit=git('rev-parse','HEAD').decode().strip()
    assert git('rev-parse',args.tag+'^{commit}').decode().strip()==commit
    assert not git('status','--porcelain'), 'Commit source changes before packaging'
    extras={'repository.bundle':args.bundle}
    extras['prior-release/Tetris-RL-Offline-Sprint-v1.zip']=args.artifacts_dir/'Tetris-RL-Offline-Sprint-v1.zip'
    extras['checkpoints/candidates-60k.pt']=ROOT/'runs/candidates-60k/latest.pt'
    extras['models/candidates-60k.pt']=ROOT/'runs/candidates-60k/policy.pt'
    for seed in (1,2):
        folder=ROOT/f'runs/robustness-v1/seed{seed}'
        for steps in (60000,100000):
            extras[f'checkpoints/seed{seed}-{steps}.pt']=folder/f'checkpoint-{steps}.pt'
        for name in ('config.json','training.jsonl','status.json'):
            extras[f'evidence/training/seed{seed}/{name}']=folder/name
    for name in ('config.json','training.jsonl','results.json'):
        path=ROOT/'runs/candidates-60k'/name
        if path.exists(): extras[f'evidence/training/seed0/{name}']=path
    for segment in ('seed92000','seed92000-resumed'):
        for name in ('protocol.json','results.json','status.json','inputs.jsonl',
                     'locks.jsonl','milestones.jsonl','state.pkl'):
            relative=f'runs/offline-endurance-v1/{segment}/{name}'
            extras[relative]=ROOT/relative
    for name in ('offline-sprint-demo.mp4','offline-sprint.png','endurance-final.png'):
        extras['demo/'+name]=args.artifacts_dir/name
    for name,path in extras.items():
        assert path.is_file(),f'Missing artifact: {name}: {path}'
    manifest={}
    metadata=dict(tag=args.tag,commit=commit,branch=git('branch','--show-current').decode().strip(),
        checkpoint_date='2026-10-21',scope='Modern offline checkpoint before NES work',
        endurance=dict(seed=92000,placements=38666,lines=15464,alive_at_stop=True,replay_verified=True),
        note='Full training and game states use pickle; only load trusted local artifacts.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        source=git('archive','--format=tar','HEAD')
        with tarfile.open(fileobj=io.BytesIO(source)) as tar:
            for member in tar:
                if not member.isfile(): continue
                data=tar.extractfile(member).read()
                archive.writestr(PREFIX+member.name,data)
                manifest[member.name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        for name,path in sorted(extras.items()):
            assert name not in manifest,f'Duplicate archive member: {name}'
            archive.write(path,PREFIX+name)
            manifest[name]=dict(bytes=path.stat().st_size,sha256=sha_file(path))
        data=json.dumps(metadata,indent=2).encode('utf-8')
        archive.writestr(PREFIX+'checkpoint.json',data)
        manifest['checkpoint.json']=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        archive.writestr(PREFIX+'manifest.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
        for name,entry in manifest.items():
            with archive.open(PREFIX+name) as stream:
                assert hashlib.file_digest(stream,'sha256').hexdigest()==entry['sha256'],name
    summary=dict(**metadata,archive_bytes=args.output.stat().st_size,archive_sha256=sha_file(args.output),
                 files=len(manifest),integrity='All archive CRCs and SHA-256 file hashes verified',manifest=manifest)
    args.output.with_suffix('.manifest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='manifest'}))


if __name__=='__main__':
    main()
