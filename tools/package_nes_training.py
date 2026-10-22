"""Archive the committed NES learner, policies, full resume states and evidence."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import zipfile


ROOT=Path(__file__).resolve().parents[1]


def file_hash(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--artifacts-dir',type=Path,default=ROOT.parents[1]/'outputs')
    args=parser.parse_args()
    def git(*arguments): return subprocess.check_output(['git','-C',str(ROOT),*arguments])
    assert not git('status','--porcelain'),'Commit the source snapshot first'
    result=json.loads((ROOT/'runs/nes-trained-comparison-v2/results.json').read_text(encoding='utf-8'))
    selected=ROOT/result['selected_path']
    step=result['selected_metadata']['training_decisions']
    selected_checkpoint=selected.parent/(f'checkpoint-{step}.pt' if step else 'checkpoint-warm.pt')
    extras={'checkpoints/nes-seed0-selected.pt':selected_checkpoint,
            'checkpoints/nes-seed0-50000.pt':ROOT/'runs/nes-training-v2/seed0/latest.pt'}
    for folder in ('nes-training-v1/seed0','nes-training-v2/seed0',
                   'nes-trained-comparison-v1','nes-trained-comparison-v2'):
        for path in (ROOT/'runs'/folder).rglob('*'):
            if path.is_file() and (path.suffix in ('.json','.jsonl','.py') or
                                   path.suffix=='.pt' and path.name.startswith('policy-')):
                extras['runs/'+str(path.relative_to(ROOT/'runs')).replace('\\','/')]=path
    for name in ('nes-trained-demo.mp4','nes-trained.png'):
        extras['demo/'+name]=args.artifacts_dir/name
    manifest={}
    prefix='Tetris-NES-Training/'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        with tarfile.open(fileobj=io.BytesIO(git('archive','--format=tar','HEAD'))) as source:
            for member in source:
                if not member.isfile(): continue
                data=source.extractfile(member).read()
                archive.writestr(prefix+member.name,data)
                manifest[member.name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        for name,path in sorted(extras.items()):
            assert path.is_file() and name not in manifest,name
            archive.write(path,prefix+name)
            manifest[name]=dict(bytes=path.stat().st_size,sha256=file_hash(path))
        metadata=dict(git_commit=git('rev-parse','HEAD').decode().strip(),total_rl_decisions=50000,
            selected_rl_decisions=step,selected_policy=str(selected.relative_to(ROOT)),manifest=manifest,
            scope='Single-seed pilot; RL did not beat heuristic/imitation reliably; provisional NES simulator')
        archive.writestr(prefix+'training-manifest.json',json.dumps(metadata,indent=2))
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
        for name,entry in manifest.items():
            with archive.open(prefix+name) as stream:
                assert hashlib.file_digest(stream,'sha256').hexdigest()==entry['sha256'],name
    receipt=dict(path=str(args.output),bytes=args.output.stat().st_size,sha256=file_hash(args.output),
                 files=len(manifest),git_commit=metadata['git_commit'],integrity='CRC and all file SHA-256 hashes verified')
    args.output.with_suffix('.manifest.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__': main()
