"""Create a small standalone NES engine development snapshot, not a trained bot."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


ROOT=Path(__file__).resolve().parents[1]  # nes-bot/
REPO=ROOT.parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--artifacts-dir',type=Path,default=REPO.parents[1]/'outputs')
    args=parser.parse_args()
    files={str(path.relative_to(ROOT)).replace('\\','/'):path for path in (ROOT/'nes').glob('*') if path.is_file()}
    files['README.md']=ROOT/'nes/README.md'
    files['RULES.md']=ROOT/'nes/RULES.md'
    for name in ('tests/test_nes.py','pytest.ini','requirements-nes.txt','requirements-nes-dev.txt',
                 'requirements-nes-validation.txt','setup-nes.ps1','play-nes.ps1','LICENSE',
                 'tools/verify_nes_reference.py','tools/package_nes.py','progress.md'):
        files[name]=(REPO if name in ('pytest.ini','LICENSE','progress.md') else ROOT)/name
    for path in (ROOT/'evidence/nes-v1').glob('*.json'):
        files['evidence/'+path.name]=path
    for path in (REPO/'runs/nes-dev-v2').rglob('*.json'):
        files['evidence/probes/'+str(path.relative_to(REPO/'runs/nes-dev-v2')).replace('\\','/')]=path
    for name in ('nes-engine-demo.mp4','nes-engine.png'):
        files['demo/'+name]=args.artifacts_dir/name
    manifest={name:dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
              for name,path in files.items()}
    metadata=dict(scope='Provisional NES engine development snapshot; no trained NES policy or ROM parity claim',
        git_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD']).decode().strip(),
        modern_checkpoint='checkpoint/modern-offline-2026-10-21',manifest=manifest)
    prefix='NES-Tetris-Engine/'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,path in sorted(files.items()): archive.write(path,prefix+name)
        archive.writestr(prefix+'manifest.json',json.dumps(metadata,indent=2))
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
        for name,entry in manifest.items():
            assert hashlib.sha256(archive.read(prefix+name)).hexdigest()==entry['sha256']
    receipt=dict(path=str(args.output),bytes=args.output.stat().st_size,
        sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),files=len(manifest),git_commit=metadata['git_commit'],
        integrity='Archive CRCs and all content-file SHA-256 hashes verified')
    args.output.with_suffix('.manifest.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
