"""Verify original-game input saves and optionally export their screen video."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from nes.engine import FPS
from nes_original.env import OriginalGame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('replay',type=Path)
    parser.add_argument('--video',type=Path)
    parser.add_argument('--video-frames',type=int,default=1800)
    args = parser.parse_args()
    record = json.loads(args.replay.read_text(encoding='utf-8'))
    writer = None
    frames = 0
    if args.video:
        if args.video_frames <= 0:
            parser.error('--video-frames must be positive')
        executable = shutil.which('ffmpeg')
        if not executable:
            parser.error('ffmpeg must be on PATH for video export')
        args.video.parent.mkdir(parents=True,exist_ok=True)
        writer = subprocess.Popen([executable,'-hide_banner','-loglevel','error','-n',
            '-f','rawvideo','-pix_fmt','rgb24','-s','256x240','-r',str(FPS),'-i','-',
            '-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(args.video)],stdin=subprocess.PIPE)

    def capture(game):
        nonlocal frames
        if writer is not None and frames < args.video_frames:
            writer.stdin.write(game.screen().tobytes())
            frames += 1
    try:
        with OriginalGame() as game:
            game.load_replay(record,callback=capture)
            result = dict(record['result'],exact_ram_rgb_replay=True,video_frames=frames)
    finally:
        if writer is not None:
            writer.stdin.close()
            if writer.wait() != 0:
                raise RuntimeError('ffmpeg video export failed')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
