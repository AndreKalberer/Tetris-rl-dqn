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
    parser.add_argument('--video-start-frame',type=int,default=1)
    parser.add_argument('--mesen',type=Path,help='Portable MesenCE executable for reference-core journals')
    parser.add_argument('--watchdog-frames',type=int,
        help='Mesen-only host stall threshold (minimum60, default600); recovery traces may need3600')
    parser.add_argument('--transport',choices=('file','pipe','pipe-blocking'),
        help='Mesen-only IPC transport (default pipe); does not change recorded inputs or emulated timing')
    args = parser.parse_args()
    if args.video_start_frame<1:
        parser.error('--video-start-frame must be positive')
    if args.watchdog_frames is not None and args.watchdog_frames<60:
        parser.error('--watchdog-frames must be at least60, matching the Mesen host watchdog')
    record = json.loads(args.replay.read_text(encoding='utf-8'))
    factory=OriginalGame
    if 'mesence' in record.get('protocol',''):
        if not args.mesen:
            parser.error('This reference replay needs --mesen with its recorded portable build')
        from nes_original.mesen import MesenGame
        factory=lambda:MesenGame(executable=args.mesen,
            session_root=args.replay.parent/'reference-replay-sessions',
            transport=args.transport if args.transport is not None else 'pipe',
            watchdog_frames=args.watchdog_frames if args.watchdog_frames is not None else 600)
    elif args.watchdog_frames is not None or args.transport is not None:
        parser.error('--watchdog-frames and --transport require a Mesen reference replay and --mesen')
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
        if (writer is not None and frames < args.video_frames and
                len(game.inputs)-game.start_frame >= args.video_start_frame):
            writer.stdin.write(game.screen().tobytes())
            frames += 1
    try:
        with factory() as game:
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
