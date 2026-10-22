"""Original-game level-255 development attempts with durable input journals.

Only the existing Mesen controller API advances the game. Journals are recovery
evidence, not successful replay proofs; evaluate() separately reconstructs each
finished attempt and verifies its full RAM/RGB/CPU/PPU fingerprint.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mesen', type=Path, required=True)
    parser.add_argument('--jit-deps', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[33])
    parser.add_argument('--target-level', type=int, default=255)
    parser.add_argument('--max-frames', type=int, default=2000000)
    parser.add_argument('--cpu-mask', type=lambda value: int(value, 0), default=32)
    parser.add_argument('--emulator-cpu-mask', type=lambda value: int(value, 0), default=128)
    parser.add_argument('--process-timeout', type=int, default=21600)
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--selector', choices=('retained-prior', 'dellacherie'), default='retained-prior')
    args = parser.parse_args()
    if not 19 <= args.target_level <= 255 or args.max_frames < 1:
        parser.error('Use target level19..255 and a positive gameplay frame budget')
    if not args.seeds or len(args.seeds) != len(set(args.seeds)) or any(not 0 <= s <= 32766 for s in args.seeds):
        parser.error('Pass distinct genuine menu waits0..32766')
    if args.cpu_mask <= 0 or args.emulator_cpu_mask <= 0 or args.cpu_mask & args.emulator_cpu_mask:
        parser.error('Planner and emulator require positive disjoint CPU masks')
    if args.policy and args.selector != 'retained-prior':
        parser.error('Choose learned policy or an experimental selector')
    if args.output.exists():
        parser.error('Use a fresh output directory; attempts are never overwritten or retried')
    sys.path.insert(0, str(args.jit_deps.resolve()))
    from nes_original.verify_batch import limit_cpu
    limit_cpu(args.cpu_mask)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from nes_original.learn import Config, evaluate, load_policy, write_json
    from nes_original.mesen import MesenGame
    model = None
    selector = None
    if args.selector == 'dellacherie':
        from nes_original.dellacherie import Dellacherie
        selector = Dellacherie()
    config = Config(prior='lookahead', tap_period=2, threads=1)
    if args.policy:
        model, _ = load_policy(args.policy)
        config = model.feature_config
        torch.set_num_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True)
    started = time.monotonic()
    source_root = Path(__file__).resolve().parents[1]
    source_hashes = {str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for package in ('nes', 'nes_original') for path in (source_root/package).glob('*')
                     if path.is_file() and path.suffix in ('.py', '.lua')}
    protocol = dict(started_utc=datetime.now(timezone.utc).isoformat(),
        start_level=18, target_level=args.target_level, menu_waits=args.seeds,
        max_frames_per_game=args.max_frames, source_files=source_hashes,
        runtime_latency_charged=True, config=asdict(config), selector=args.selector, planner_cpu_mask=args.cpu_mask,
        emulator_cpu_mask=args.emulator_cpu_mask, process_timeout_seconds=args.process_timeout,
        policy=str(args.policy.resolve()) if args.policy else None,
        policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest() if args.policy else None,
        scope='Development endurance attempt, including known starts; not reserved acceptance or an RL-gain claim',
        controls='Original menus and controller input only; no RAM/RNG/ROM edits or savestate injection',
        journal='Full input prefix and cached RAM/clock every600gameplay frames; not RGB-verified until final replay',
        target='Reach the requested raw level at a settled piece boundary and verify the finished input replay')
    write_json(output/'protocol.json', protocol)

    def status(state, **details):
        write_json(output/'status.json', dict(state=state, pid=os.getpid(),
            time_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic()-started,
            target_level=args.target_level, goal_achieved=False, **details))

    class JournaledGame(MesenGame):
        def __init__(self, role):
            super().__init__(executable=args.mesen, session_root=output/'reference-sessions',
                             affinity=args.emulator_cpu_mask, action_timeout=60,
                             process_timeout=args.process_timeout)
            self.role = role
            self.last_journal_length = 0

        def persist(self, error=None):
            if not self.started:
                return
            stem = output/f'{self.role}-seed-{self.seed}'
            inputs = bytes(self.inputs)
            temporary = stem.with_suffix('.inputs.tmp')
            with temporary.open('wb') as stream:
                stream.write(inputs)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(stem.with_suffix('.inputs'))
            observation_error = None
            try:
                obs = self.observe()
                cached = dict(level=obs.level, lines=obs.lines, score=obs.score, placements=obs.placements,
                              frames=obs.frames, phase=obs.phase, over=obs.over)
            except Exception as failure:
                # Preserve the input prefix even if corrupted counter RAM can
                # no longer be decoded. Do not obscure the original fault.
                cached = None
                observation_error = repr(failure)
            write_json(stem.with_suffix('.journal.json'), dict(input_count=len(inputs),
                inputs_sha256=hashlib.sha256(inputs).hexdigest(), start_frame=self.start_frame,
                seed=self.seed, start_level=self.start_level, rom_sha256=self.rom_sha256,
                identity=self.identity, cached_observation=cached,
                cached_ram_sha256=hashlib.sha256(self._ram.tobytes()).hexdigest(),
                clock={key:self._request[key] for key in
                    ('pc','cpu_cycle','video_frame','ppu_scanline','ppu_cycle')},
                exact_replay_verified=False, stable_boundary_certified=False, error=error,
                observation_error=observation_error,
                fault_ram_hex=self._request['ram_hex'] if error or observation_error else None))
            self.last_journal_length = len(inputs)
            status('running', phase=self.role, menu_wait=self.seed, cached_observation=cached,
                   observation_is_uncertified=True, error=error)

        def reset(self, **kwargs):
            observation = super().reset(**kwargs)
            self.persist()
            return observation

        def step(self, buttons=0):
            try:
                observation = super().step(buttons)
            except Exception as error:
                self.persist(repr(error))
                raise
            if len(self.inputs)-self.last_journal_length >= 600 or observation.over:
                self.persist()
            return observation

        def replay(self):
            self.persist()
            return super().replay()

    created = 0
    def factory():
        nonlocal created
        role = 'gameplay' if created % 2 == 0 else 'verification'
        created += 1
        return JournaledGame(role)

    status('initializing')
    try:
        rows = evaluate(model, output/'cases', seeds=args.seeds, runtime=True, config=config,
                        max_frames=args.max_frames, target_level=args.target_level, game_factory=factory,
                        context_selector=selector)
        attained = any(row['success'] and row['exact_ram_rgb_replay'] for row in rows)
        status('complete', results=rows, target_verified=attained)
        print(json.dumps(dict(target_verified=attained, results=rows)), flush=True)
    except BaseException as error:
        status('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
