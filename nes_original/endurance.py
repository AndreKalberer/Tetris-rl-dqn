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
    parser.add_argument('--endless', action='store_true',
                        help='no target: play until the game ends (top-out or crash) or --max-frames runs out')
    parser.add_argument('--max-frames', type=int, default=2000000)
    parser.add_argument('--cpu-mask', type=lambda value: int(value, 0), default=32)
    parser.add_argument('--emulator-cpu-mask', type=lambda value: int(value, 0), default=128)
    parser.add_argument('--process-timeout', type=int, default=21600)
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--no-softdrop-at-1g',action='store_true',
                        help='Experimental controller timing: omit Down during ordinary1G fall')
    parser.add_argument('--no-softdrop-levels',type=int,nargs='*',default=[],
                        help='Experimental neutral post-target input only at listed ordinary1G levels')
    parser.add_argument('--nonactive-input',choices=('neutral','down','left','right'),default='neutral',
                        help='Legal input during entry/clear/render phases; active controls stay separate')
    parser.add_argument('--stall-rescue-input',choices=('neutral','down','left','right','a','b'),default='neutral',
                        help='Optional legal input on unchanged nonactive public observations')
    parser.add_argument('--stall-rescue-after',type=int,default=120)
    parser.add_argument('--stall-rescue-fallback',choices=('neutral','down','left','right','a','b'),default='neutral')
    parser.add_argument('--stall-rescue-primary-frames',type=int,default=8)
    parser.add_argument('--transport',choices=('pipe','pipe-blocking','file'),default='pipe')
    parser.add_argument('--verification-transport',choices=('pipe','pipe-blocking','file'),
                        help='Optional separate transport for the fresh verification process')
    parser.add_argument('--stop-on-success',action='store_true',
                        help='End the development suite after the first fully verified target')
    parser.add_argument('--selector', choices=('retained-prior', 'dellacherie', 'risk-aware',
                        'dellacherie-alerts', 'dellacherie-observed-167', 'tuned', 'tuned-alerts', 'tuned-crash-map'),
                        default='retained-prior')
    parser.add_argument('--weights', default='score_start',
                        help='tuned selectors: weights.json from nes_original.tune, or score_start / dellacherie')
    parser.add_argument('--beam', type=int, default=3,
                        help='tuned selectors: proposals expanded with the visible next piece (0 = one-ply)')
    parser.add_argument('--zone-weights',
                        help='tuned selectors: weights.json used only within --zone-levels (e.g. no-singles play)')
    parser.add_argument('--zone-levels', type=int, nargs=2, default=(156, 158))
    parser.add_argument('--trace-levels', type=int, nargs=2,
                        help='diagnostics: log every decision input between these levels to decision-trace.jsonl')
    parser.add_argument('--extra-zone', nargs='+', action='append', default=[],
                        help='tuned selectors: WEIGHTS FIRST LAST [CHILD_BAN_MASK [EXPECT]] - more level zones')
    parser.add_argument('--zone-ban', type=int, default=2,
                        help='clear sizes banned in zone lookahead children (bitmask; 2 = singles)')
    parser.add_argument('--zone-expect', type=int, default=2,
                        help='zone lookahead: also average over the unknown piece after next')
    parser.add_argument('--expect', type=int, default=0,
                        help='tuned selectors: also average over the unknown piece after next')
    parser.add_argument('--reaction-frames', type=int, default=3,
                        help='neutral frames after spawn before the first input (planning time past '
                             'max(1, this) frames is charged as idle frames)')
    parser.add_argument('--alt-weights',
                        help='tuned selectors: low-stack tetris weights.json used outside the zones')
    parser.add_argument('--alt-levels', type=int, nargs='+',
                        help='raw level ranges for --alt-weights: lo hi [lo hi ...] (default: the file\'s)')
    parser.add_argument('--stall-grace-frames', type=int,
                        help='Opt-in: extra frames for a stall that looks like a recoverable ROM loop')
    args = parser.parse_args()
    if args.endless:
        args.target_level = 10**9
    if not 0 <= args.reaction_frames <= 10:
        parser.error('Reaction frames must be 0..10')
    if args.alt_levels and len(args.alt_levels)%2:
        parser.error('--alt-levels takes level pairs')
    if not 19 <= args.target_level <= 10**9 or args.max_frames < 1:
        parser.error('Use target level19+ (256+ = absolute levels across rebirths) and a positive frame budget')
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
    nonactive_input={'neutral':0,'down':32,'left':64,'right':128}[args.nonactive_input]
    buttons={'neutral':0,'down':32,'left':64,'right':128,'a':1,'b':2}
    stall_rescue_input=buttons[args.stall_rescue_input]
    stall_rescue_fallback=buttons[args.stall_rescue_fallback]
    model = None
    selector = None
    if args.selector == 'dellacherie':
        from nes_original.dellacherie import Dellacherie
        selector = Dellacherie()
    elif args.selector == 'risk-aware':
        from nes_original.robust import RiskAwareNextPiece
        selector = RiskAwareNextPiece()
    elif args.selector == 'dellacherie-alerts':
        from nes_original.crash_alerts import DocumentedHazardPrioritizer
        selector = DocumentedHazardPrioritizer()
    elif args.selector == 'dellacherie-observed-167':
        from nes_original.crash_alerts import Observed167TriplePrioritizer
        selector = Observed167TriplePrioritizer()
    elif args.selector in ('tuned', 'tuned-alerts', 'tuned-crash-map'):
        from nes_original.sim import tuned_selector
        selector = tuned_selector(args.weights, beam=args.beam, expect=args.expect,
                                  zone_weights=args.zone_weights, zone_levels=tuple(args.zone_levels),
                                  zone_ban=args.zone_ban, zone_expect=args.zone_expect,
                                  extra_zones=[tuple(zone) for zone in args.extra_zone],
                                  reaction=args.reaction_frames, alt_weights=args.alt_weights,
                                  alt_levels=[(args.alt_levels[i], args.alt_levels[i+1])
                                              for i in range(0, len(args.alt_levels), 2)] if args.alt_levels else None)
        if args.selector == 'tuned-alerts':
            from nes_original.crash_alerts import Observed167TriplePrioritizer
            selector = Observed167TriplePrioritizer(base_selector=selector)
        elif args.selector == 'tuned-crash-map':
            from nes_original.crash_model import CrashMapPrioritizer
            selector = CrashMapPrioritizer(base_selector=selector)
    config = Config(prior='lookahead', tap_period=2, threads=1, reaction_frames=args.reaction_frames)
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
        runtime_latency_charged=True, config=asdict(config), selector=args.selector,
        no_softdrop_at_1g=args.no_softdrop_at_1g,
        no_softdrop_levels=args.no_softdrop_levels,transport=args.transport,
        nonactive_input=args.nonactive_input,nonactive_input_byte=nonactive_input,
        stall_rescue_input=args.stall_rescue_input,stall_rescue_after=args.stall_rescue_after,
        stall_rescue_fallback=args.stall_rescue_fallback,
        stall_rescue_primary_frames=args.stall_rescue_primary_frames,
        verification_transport=args.verification_transport or args.transport,
        stop_on_success=args.stop_on_success,stall_grace_frames=args.stall_grace_frames,
        weights=args.weights if args.selector.startswith('tuned') else None,
        beam=args.beam if args.selector.startswith('tuned') else None,
        expect=args.expect if args.selector.startswith('tuned') else None,
        zone_weights=args.zone_weights if args.selector.startswith('tuned') else None,
        zone_levels=list(args.zone_levels) if args.zone_weights else None,
        zone_ban=args.zone_ban if args.zone_weights else None,
        zone_expect=args.zone_expect if args.zone_weights else None,
        extra_zones=args.extra_zone or None,
        reaction_frames=args.reaction_frames,
        alt_weights=args.alt_weights if args.selector.startswith('tuned') else None,
        alt_levels=args.alt_levels,
        selector_parameters=selector.stats() if selector and hasattr(selector, 'stats') else None,
        planner_cpu_mask=args.cpu_mask,
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
                             process_timeout=args.process_timeout,
                             stall_grace_frames=args.stall_grace_frames,
                             transport=(args.verification_transport or args.transport)
                                if role=='verification' else args.transport)
            self.role = role
            self.last_journal_length = 0
            self.journal_fault = None

        def _step_raw(self, buttons):
            before = self._ram.copy() if self.started and self._ram is not None else None
            try:
                return super()._step_raw(buttons)
            finally:
                if before is not None and self._ram is not None:
                    previous_phase, phase = int(before[0x68]), int(self._ram[0x68])
                    previous_working, working = int(before[0x48]), int(self._ram[0x48])
                    active_exit = (previous_phase == 1 and phase != 1) or (previous_working == 1 and working != 1)
                    score_entry = (previous_phase != 5 and phase == 5) or (previous_working != 5 and working == 5)
                    if active_exit or score_entry:
                        # Read-only diagnostics, never supplied to the selector.
                        # These may be mid-copy samples; settled counters are
                        # certified separately by the evaluator's progress audit.
                        event = dict(input_count=len(self.inputs),input=int(buttons),
                            event='active_exit' if active_exit else 'score_entry',
                            settled_boundary_certified=False,
                            before_player_ram_hex=before[0x60:0x80].tobytes().hex(),
                            after_player_ram_hex=self._ram[0x60:0x80].tobytes().hex(),
                            before_working_player_ram_hex=before[0x40:0x60].tobytes().hex(),
                            after_working_player_ram_hex=self._ram[0x40:0x60].tobytes().hex(),
                            pre_level=int(before[0x64]),post_level=int(self._ram[0x64]),
                            orientation=int(self._ram[0x62]),y=int(self._ram[0x61]),
                            raw_pushdown=int(self._ram[0x6f]),phase=phase,
                            working_pushdown=int(self._ram[0x4f]),working_phase=working,
                            clock={key:self._request[key] for key in
                                ('pc','cpu_cycle','video_frame','ppu_scanline','ppu_cycle')})
                        with (output/f'{self.role}-seed-{self.seed}-events.jsonl').open('a',encoding='utf-8') as stream:
                            stream.write(json.dumps(event)+'\n')

        def persist(self, error=None):
            if not self.started:
                return
            if error is not None:
                self.journal_fault = error
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
                exact_replay_verified=False, stable_boundary_certified=False, error=self.journal_fault,
                observation_error=observation_error,
                fault_ram_hex=self._request['ram_hex'] if self.journal_fault or observation_error else None))
            self.last_journal_length = len(inputs)
            status('running', phase=self.role, menu_wait=self.seed, cached_observation=cached,
                   observation_is_uncertified=True, error=self.journal_fault)

        def reset(self, **kwargs):
            self.journal_fault = None
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
    active_game = None
    def factory():
        nonlocal created, active_game
        role = 'gameplay' if created % 2 == 0 else 'verification'
        created += 1
        game = JournaledGame(role)
        if role == 'gameplay':
            active_game = game
        return game

    def log_alert_selection(observation, plan):
        # Diagnostic output only; the selector still receives exactly the
        # public observation/proposal pair and no emulator/internal RNG state.
        selected = selector(observation, plan)
        log_alert_selection.wants_down = bool(getattr(selector, 'wants_down', False))
        decision = selector.last_decision
        if args.trace_levels and args.trace_levels[0] <= observation.level <= args.trace_levels[1]:
            # Full decision input for offline replay (diagnostics only; never fed back).
            actions = plan.actions
            with (output/'decision-trace.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(menu_wait=active_game.seed,placements=observation.placements,
                    level=observation.level,lines=observation.lines,score=observation.score,
                    next_orientation=int(getattr(observation,'next_orientation',-1)),
                    board=(observation.board!=0).astype(int).tolist(),
                    control=dict(x=int(observation.x),y=int(observation.y),orientation=int(observation.orientation),
                                 das=int(observation.das)),
                    candidates=[dict(board=(b!=0).astype(int).tolist(),lines=int(l),terminal=bool(t),
                        x=int(c.x),y=int(c.y),orientation=int(c.orientation),das=int(c.das),
                        soft_points=int(c.soft_points)) for b,l,t,c in
                        zip(actions.boards,actions.lines,actions.terminal,actions.controls)],
                    selected=int(selected),counts=list(getattr(selector,'counts',[])),
                    counts_known=bool(getattr(selector,'counts_known',False)),
                    wants_down=bool(getattr(selector,'wants_down',False))))+'\n')
        if (any(row['alert'] for row in decision['candidates']) or
                decision['selection_tier'] != 'unalerted_nonterminal'):
            with (output/'alert-decisions.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(menu_wait=active_game.seed,
                    placements=observation.placements,level=observation.level,lines=observation.lines,
                    score=observation.score,decision=decision))+'\n')
        return selected

    status('initializing')
    try:
        rows = evaluate(model, output/'cases', seeds=args.seeds, runtime=True, config=config,
                        max_frames=args.max_frames, target_level=args.target_level, game_factory=factory,
                        context_selector=log_alert_selection if args.selector in
                            ('dellacherie-alerts', 'dellacherie-observed-167', 'tuned-alerts', 'tuned-crash-map') else selector,
                        no_softdrop_at_1g=args.no_softdrop_at_1g,
                        no_softdrop_levels=args.no_softdrop_levels,
                        nonactive_input=nonactive_input,
                        stall_rescue_input=stall_rescue_input,stall_rescue_after=args.stall_rescue_after,
                        stall_rescue_fallback_input=stall_rescue_fallback,
                        stall_rescue_primary_frames=args.stall_rescue_primary_frames,
                        stop_on_success=args.stop_on_success)
        attained = any(row['success'] and row['exact_ram_rgb_replay'] for row in rows)
        status('complete', results=rows, target_verified=attained,
               selector_statistics=selector.stats() if selector and hasattr(selector, 'stats') else None)
        print(json.dumps(dict(target_verified=attained, results=rows)), flush=True)
    except BaseException as error:
        status('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
