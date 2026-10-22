"""Bounded visible-next-piece lookahead, evaluated in the original game.

Future control/boards are approximate planner features. Only the actual emulator
executes the selected target. No native snapshots, RNG or unknown preview is
used to search. A fixed beam preserves deterministic logical decisions.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from nes.actions import ActionRules,enumerate_actions
from nes.engine import Control,Observation,Phase,SPAWN,TYPE,control_frame
from nes_original.env import OriginalGame
from nes_original.feedback import FeedbackController
from nes_original.progression import after_clear


class NextPiece:
    def __init__(self, beam=3,tap_period=0,jit=False,line_power=1):
        if beam < 1:
            raise ValueError('Beam must be positive')
        self.beam = beam
        self.rules = ActionRules(horizontal_period=tap_period)
        if line_power not in (1,2):
            raise ValueError('Line reward exponent must be1 or2')
        self.line_power = line_power
        self.fast = None
        if jit:
            from nes_original import fast
            fast.warmup(line_power)  # Compile every argument type before gameplay.
            self.fast = fast
        self.calls = self.expansions = 0
        self.times = []

    def __call__(self, obs, plan):
        start = time.perf_counter()
        actions = plan.actions
        shortlist,seen = [],set()
        for index in np.argsort(-actions.heuristic,kind='stable'):
            if actions.terminal[index]:
                continue
            board = actions.boards[index] != 0
            key = board.tobytes()
            if key not in seen:
                seen.add(key)
                shortlist.append(int(index))
            if len(shortlist) >= self.beam:
                break
        if not shortlist:
            self.last_scores = actions.heuristic.astype(np.float32).copy()
            return int(actions.heuristic.argmax())
        next_pose = SPAWN[TYPE[obs.next_orientation]]
        scored = []
        for index in shortlist:
            board = actions.boards[index] != 0
            level,lines = after_clear(obs.level,obs.lines,int(actions.lines[index]))
            state = Control(orientation=next_pose,das=actions.controls[index].das,soft=0)
            failed = False
            # Conservative projection of native spawn settling, with no input.
            for _ in range(2):
                state,failed = control_frame(board,state,0,level)
                if failed:
                    break
            if failed:
                value = -1000.0
            else:
                # Unknown next-next preview is fixed dummy context and its
                # features are not read: only child geometric heuristic values.
                future = Observation(board,state,0,Phase.ACTIVE,level,lines,
                    obs.score,obs.placements+1,obs.frames+len(plan.paths[index]),obs.start_level)
                if self.fast:
                    value = self.fast.best_future(board,tuple(vars(state).values()),level,
                        self.rules.horizontal_period,self.rules.reaction_frames,self.rules.navigation_frames,
                        self.rules.rotation_period,self.line_power)
                else:
                    children = enumerate_actions(future,self.rules)
                    h = children.heuristic + .76*(children.lines**self.line_power-children.lines)
                    value = float(h.max())
                self.expansions += 1
            score = .76*int(actions.lines[index])**self.line_power + value
            scored.append(score)
        self.calls += 1
        self.times.append(time.perf_counter()-start)
        cache = {(actions.boards[i]!=0).tobytes():float(value) for i,value in zip(shortlist,scored)}
        self.last_scores = np.asarray([cache.get((board!=0).tobytes(),float(actions.heuristic[i]))
            for i,board in enumerate(actions.boards)],dtype=np.float32)
        return shortlist[int(np.argmax(scored))]

    def action_values(self,obs,plan):
        """Per-target prior for the learner; use a full beam for all actions."""
        self(obs,plan)
        return self.last_scores.copy()

    def stats(self):
        return dict(beam=self.beam,line_power=self.line_power,compiled=bool(self.fast),calls=self.calls,expanded_boards=self.expansions,
            selector_p50_ms=float(np.median(self.times)*1000) if self.times else 0,
            selector_p95_ms=float(np.percentile(self.times,95)*1000) if self.times else 0,
            selector_max_ms=max(self.times,default=0)*1000,
            information='Current board/piece and single visible next piece only; unknown next-next never queried')


def episode(seed, *, beam=3, baseline=False, runtime=True, output=None,tap_period=0,jit=False,line_power=1,
            max_frames=120000, wall_seconds=600):
    lookahead = NextPiece(beam,tap_period,jit and not baseline,line_power)
    controller = FeedbackController(fixed_reaction=not runtime,
        context_selector=None if baseline else lookahead,tap_period=tap_period)
    with OriginalGame() as game:
        o = game.reset(start_level=18,seed=seed)
        started = time.perf_counter()
        error = None
        while (not o.over and (o.level<30 or controller.stage is not None)
               and o.frames<max_frames and time.perf_counter()-started<wall_seconds):
            before = o
            try:
                o = game.step(controller.command(o))
                controller.transition(before,o)
            except (AssertionError,RuntimeError,ValueError) as exc:
                error = repr(exc)
                break
        record = game.replay()
        with OriginalGame() as replayed:
            replayed.load_replay(record)
        result = dict(seed=seed,policy='one-ply heuristic' if baseline else 'visible-next-piece beam',
            beam=0 if baseline else beam,level=o.level,lines=o.lines,score=o.score,
            placements=o.placements,frames=o.frames,success=o.level>=30 and not error,
            over=o.over,error=error,censored=not o.over and o.level<30,
            wall_seconds=time.perf_counter()-started,exact_ram_rgb_replay=True,
            controller=controller.stats(),lookahead=lookahead.stats(),
            scope='Original-game development comparison; not reserved acceptance or RL result')
        if output:
            output = Path(output)
            output.mkdir(parents=True,exist_ok=True)
            game.save_replay(output/'replay.json')
            (output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds',type=int,nargs='+',default=[17,33])
    p.add_argument('--beam',type=int,default=3)
    p.add_argument('--baseline',action='store_true')
    p.add_argument('--tap-period',type=int,default=0)
    p.add_argument('--jit',action='store_true')
    p.add_argument('--line-power',type=int,choices=[1,2],default=1)
    p.add_argument('--jit-deps',type=Path)
    p.add_argument('--fixed-reaction',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    if args.jit_deps:
        import sys
        sys.path.insert(0,str(args.jit_deps.resolve()))
    # Use spare logical CPUs8/10; neither the learner5/7 nor modern worker2/4.
    from nes_original.learn import cpu_limits
    cpu_limits(mask=1280)
    if args.output.exists():
        p.error('Use a new output directory to preserve prior experiments')
    args.output.mkdir(parents=True)
    rows = []
    for seed in args.seeds:
        row = episode(seed,beam=args.beam,baseline=args.baseline,runtime=not args.fixed_reaction,tap_period=args.tap_period,jit=args.jit,line_power=args.line_power,
                      output=args.output/f'seed{seed}')
        rows.append(row)
        (args.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(row),flush=True)


if __name__ == '__main__':
    main()
