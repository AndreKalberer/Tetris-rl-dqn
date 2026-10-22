"""Play/evaluate a controller in original NTSC A-Type; record exact inputs."""
import argparse
from collections import deque
import json
from pathlib import Path
import time

import numpy as np

from nes.engine import FPS, control_frame
from nes_original.env import Buttons, OriginalGame
from nes_original.planner import control, proposals


class Controller:
    def __init__(self, *, fixed_reaction=False, policy='heuristic', seed=0, selector=None, board_input=False):
        self.fixed_reaction = fixed_reaction
        self.policy = policy
        self.rng = np.random.default_rng(seed)
        self.selector, self.board_input = selector, board_input
        self.queue = deque()
        self.idle = 0
        self.expected = None
        self.stage = None
        self.plans = self.overruns = self.extra_idle_frames = self.mismatches = 0
        self.drop_adjustments = self.timer_differences = self.drop_frames = 0
        self.planning_seconds = 0.0

    def command(self, obs):
        if self.idle:
            self.idle -= 1
            self.extra_idle_frames += 1
            return 0
        if obs.phase != 1:
            return 0
        if self.queue:
            return self.queue.popleft()
        if self.stage == 'path':
            # PPU boundaries can split main-loop RAM updates. A proposed lock
            # frame is an estimate: keep legal Down held until the real game
            # reports lock. Never advance to another piece on a predicted lock.
            self.drop_frames += 1
            if self.drop_frames > 16:
                self.mismatches += 1
                raise AssertionError('Original-game lock did not arrive within drop grace')
            self.drop_adjustments += 1
            return int(Buttons.DOWN)
        started = time.perf_counter()
        plan = proposals(obs,board_input=self.board_input)
        if plan is not None:
            selected = (int(self.selector(plan)) if self.selector is not None else
                        int(plan.actions.heuristic.argmax()) if self.policy == 'heuristic'
                        else int(self.rng.integers(len(plan.paths))))
        elapsed = time.perf_counter()-started
        self.planning_seconds += elapsed
        self.plans += 1
        if elapsed > 3/FPS:
            self.overruns += 1
            if not self.fixed_reaction:
                # Discard the stale plan, spend every elapsed frame as idle,
                # then observe and replan. The queued hardware input still runs.
                self.idle = max(0,int(np.ceil(elapsed*FPS))-1)
                self.extra_idle_frames += 1
                return 0
        self.start_plan(obs,plan,selected if plan is not None else 0)
        return self.queue.popleft()

    def start_plan(self, obs, plan, selected):
        """Prepare a chosen legal path; used by actual-emulator training too."""
        if self.stage is not None or self.queue:
            raise RuntimeError('Cannot replace a live controller path')
        if plan is None:
            state, _ = control_frame(obs.board,control(obs),obs.pending,obs.level)
            self.expected = (state,None,None,None)
            self.queue = deque([0])
        else:
            self.expected = (plan.actions.controls[selected],plan.actions.boards[selected] != 0,
                obs.lines+int(plan.actions.lines[selected]),bool(plan.actions.terminal[selected]))
            self.queue = deque(plan.paths[selected])
        self.stage = 'path'
        self.drop_frames = 0

    def transition(self, before, after):
        def fail(message):
            self.mismatches += 1
            raise AssertionError(message)
        if self.stage == 'path' and before.phase == 1:
            if after.phase != 1:
                actual, expected = control(after), self.expected[0]
                if after.phase != 2 or (actual.x,actual.y,actual.orientation) != (expected.x,expected.y,expected.orientation):
                    fail('Original-game lock control differs from planned path')
                if actual != expected:
                    self.timer_differences += 1
                if self.queue:
                    self.drop_adjustments += 1
                    self.queue.clear()
                self.stage = 'entry'
        if self.stage == 'entry' and (after.phase == 1 or after.over):
            _,board,lines,terminal = self.expected
            if board is not None:
                if not np.array_equal(after.board,board) or after.lines != lines or after.over != terminal:
                    fail('Original-game board/lines/termination differs from proposal')
            self.expected = self.stage = None

    def stats(self):
        return dict(policy=self.policy,plans=self.plans,planning_seconds=self.planning_seconds,
            planning_overruns=self.overruns,extra_idle_frames=self.extra_idle_frames,
            path_mismatches=self.mismatches,runtime_latency_charged=not self.fixed_reaction,
            reaction_frames=3,input_pipeline_frames=1,
            drop_timing_adjustments=self.drop_adjustments,lock_timer_differences=self.timer_differences)


def play(*, level=18, seed=17, max_frames=120000, max_placements=5000,
         wall_seconds=300, fixed_reaction=False, policy='heuristic', output=None,
         callback=None):
    controller = Controller(fixed_reaction=fixed_reaction,policy=policy,seed=seed)
    with OriginalGame() as game:
        obs = game.reset(start_level=level,seed=seed)
        started = time.perf_counter()
        error = None
        while (not obs.over and obs.frames < max_frames and obs.placements < max_placements
               and time.perf_counter()-started < wall_seconds):
            before = obs
            try:
                action = controller.command(obs)
            except AssertionError as exc:
                error = str(exc)
                break
            obs = game.step(action)
            if callback:
                callback(game)
            try:
                controller.transition(before,obs)
            except AssertionError as exc:
                error = str(exc)
                break
        reason = ('path_mismatch' if error else 'natural_top_out' if obs.over else
                  'frame_limit' if obs.frames >= max_frames else 'placement_limit'
                  if obs.placements >= max_placements else 'wall_limit')
        record = game.replay()
        result = dict(record['result'],start_level=level,menu_wait_seed=seed,
            stopped_by=reason,error=error,wall_seconds=time.perf_counter()-started,
            completed=obs.over and not error,controller=controller.stats(),
            scope='Original Nintendo USA game in nes-py; development probe, not reserved benchmark')
        # Reconstruct in another native instance, including power-on and menus.
        with OriginalGame() as replayed:
            replayed.load_replay(record)
        result['exact_ram_rgb_replay'] = True
        if output:
            output = Path(output)
            output.mkdir(parents=True,exist_ok=True)
            game.save_replay(output/'replay.json')
            for name,data in (('result.json',result),):
                (output/name).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
            np.save(output/'last-screen.npy',game.screen())
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--level',type=int,default=18)
    parser.add_argument('--seed',type=int,default=17,help='Level-menu idle frame count, not an RNG RAM patch')
    parser.add_argument('--max-frames',type=int,default=120000)
    parser.add_argument('--max-placements',type=int,default=5000)
    parser.add_argument('--wall-seconds',type=float,default=300)
    parser.add_argument('--fixed-reaction',action='store_true',help='Engineering only: CPU latency logged but not charged')
    parser.add_argument('--policy',choices=['heuristic','random'],default='heuristic')
    parser.add_argument('--output',type=Path,default=Path('runs/nes-original-development/probe'))
    args = parser.parse_args()
    result = play(level=args.level,seed=args.seed,max_frames=args.max_frames,
        max_placements=args.max_placements,wall_seconds=args.wall_seconds,
        fixed_reaction=args.fixed_reaction,policy=args.policy,output=args.output)
    print(json.dumps(result,indent=2))
    if result['error']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
