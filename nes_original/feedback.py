"""Native-game feedback actions with approximate planning features.

Unlike the strict preview-audit controller, these actions mean 'try this target
pose with this fixed controller', not 'the predicted after-board will occur'.
Target misses and prediction differences are data, never simulated transitions.
"""
import time

import numpy as np

from nes.actions import ActionRules,rotate_button
from nes.engine import CELLS,FPS,TYPE
from nes_original.env import Buttons,from_game_bits
from nes_original.planner import proposals


class FeedbackController:
    def __init__(self, *, fixed_reaction=False, selector=None, context_selector=None,tap_period=0,
                 no_softdrop_at_1g=False, no_softdrop_levels=(), nonactive_input=0,
                 stall_rescue_input=0, stall_rescue_after=120,
                 stall_rescue_fallback_input=0, stall_rescue_primary_frames=8):
        self.fixed_reaction,self.selector = fixed_reaction,selector
        self.context_selector = context_selector
        self.tap_period = tap_period
        self.no_softdrop_at_1g = bool(no_softdrop_at_1g)
        try:
            levels = tuple(no_softdrop_levels)
        except TypeError as error:
            raise ValueError('No-softdrop levels must be an iterable of integers29..255') from error
        if any(isinstance(level, bool) or not isinstance(level, int) or not 29 <= level <= 255
               for level in levels):
            raise ValueError('No-softdrop levels must be integers29..255; booleans are excluded')
        self.no_softdrop_levels = frozenset(levels)
        if (isinstance(nonactive_input, bool) or not isinstance(nonactive_input, int)
                or nonactive_input not in (0, 1, 2, 32, 64, 128)):
            raise ValueError('Nonactive input must be one legal native button byte0,1,2,32,64,128')
        self.nonactive_input = int(nonactive_input)
        self.nonactive_input_frames = 0
        if (isinstance(stall_rescue_input, bool) or not isinstance(stall_rescue_input, int)
                or stall_rescue_input not in (0, 1, 2, 32, 64, 128)):
            raise ValueError('Stall rescue input must be one legal native button byte0,1,2,32,64,128')
        if (isinstance(stall_rescue_after, bool) or not isinstance(stall_rescue_after, int)
                or not 1 <= stall_rescue_after <= 599):
            raise ValueError('Stall rescue threshold must be an integer1..599 frames')
        self.stall_rescue_input, self.stall_rescue_after = int(stall_rescue_input), stall_rescue_after
        if (isinstance(stall_rescue_fallback_input, bool) or not isinstance(stall_rescue_fallback_input, int)
                or stall_rescue_fallback_input not in (0, 1, 2, 32, 64, 128)):
            raise ValueError('Stall rescue fallback must be one legal native button byte0,1,2,32,64,128')
        if (isinstance(stall_rescue_primary_frames, bool) or not isinstance(stall_rescue_primary_frames, int)
                or not 1 <= stall_rescue_primary_frames <= 599):
            raise ValueError('Stall rescue primary duration must be an integer1..599 frames')
        if stall_rescue_fallback_input and (not stall_rescue_input
                or stall_rescue_after+stall_rescue_primary_frames > 599):
            raise ValueError('Configured fallback requires primary input and an escalation deadline<=599')
        self.stall_rescue_fallback_input = int(stall_rescue_fallback_input)
        self.stall_rescue_primary_frames = stall_rescue_primary_frames
        self.stall_rescue_attempts = self.stall_rescue_frames = 0
        self.stall_rescue_fallback_activations = self.stall_rescue_fallback_frames = 0
        self._reset_stall_window()
        self.rules = ActionRules(horizontal_period=tap_period)
        self.stage = None
        self.idle = 0
        self.plans = self.overruns = self.extra_idle_frames = 0
        self.target_misses = self.proposal_board_differences = self.board_audit_failures = 0
        self.top_boundary_locks = 0
        self.piece_type_differences = 0
        self.planning_seconds = 0.0

    def start_plan(self,obs,plan,selected):
        if self.stage:
            raise RuntimeError('Cannot replace an in-progress target action')
        self.plan,self.selected = plan,selected
        target = plan.actions.controls[selected] if plan else obs
        self.target_x,self.target_orientation = target.x,target.orientation
        self.before = obs
        self.elapsed = self.horizontal = 0
        self.dropping = plan is None
        self.stage = 'active'
        self.above_board = False

    def _drop_button(self, obs):
        # At ordinary1G, gravity already advances each frame. This optional
        # input variation removes push-down points; it does not patch physics.
        # Retain Down while the signed initial soft timer is negative (0x80..
        # 0xFF), because Down cancels that timer and can change vertical motion.
        # Restrict this to ordinary fall-byte values too: the ROM's signed
        # comparison can skip natural gravity after a byte wrap/high value.
        enabled = self.no_softdrop_at_1g or obs.level in self.no_softdrop_levels
        if (enabled and obs.level >= 29 and 0 <= obs.soft < 128
                and 0 <= obs.fall <= 127):
            return 0
        return int(Buttons.DOWN)

    def command(self,obs):
        if self.stall_rescue_input and obs.phase == 1:
            self._reset_stall_window()
        if self.idle:
            self.idle -= 1
            self.extra_idle_frames += 1
            return self._nonactive_button(obs) if obs.phase != 1 else 0
        if obs.phase != 1:
            return self._nonactive_button(obs)
        if self.stage == 'settle':
            return 0
        if self.stage is None:
            start = time.perf_counter()
            plan = proposals(obs,self.rules,board_input=True)
            selected = (int(self.context_selector(obs,plan)) if plan and self.context_selector else
                        int(self.selector(plan)) if plan and self.selector else
                        int(plan.actions.heuristic.argmax()) if plan else 0)
            elapsed = time.perf_counter()-start
            self.plans += 1
            self.planning_seconds += elapsed
            if elapsed > 3/FPS:
                self.overruns += 1
                if not self.fixed_reaction:
                    self.idle = max(0,int(np.ceil(elapsed*FPS))-1)
                    self.extra_idle_frames += 1
                    return 0
            self.start_plan(obs,plan,selected)
        frame = self.elapsed
        self.elapsed += 1
        if frame < 3:
            return 0
        navigation = frame-3
        if TYPE[obs.orientation] != TYPE[self.target_orientation]:
            self.piece_type_differences += 1
            self.dropping = True
        if navigation >= 48 or self.dropping:
            self.dropping = True
            return self._drop_button(obs)
        # Horizontal changes at most once every six native frames. Rotation
        # pulses at most once every four. Own pending input must settle before
        # treating a target as reached. All remaining physics are original code.
        if self.tap_period:
            self.horizontal = (int(Buttons.LEFT if obs.x>self.target_x else
                                   Buttons.RIGHT if obs.x<self.target_x else 0)
                               if navigation%self.tap_period == 0 else 0)
        elif navigation%6 == 0:
            self.horizontal = int(Buttons.LEFT if obs.x>self.target_x else
                                  Buttons.RIGHT if obs.x<self.target_x else 0)
        rotation = 0
        if obs.orientation != self.target_orientation and navigation%4 == 0:
            rotation = from_game_bits(rotate_button(obs.orientation,self.target_orientation))
        if (obs.x == self.target_x and obs.orientation == self.target_orientation
                and not self.horizontal and not obs.pending&3 and not obs.previous&3):
            self.dropping = True
            return self._drop_button(obs)
        return self.horizontal|rotation

    def _reset_stall_window(self):
        self._stall_key = self._stall_start_frame = self._stall_last_frame = None
        self._stall_activated = False
        self._stall_fallback_activated = False
        self._stall_last_rescue_frame = None

    def _stall_button(self, obs):
        """Legal timing strategy on unchanged public state, not a crash verdict."""
        frame = getattr(obs, 'frames', None)
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            self._reset_stall_window()
            return None
        try:
            key = (obs.phase, obs.orientation, obs.x, obs.y, obs.level, obs.lines,
                   obs.placements, obs.score, obs.board.tobytes())
        except (AttributeError, TypeError, ValueError):
            self._reset_stall_window()
            return None
        if self._stall_last_frame is not None and frame < self._stall_last_frame:
            self._reset_stall_window()
            return None
        if key != self._stall_key:
            self._reset_stall_window()
            self._stall_key = key
            self._stall_start_frame = self._stall_last_frame = frame
            return None
        self._stall_last_frame = frame
        elapsed = frame-self._stall_start_frame
        if elapsed < self.stall_rescue_after:
            return None
        if not self._stall_activated:
            self._stall_activated = True
            self.stall_rescue_attempts += 1
        fallback = (self.stall_rescue_fallback_input != 0
                    and elapsed >= self.stall_rescue_after+self.stall_rescue_primary_frames)
        if fallback and not self._stall_fallback_activated:
            self._stall_fallback_activated = True
            self.stall_rescue_fallback_activations += 1
        if frame != self._stall_last_rescue_frame:
            # Total rescue frames includes fallback; fallback_frames is a subset.
            self.stall_rescue_frames += 1
            if fallback:
                self.stall_rescue_fallback_frames += 1
            self._stall_last_rescue_frame = frame
        return self.stall_rescue_fallback_input if fallback else self.stall_rescue_input

    def _nonactive_button(self, obs):
        # Count all calls routed to this profile, including default neutral0.
        # Active settling/reaction/navigation/drop never call this method.
        self.nonactive_input_frames += 1
        if self.stall_rescue_input:
            rescue = self._stall_button(obs)
            if rescue is not None:
                return rescue
        return self.nonactive_input

    def transition(self,before,after):
        if self.stage == 'active' and before.phase == 1 and after.phase != 1:
            if after.phase != 2:
                raise AssertionError('Original game left active control without lock phase')
            self.stage = 'entry'
            if (after.x,after.orientation) != (self.target_x,self.target_orientation):
                self.target_misses += 1
            self.above_board = any(after.y+dy<0 for dx,dy in CELLS[after.orientation])
        if self.stage == 'entry' and after.phase == 1 and not after.over:
            # Saved player RAM copies phase before orientation. A PPU boundary
            # can expose a new active flag with the old piece ID. Require two
            # consecutive matching active orientations before the next choice.
            self.spawn_orientation = after.orientation
            self.stage = 'settle'
            return
        if self.stage == 'settle':
            if after.phase == 2:
                self.above_board |= any(after.y+dy<0 for dx,dy in CELLS[after.orientation])
                self.stage = 'entry'
                return
            if after.phase == 1 and after.orientation != self.spawn_orientation:
                self.spawn_orientation = after.orientation
                return
        if ((self.stage == 'settle' and after.phase == 1) or
                (self.stage in ('entry','settle') and after.over)):
            old = self.before
            placed,cleared = after.placements-old.placements,after.lines-old.lines
            expected = int(old.board.sum()) + 4*placed - 10*cleared
            if self.above_board:
                # Original program can write above-screen cells outside the
                # visible 200-byte board. Preserve and count this NES quirk.
                self.top_boundary_locks += 1
            elif int(after.board.sum()) != expected:
                self.board_audit_failures += 1
                raise AssertionError('Native board occupancy/line/placement accounting diverged')
            if self.plan and not np.array_equal(after.board,self.plan.actions.boards[self.selected] != 0):
                self.proposal_board_differences += 1
            self.stage = None

    def stats(self):
        return dict(plans=self.plans,planning_seconds=self.planning_seconds,
            planning_overruns=self.overruns,extra_idle_frames=self.extra_idle_frames,
            runtime_latency_charged=not self.fixed_reaction,target_misses=self.target_misses,
            proposal_board_differences=self.proposal_board_differences,
            board_audit_failures=self.board_audit_failures,top_boundary_locks=self.top_boundary_locks,
            piece_type_differences=self.piece_type_differences,
            action_interface='Native feedback targets; approximate proposal features',
            reaction_frames=3,navigation_frames=48,horizontal_change_period=6 if not self.tap_period else 1,
            horizontal_pulse_period=self.tap_period,rotation_period=4,
            no_softdrop_at_1g=self.no_softdrop_at_1g,
            no_softdrop_levels=sorted(self.no_softdrop_levels),
            nonactive_input=self.nonactive_input,
            nonactive_input_frames=self.nonactive_input_frames,
            stall_rescue_input=self.stall_rescue_input,
            stall_rescue_after=self.stall_rescue_after,
            stall_rescue_attempts=self.stall_rescue_attempts,
            stall_rescue_frames=self.stall_rescue_frames,
            stall_rescue_fallback_input=self.stall_rescue_fallback_input,
            stall_rescue_primary_frames=self.stall_rescue_primary_frames,
            stall_rescue_fallback_activations=self.stall_rescue_fallback_activations,
            stall_rescue_fallback_frames=self.stall_rescue_fallback_frames)
