"""Fast visible-state proposals, checked against original-game execution.

The provisional control model proposes paths; it does not transition this game.
nes-py has one controller frame in flight; the reference input-polled boundary
has none. Honor the observation's declared pipeline when projecting/draining.
"""
from dataclasses import dataclass

import numpy as np

from nes.actions import ActionRules, enumerate_actions
from nes.engine import Control, Observation, Phase, TYPE, control_frame
from nes_original.env import from_game_bits


@dataclass
class Plan:
    actions: object
    paths: tuple


def control(observation):
    o = observation
    return Control(o.x,o.y,o.orientation,o.fall,o.das,o.soft,o.soft_points,o.previous)


def proposals(observation, rules=ActionRules(), *, board_input=False):
    o = observation
    if o.phase != 1:
        raise ValueError('Plan only for an active original-game piece')
    pipeline = getattr(o,'input_pipeline',1)
    state, failed = (control_frame(o.board, control(o), o.pending, o.level)
                     if pipeline else (control(o),False))
    if failed:
        # The in-flight input already locks the piece; no proposal can change it.
        return None
    projected = Observation(o.board,state,TYPE[o.next_orientation],Phase.ACTIVE,
        o.level,o.lines,o.score,o.placements,o.frames+pipeline,o.start_level)
    actions = enumerate_actions(projected,rules,board_input=board_input)
    paths = tuple(tuple(from_game_bits(button) for button in path) + ((0,) if pipeline else ())
                  for path in actions.paths)
    return Plan(actions, paths)


def execute(game, plan, selected, *, callback=None):
    """Execute one proposal in original code, then verify control/board/lines.

    This is a development/training action primitive. It does not charge CPU
    planning latency; the runtime controller must account for that separately.
    """
    before = game.observe()

    def tick(action):
        obs = game.step(action)
        if callback:
            callback(game, action)
        return obs

    if plan is None:
        obs = tick(0)
        if obs.phase != 2:
            raise AssertionError('In-flight original-game input did not lock as predicted')
    else:
        path = plan.paths[selected]
        for frame, action in enumerate(path):
            obs = tick(action)
            if obs.phase != 1:
                break
        if frame != len(path)-1 or obs.phase != 2:
            raise AssertionError('Original-game control phase differs from proposal')
        if control(obs) != plan.actions.controls[selected]:
            raise AssertionError('Original-game control state differs from proposal')
    for _ in range(300):
        if game.over or game.observe().phase == 1:
            break
        tick(0)
    else:
        raise AssertionError('Original-game entry processing exceeded its bound')
    obs = game.observe()
    if plan is not None:
        if obs.lines-before.lines != int(plan.actions.lines[selected]):
            raise AssertionError('Original-game cleared lines differ from proposal')
        if not np.array_equal(obs.board, plan.actions.boards[selected] != 0):
            raise AssertionError('Original-game locked board differs from proposal')
        if obs.over != bool(plan.actions.terminal[selected]):
            raise AssertionError('Original-game termination differs from proposal')
    return obs
