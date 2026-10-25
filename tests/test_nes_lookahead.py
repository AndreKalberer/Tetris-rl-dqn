from dataclasses import replace
import importlib.util
from types import SimpleNamespace
import unittest

import numpy as np

from nes.actions import ActionRules,enumerate_actions,structural
from nes.engine import CELLS,Control,Observation,Phase,SPAWN,control_frame,TYPE
from nes_original.lookahead import NextPiece
from nes_original.planner import proposals


def fixture(next_pose=2,level=18):
    board=np.zeros((20,10),dtype=bool)
    for x,h in enumerate([2,5,3,0,3,0,4,5,5,3]):
        if h:
            board[-h:,x]=True
    return SimpleNamespace(board=board,x=5,y=0,orientation=7,fall=0,das=0,
        soft=160,soft_points=0,previous=0,pending=0,next_orientation=next_pose,
        phase=1,level=level,lines=0,score=0,placements=0,frames=0,start_level=18)


class LookaheadTests(unittest.TestCase):
    def test_well_is_explicit_and_default_prior_is_unchanged(self):
        from nes_original.scoring import action_scores
        plan=proposals(fixture())
        self.assertTrue(np.array_equal(action_scores(plan.actions),plan.actions.heuristic))
        board=np.zeros((20,10),dtype=bool)
        board[-4:,:9]=True
        actions=SimpleNamespace(boards=np.stack([board,board[:,::-1]]),
            lines=np.zeros(2,dtype=int),terminal=np.zeros(2,dtype=bool))
        right=action_scores(actions,2,9)
        left=action_scores(actions,2,0)
        self.assertGreater(right[0],right[1])
        self.assertTrue(np.array_equal(right,left[::-1]))

    def test_dead_core_cannot_be_reported_as_a_successful_clear(self):
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import MagicMock
        from nes_original.lookahead import episode
        game=MagicMock()
        game.__enter__.return_value=game
        game.reset.return_value=SimpleNamespace(level=30,lines=240,score=123,
            placements=600,frames=33000,over=False)
        game.replay.side_effect=RuntimeError('core no longer responds')
        game.inputs=bytearray([0,128,0])
        base=Path.cwd()/'.pytest-work/lookahead-fault'
        base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temporary:
            result=episode(17,output=temporary,game_factory=lambda:game)
            self.assertFalse(result['success'])
            self.assertFalse(result['exact_ram_rgb_replay'])
            self.assertIn('core no longer responds',result['replay_error'])
            folder=Path(temporary)
            self.assertFalse((folder/'replay.json').exists())
            raw=json.loads((folder/'incomplete-input-diagnostic.json').read_text())
            self.assertEqual(raw['input_hex'],'008000')

    def test_visible_next_changes_choice_without_mutating_board(self):
        obs=fixture()
        original=obs.board.copy()
        plan=proposals(obs)
        choices=[]
        for next_pose in SPAWN:
            obs.next_orientation=next_pose
            selector=NextPiece(3)
            choices.append(selector(obs,plan))
            self.assertLessEqual(selector.expansions,3)
        self.assertGreater(len(set(choices)),1)
        self.assertTrue(np.array_equal(obs.board,original))

    def test_rapid_profile_has_primitive_releases_and_full_model_width_at29(self):
        occupied={0:set(),2:set()}
        for period in (0,2):
            obs=fixture(level=29)
            obs.board[:]=False
            obs.y=2
            obs.soft=0  # Steady spawn, excluding the initial game's grace timer.
            obs.lines=230
            for orientation in SPAWN:
                obs.orientation=orientation
                plan=proposals(obs,ActionRules(horizontal_period=period))
                for i,path in enumerate(plan.paths):
                    occupied[period].update(np.flatnonzero(plan.actions.boards[i].any(axis=0)).tolist())
                    if period:
                        self.assertFalse(any(a&0xC0 and b&0xC0 for a,b in zip(path,path[1:])))
        self.assertEqual(occupied[0],set(range(2,8)))
        self.assertEqual(occupied[2],set(range(10)))


@unittest.skipUnless(importlib.util.find_spec('numba'),'Optional isolated CPU compilation dependency')
class CompiledTests(unittest.TestCase):
    def test_well_search_matches_independent_numpy_enumeration(self):
        from nes_original.fast import best_future
        from nes_original.scoring import action_scores
        rng=np.random.default_rng(891)
        for case in range(60):
            board=np.zeros((20,10),dtype=bool)
            for x,h in enumerate(rng.integers(0,12,10)):
                if h: board[-h:,x]=True
            state=Control(orientation=SPAWN[case%7],fall=1,soft=0)
            level=(18,19,29)[case%3]
            well=(0,9)[case%2]
            obs=Observation(board,state,0,Phase.ACTIVE,level,0,0,0,0,18)
            actions=enumerate_actions(obs,ActionRules(horizontal_period=2))
            expected=action_scores(actions,2,well).max()
            actual=best_future(board,tuple(vars(state).values()),level,2,3,48,4,2,well)
            self.assertAlmostEqual(float(expected),actual,places=3)

    def test_compiled_frame_matches_reference_for_randomized_controls(self):
        from nes_original.fast import frame
        rng=np.random.default_rng(917)
        for _ in range(3000):
            board=(rng.random((20,10))<.12)
            board[:5]=False
            state=Control(x=int(rng.integers(1,9)),y=int(rng.integers(0,18)),
                orientation=int(rng.integers(19)),fall=int(rng.integers(256)),
                das=int(rng.integers(256)),soft=int(rng.integers(256)),
                soft_points=int(rng.integers(256)),previous=int(rng.integers(256))&0xC7)
            buttons=int(rng.integers(256))&0xC7
            level=int(rng.choice([0,9,18,19,28,29]))
            expected,failed=control_frame(board,state,buttons,level)
            actual,actual_failed=frame(board,tuple(vars(state).values()),buttons,level)
            self.assertEqual(tuple(vars(expected).values()),actual)
            self.assertEqual(failed,actual_failed)

    def test_best_future_matches_reference_across_boards_pieces_speeds_profiles(self):
        from nes_original.fast import best_future
        rng=np.random.default_rng(981)
        for case in range(120):
            board=np.zeros((20,10),dtype=bool)
            for x,h in enumerate(rng.integers(0,14,10)):
                if h:
                    board[-h:,x]=True
            pose=SPAWN[case%7]
            level=(0,18,19,29)[case%4]
            period=(0,2)[case%2]
            state=Control(orientation=pose,fall=1,soft=0)
            obs=Observation(board,state,0,Phase.ACTIVE,level,0,0,0,0,18)
            expected=enumerate_actions(obs,ActionRules(horizontal_period=period)).heuristic.max()
            actual=best_future(board,tuple(vars(state).values()),level,period)
            self.assertAlmostEqual(float(expected),actual,places=3)

    def test_compiled_line_clear_scoring(self):
        from nes_original.fast import lock_score
        for orientation,y,rows in [(18,19,1),(10,18,2),(17,18,4)]:
            board=np.zeros((20,10),dtype=bool)
            board[-rows:]=True
            for dx,dy in CELLS[orientation]:
                board[y+dy,5+dx]=False
            score=lock_score(board,(5,y,orientation,0,0,0,0,0))
            self.assertAlmostEqual(score,.76*rows,places=5)


if __name__=='__main__':
    unittest.main()
