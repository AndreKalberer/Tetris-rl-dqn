from types import SimpleNamespace
import unittest

import numpy as np

from nes_original.crash_model import (CRASH_MAP,NEEDS_PUSHDOWN,NO_SOFTDROP_LEVELS,CrashMapPrioritizer,
                                      cleared_rows,crash_map_alerts)


class Scores:
    def __init__(self,scores):
        self.scores = scores

    def score_actions(self,observation,plan):
        return np.asarray(self.scores,dtype=float)


def low_single_board():
    board = np.zeros((20,10),np.uint8)
    board[19,:7] = 1     # an I lying in row19 columns 7-9 completes it... with one more cell
    board[19,7] = 1
    return board


def control(x=8,y=19,orientation=18,soft_points=0):
    return SimpleNamespace(x=x,y=y,orientation=orientation,soft_points=soft_points)


def plan(lines,controls=None,terminal=None):
    lines = np.asarray(lines)
    return SimpleNamespace(actions=SimpleNamespace(lines=lines,controls=controls,
        terminal=np.zeros(len(lines),bool) if terminal is None else np.asarray(terminal)))


def observation(level,lines,board=None,placements=100):
    return SimpleNamespace(level=level,lines=lines,score=999999,placements=placements,
        board=np.zeros((20,10),np.uint8) if board is None else board)


class CrashMapTest(unittest.TestCase):
    def test_always_fatal_entries_alert_without_geometry(self):
        rows = crash_map_alerts(observation(169,1635),plan([0,1,2,3]))
        self.assertEqual([r['alert'] is not None for r in rows],[False,False,True,True])
        # The verified 168/170 doubles crashed only with pushdown (model 0% without).
        rows = crash_map_alerts(observation(169,1638),plan([0,1,2,3]),counts=[0]*5)
        self.assertEqual(rows[2]['alert'],None)

    def test_conditions_decide_conditional_entries(self):
        # Horizontal I (orientation 18) at x=8,y=19 fills columns 6..9 of row 19.
        board = np.zeros((20,10),np.uint8)
        board[19,:6] = 1
        c = control(x=8,y=19,orientation=18)
        self.assertEqual(cleared_rows(board,c),[19])
        obs = observation(156,1505,board)
        safe = crash_map_alerts(obs,plan([1],[c]),counts=[0,3,0,0,0])  # 4th single: no carry, low row
        self.assertIsNone(safe[0]['alert'])
        carry = crash_map_alerts(obs,plan([1],[c]),counts=[0,9,0,0,0])  # 10th single
        self.assertIsNotNone(carry[0]['alert'])
        unknown = crash_map_alerts(obs,plan([1],[c]),counts=None)
        self.assertIsNotNone(unknown[0]['alert'])

    def test_zero_line_pushdown_rule(self):
        obs = observation(249,3232)
        rows = crash_map_alerts(obs,plan([0,0,0,0],[control(orientation=17,soft_points=1),
            control(orientation=2,soft_points=0),control(orientation=17,soft_points=2),
            control(orientation=17,soft_points=3)]),counts=[0]*5)
        self.assertEqual([r['alert'] is not None for r in rows],[True,False,True,False])

    def test_prefers_unalerted_survivor(self):
        selector = CrashMapPrioritizer(Scores([1.,0.,5.]))   # best score is a fatal double within169
        self.assertEqual(selector(observation(169,1635),plan([0,1,2])),0)
        self.assertEqual(selector.stats()['selection_tiers']['unalerted_nonterminal'],1)

    def test_falls_back_only_when_every_survivor_is_alerted(self):
        selector = CrashMapPrioritizer(Scores([1.,3.]))
        self.assertEqual(selector(observation(169,1635),plan([2,3],terminal=[False,False])),1)
        self.assertEqual(selector.stats()['selection_tiers']['alerted_nonterminal_fallback'],1)

    def test_tracks_clear_counts_from_game_start(self):
        selector = CrashMapPrioritizer(Scores([0.]))
        for placements,lines in ((0,0),(1,0),(2,1),(3,1),(3,1),(4,4)):
            selector(observation(18,lines,placements=placements),plan([0]))
        self.assertTrue(selector.counts_known)
        self.assertEqual(selector.counts,[2,1,0,1,0])
        selector(observation(18,9,placements=6),plan([0]))  # two placements at once: unknown
        self.assertFalse(selector.counts_known)
        late = CrashMapPrioritizer(Scores([0.]))
        late(observation(18,50,placements=120),plan([0]))  # joined mid-game
        self.assertFalse(late.counts_known)

    def test_preempts_pending_carry_while_safe(self):
        board = np.zeros((20,10),np.uint8)
        board[19,:6] = 1
        c = control(x=8,y=19,orientation=18)
        selector = CrashMapPrioritizer(Scores([5.,1.]))
        selector.counts,selector.counts_known,selector.previous = [0,1099,0,0,0],True,(99,1495)
        # Within 155 the 1,100th single is safe: take it now (index1), not later at 156.
        self.assertEqual(selector(observation(155,1495,board,placements=100),plan([0,1],[c,c])),1)
        self.assertEqual(selector.preemptions,1)
        selector = CrashMapPrioritizer(Scores([5.,1.]),preempt=False)
        selector.counts,selector.counts_known,selector.previous = [0,1099,0,0,0],True,(99,1495)
        self.assertEqual(selector(observation(155,1495,board,placements=100),plan([0,1],[c,c])),0)

    def test_high_stack_accepts_partly_risky_clear(self):
        board = np.zeros((20,10),np.uint8)
        board[19,:6] = 1
        c = control(x=8,y=19,orientation=18)
        def pick(stack_rows):
            b = board.copy()
            b[20-stack_rows:,0] = 1           # a column that sets the stack height
            selector = CrashMapPrioritizer(Scores([5.,1.]),preempt=False)
            selector.counts,selector.counts_known,selector.previous = [0,1099,0,0,0],True,(99,1525)
            return selector(observation(158,1525,b,placements=100),plan([1,0],[c,c]))
        # The 1,100th single within158 is modelled 25% fatal: refused on a low stack,
        # taken (index 0 has the better score) once the stack reaches 11 rows.
        self.assertEqual(pick(6),1)
        self.assertEqual(pick(12),0)

    def test_249_fallback_prefers_most_pushdown(self):
        selector = CrashMapPrioritizer(Scores([9.,1.,5.]),preempt=True)
        selector.counts,selector.counts_known,selector.previous = [0]*5,True,(99,3237)
        cands = [control(orientation=17,soft_points=0),control(orientation=17,soft_points=2),
                 control(orientation=17,soft_points=1)]
        self.assertEqual(selector(observation(249,3237,placements=100),plan([0,0,0],cands)),1)
        self.assertEqual(selector.stats()['selection_tiers']['alerted_nonterminal_fallback'],1)

    def test_softdrop_levels(self):
        self.assertTrue(all(155<=key[1]<=173 for key in CRASH_MAP))
        for level in NEEDS_PUSHDOWN:
            self.assertNotIn(level,NO_SOFTDROP_LEVELS)
        self.assertIn(155,NO_SOFTDROP_LEVELS)
        self.assertIn(248,NO_SOFTDROP_LEVELS)


if __name__=='__main__':
    unittest.main()
