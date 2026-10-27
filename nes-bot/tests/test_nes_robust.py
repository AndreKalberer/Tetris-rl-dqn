import importlib.util
from types import SimpleNamespace
import unittest

import numpy as np

from nes.actions import ActionRules, enumerate_actions
from nes.engine import Control, Observation, Phase, SPAWN, control_frame
from nes_original.planner import proposals


def fixture():
    board = np.zeros((20, 10), dtype=bool)
    for col, height in enumerate((2, 5, 3, 0, 3, 0, 4, 5, 5, 3)):
        if height:
            board[-height:, col] = True
    return SimpleNamespace(board=board, x=5, y=2, orientation=7, fall=0, das=0,
        soft=0, soft_points=0, previous=0, pending=0, input_pipeline=0,
        next_orientation=2, phase=1, level=29, lines=230, score=0,
        placements=0, frames=0, start_level=18)


@unittest.skipUnless(importlib.util.find_spec('numba'), 'Optional isolated compiled search dependency')
class RiskSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nes_original.robust import RiskAwareNextPiece
        cls.policy_class = RiskAwareNextPiece
        cls.policy = RiskAwareNextPiece(root_beam=4, future_beam=3)

    def test_compiled_afterboards_controls_and_scores_match_python_family(self):
        from nes_original.robust import _candidates
        rng = np.random.default_rng(299)
        for case in range(28):
            board = np.zeros((20, 10), dtype=bool)
            for col, height in enumerate(rng.integers(0, 14, 10)):
                if height:
                    board[-height:, col] = True
            state = Control(orientation=SPAWN[case % 7], y=2, soft=0)
            level = (18, 29, 138, 235)[case % 4]
            obs = Observation(board, state, 0, Phase.ACTIVE, level, 230, 0, 0, 0, 18)
            actions = enumerate_actions(obs, ActionRules(horizontal_period=2))
            boards, controls, lines, terminal, scores = _candidates(
                board, tuple(vars(state).values()), level, 2, 3, 48, 4, 1, 6, .15, 1000.)
            reference = {(b.astype(bool).tobytes(), c.das, bool(dead)): (int(clear), float(score))
                for b, c, dead, clear, score in zip(actions.boards, actions.controls, actions.terminal,
                    actions.lines, actions.heuristic)}
            actual_keys = {(b.tobytes(), int(c[4]), bool(dead))
                for b, c, dead in zip(boards, controls, terminal)}
            self.assertEqual(actual_keys, set(reference))
            for b, c, clear, dead, score in zip(boards, controls, lines, terminal, scores):
                height = np.where(b.any(axis=0), 20-b.argmax(axis=0), 0).max()
                expected_lines, expected_score = reference[(b.tobytes(), int(c[4]), bool(dead))]
                self.assertEqual(clear, expected_lines)
                self.assertAlmostEqual(score, expected_score-.15*max(0, height-6)**2, places=3)

    def test_seven_unknown_types_match_independent_python_enumeration(self):
        from nes_original.robust import _hypotheses
        obs = fixture()
        actual = _hypotheses(obs.board, 0, 29, 2, 3, 48, 4, 1, 6, .15, 1000.)
        expected = []
        for pose in SPAWN:
            state = Control(orientation=pose, soft=0)
            for _ in range(2):
                state, failed = control_frame(obs.board, state, 0, 29)
                self.assertFalse(failed)
            actions = enumerate_actions(Observation(obs.board, state, 0, Phase.ACTIVE,
                29, 230, 0, 0, 0, 18), ActionRules(horizontal_period=2))
            occupied = actions.boards != 0
            heights = np.where(occupied.any(axis=1), 20-occupied.argmax(axis=1), 0).max(axis=1)
            expected.append(np.max(actions.heuristic-.15*np.maximum(0, heights-6)**2))
        np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=0)

    def test_visible_only_deterministic_and_input_boards_unchanged(self):
        class VisibleOnly:
            def __init__(self, obs):
                self.__dict__.update(vars(obs))
            def __getattr__(self, name):
                raise AssertionError('Policy read undeclared/hidden state: '+name)
        obs = VisibleOnly(fixture())
        plan = proposals(obs, ActionRules(horizontal_period=2))
        before, candidate_boards = obs.board.copy(), plan.actions.boards.copy()
        selected = self.policy(obs, plan)
        self.assertEqual(selected, self.policy(obs, plan))
        self.assertIn(selected, range(len(plan.paths)))
        np.testing.assert_array_equal(obs.board, before)
        np.testing.assert_array_equal(plan.actions.boards, candidate_boards)
        self.assertLessEqual(len(self.policy.last_details), 4)
        self.assertLessEqual(self.policy.hypothesis_searches, 2*4*3*7)
        # Unavailable future state must never influence the same public input.
        obs.rng = object()
        obs.future_queue = object()
        self.assertEqual(selected, self.policy(obs, plan))

    def test_visible_preview_changes_choice_and_hypothesis_bound_is_per_call(self):
        obs = fixture()
        plan = proposals(obs, ActionRules(horizontal_period=2))
        choices = []
        for preview in SPAWN:
            obs.next_orientation = preview
            prior_count = self.policy.hypothesis_searches
            choices.append(self.policy(obs, plan))
            self.assertLessEqual(self.policy.hypothesis_searches-prior_count, 4*3*7)
        self.assertGreater(len(set(choices)), 1)

    def test_blocked_hypothetical_spawn_is_penalized_without_state_mutation(self):
        from nes_original.robust import _hypotheses
        board = np.ones((20, 10), dtype=bool)
        actual = _hypotheses(board, 0, 29, 2, 3, 48, 4, 1, 6, .15, 1234.)
        np.testing.assert_array_equal(actual, np.full(7, -1234.))
        self.assertTrue(board.all())


if __name__ == '__main__':
    unittest.main()
