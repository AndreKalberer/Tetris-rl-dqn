"""Public-observation contract regressions; mock persistence, no native game."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from nes_original.feedback import ControllerObservationError, FeedbackController


def observation(frame=1, **changes):
    values = dict(board=np.zeros((20, 10), dtype=bool), x=5, y=1,
        orientation=2, next_orientation=7, fall=0, das=0, soft=0, soft_points=0,
        previous=0, pending=0, input_pipeline=0, phase=1, level=18, lines=0,
        score=0, placements=0, frames=frame, start_level=18, over=False)
    values.update(changes)
    return SimpleNamespace(**values)


class ObservationContractTests(unittest.TestCase):
    def entry_controller(self):
        c = FeedbackController(fixed_reaction=True, tap_period=2,
            stall_rescue_input=128, stall_rescue_fallback_input=1)
        c.start_plan(observation(), None, 0)
        c.stage = 'entry'
        return c

    def test_two_invalid_active_samples_never_certify_settled_piece(self):
        c = self.entry_controller()
        first = observation(10, orientation=19)
        c.transition(observation(9, phase=5, orientation=19), first)
        self.assertEqual(c.stage, 'settle')
        self.assertEqual(c.command(first), 0)
        self.assertEqual(c.stage, 'settle')
        with self.assertRaisesRegex(ControllerObservationError, 'distinct public frames'):
            c.transition(first, observation(11, orientation=19))
        self.assertIsNotNone(c.stage)
        self.assertEqual(c.stats()['stall_rescue_attempts'], 0)

    def test_duplicate_invalid_samples_do_not_advance_frame_evidence(self):
        c = FeedbackController(stall_rescue_input=128)
        bad = observation(10, orientation=19)
        for _ in range(100):
            self.assertEqual(c.command(bad), 0)
        self.assertEqual(c.stage, 'observation_wait')
        with self.assertRaises(ControllerObservationError):
            c.command(observation(11, orientation=19))

    def test_transient_invalid_entry_requires_two_matching_valid_samples(self):
        c = self.entry_controller()
        bad = observation(10, orientation=19)
        c.transition(observation(9, phase=5), bad)
        first = observation(11)
        c.transition(bad, first)
        self.assertEqual(c.stage, 'settle')
        self.assertEqual(c.command(first), 0)
        second = observation(12)
        c.transition(first, second)
        self.assertIsNone(c.stage)
        self.assertEqual(c.board_audit_failures, 0)

    def test_normal_valid_spawn_preserves_existing_two_sample_timing(self):
        c = self.entry_controller()
        first, second = observation(10), observation(11)
        c.transition(observation(9, phase=5), first)
        self.assertEqual(c.stage, 'settle')
        self.assertEqual(c.command(first), 0)
        c.transition(first, second)
        self.assertIsNone(c.stage)

    def test_invalid_before_planning_recovers_only_after_two_valid_observations(self):
        c = FeedbackController(fixed_reaction=True)
        bad = observation(10, orientation=19)
        self.assertEqual(c.command(bad), 0)
        self.assertEqual(c.stage, 'observation_wait')
        first = observation(11)
        c.transition(bad, first)
        self.assertEqual(c.stage, 'observation_wait')
        self.assertEqual(c.command(first), 0)
        c.transition(first, observation(12))
        self.assertIsNone(c.stage)
        self.assertEqual(c.plans, 0)

    def test_invalid_preview_is_checked_only_when_proposal_index_is_needed(self):
        c = FeedbackController(fixed_reaction=True)
        bad = observation(10, next_orientation=19)
        with patch('nes_original.feedback.proposals', side_effect=AssertionError('Must not index invalid preview')):
            self.assertEqual(c.command(bad), 0)
            with self.assertRaises(ControllerObservationError):
                c.command(observation(11, next_orientation=19))
        # An existing valid target does not consume the preview index.
        c = FeedbackController(fixed_reaction=True)
        c.start_plan(observation(), None, 0)
        c.elapsed = 3
        self.assertEqual(c.command(bad), 32)

    def test_invalid_locked_pose_reports_contract_error_not_cells_indexerror(self):
        c = FeedbackController()
        before = observation()
        c.start_plan(before, None, 0)
        with self.assertRaisesRegex(ControllerObservationError, 'locked orientation'):
            c.transition(before, observation(2, phase=2, orientation=19))

    def test_mock_evaluate_persists_and_verifies_controller_observation_failure(self):
        from nes_original.learn import Config, evaluate
        board = np.zeros((20, 10), dtype=bool)
        board[-1, :4] = True
        states = [observation(2, phase=2), observation(3, phase=5, orientation=19,
            placements=1, board=board), observation(4, orientation=19, placements=1, board=board),
            observation(5, orientation=19, placements=1, board=board)]
        games = []
        class MockGame:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def reset(self, **kwargs):
                self.current = observation()
                self.index = 0
                return self.current
            def step(self, button):
                self.current = states[self.index]
                self.index += 1
                return self.current
            def observe(self): return self.current
            def replay(self):
                return dict(mock_only=True, result=dict(frames=self.current.frames,
                    level=self.current.level, lines=self.current.lines))
            def load_replay(self, record):
                self.loaded = record
        def factory():
            game = MockGame()
            games.append(game)
            return game
        base = Path.cwd()/'.pytest-work/feedback-observation'
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as folder:
            rows = evaluate(None, folder, seeds=[33], config=Config(tap_period=2),
                context_selector=lambda obs, plan: 0, game_factory=factory)
            row = rows[0]
            self.assertEqual(row['termination_kind'], 'controller_error')
            self.assertIn('Invalid active orientation persisted', row['error'])
            self.assertIsNone(row['observed_fault'])
            self.assertFalse(row['fault_reproduced'])
            self.assertFalse(row['success'])
            self.assertTrue(row['exact_ram_rgb_replay'])
            case = json.loads((Path(folder)/'seed-33-case.json').read_text())
            self.assertEqual(case['case_state'], 'verified')
            self.assertEqual(case['error'], row['error'])
            record = json.loads((Path(folder)/'seed-33.json').read_text())
            self.assertEqual(record['result']['frames'], 5)
            self.assertEqual(games[1].loaded, record)


if __name__ == '__main__':
    unittest.main()
