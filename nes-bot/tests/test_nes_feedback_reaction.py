"""Reaction-frame setting of the feedback controller (pure controller checks)."""
from types import SimpleNamespace
import unittest

from nes_original.env import Buttons
from nes_original.feedback import FeedbackController


def observation(x=5):
    return SimpleNamespace(phase=1, x=x, y=1, orientation=2, pending=0, previous=0, level=29, soft=0, fall=0,
                           next_orientation=2, frames=0)


def started(reaction, target_x=0):
    controller = FeedbackController(fixed_reaction=True, tap_period=2, reaction_frames=reaction)
    obs = observation()
    plan = SimpleNamespace(actions=SimpleNamespace(controls=[SimpleNamespace(x=target_x, orientation=2)]))
    controller.start_plan(obs, plan, 0)
    return controller, obs


class ReactionFrameTests(unittest.TestCase):
    def test_first_input_comes_after_the_configured_neutral_frames(self):
        for reaction in (0, 1, 3):
            with self.subTest(reaction=reaction):
                controller, obs = started(reaction)
                buttons = [controller.command(obs) for _ in range(reaction+1)]
                self.assertEqual(buttons[:reaction], [0]*reaction)
                self.assertEqual(buttons[reaction], int(Buttons.LEFT))

    def test_default_and_reporting_and_rules(self):
        self.assertEqual(FeedbackController().reaction_frames, 3)
        controller = FeedbackController(reaction_frames=0)
        self.assertEqual(controller.stats()['reaction_frames'], 0)
        self.assertEqual(controller.rules.reaction_frames, 0)

    def test_invalid_values_are_rejected(self):
        for invalid in (-1, 11, 1.0, True, '0', None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                FeedbackController(reaction_frames=invalid)

    def _late(self, late_plan, seconds):
        from unittest.mock import patch
        controller = FeedbackController(tap_period=2, reaction_frames=0, late_plan=late_plan)
        obs = observation()
        plan = SimpleNamespace(actions=SimpleNamespace(controls=[SimpleNamespace(x=0, orientation=2)]))
        with patch('nes_original.feedback.proposals', return_value=plan), \
                patch('nes_original.feedback.time.perf_counter', side_effect=[0.0, seconds]):
            controller.selector = lambda p: 0
            first = controller.command(obs)
        return controller, obs, first

    def test_late_plan_keep_starts_the_plan_after_the_planning_time(self):
        controller, obs, first = self._late('keep', 0.045)   # 2.7 frames of planning
        self.assertEqual(first, 0)
        self.assertEqual(controller.stage, 'active')
        self.assertEqual(controller.overruns, 1)
        self.assertEqual(controller.command(obs), 0)          # second planning frame
        self.assertEqual(controller.command(obs), int(Buttons.LEFT))
        self.assertEqual(controller.extra_idle_frames, 2)
        self.assertEqual(controller.plans, 1)

    def test_late_plan_replan_discards_the_plan(self):
        controller, obs, first = self._late('replan', 0.045)
        self.assertEqual(first, 0)
        self.assertIsNone(controller.stage)
        self.assertEqual(controller.idle, 2)

    def _retarget(self, fresh_boards):
        import numpy as np
        from unittest.mock import patch
        controller = FeedbackController(tap_period=2, reaction_frames=0, late_plan='retarget')
        obs = observation()
        target = np.zeros((20, 10), np.uint8); target[19, 0] = 1
        other = np.zeros((20, 10), np.uint8); other[19, 9] = 1
        plan = SimpleNamespace(actions=SimpleNamespace(controls=[SimpleNamespace(x=0, orientation=2)],
                                                       boards=[target]))
        boards = {'target': target, 'other': other}
        fresh = SimpleNamespace(actions=SimpleNamespace(
            controls=[SimpleNamespace(x=9 if name == 'other' else 1, orientation=2) for name in fresh_boards],
            boards=[boards[name] for name in fresh_boards]))
        picks = []
        controller.selector = lambda p: picks.append(p) or 0
        with patch('nes_original.feedback.proposals', side_effect=[plan, fresh]), \
                patch('nes_original.feedback.time.perf_counter', side_effect=[0.0, 0.020, 0.0, 0.002]):
            controller.command(obs)                           # late plan: one idle frame
            controller.command(obs)                           # first input frame: check
        return controller, picks

    def test_late_plan_retarget_keeps_a_reachable_board(self):
        controller, picks = self._retarget(['other', 'target'])
        self.assertEqual(controller.late_checks, 1)
        self.assertEqual(controller.late_replans, 0)
        self.assertEqual(controller.selected, 1)
        self.assertEqual(controller.target_x, 1)
        self.assertEqual(len(picks), 1)

    def test_late_plan_retarget_replans_when_the_board_is_gone(self):
        controller, picks = self._retarget(['other'])
        self.assertEqual(controller.late_replans, 1)
        self.assertEqual(controller.target_x, 9)
        self.assertEqual(len(picks), 2)

    def test_late_plan_value_is_checked(self):
        with self.assertRaises(ValueError):
            FeedbackController(late_plan='wait')
        self.assertEqual(FeedbackController().late_plan, 'replan')


if __name__ == '__main__':
    unittest.main()
