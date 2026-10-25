"""Reaction-frame setting of the feedback controller (pure controller checks)."""
from types import SimpleNamespace
import unittest

from nes_original.env import Buttons
from nes_original.feedback import FeedbackController


def observation(x=5):
    return SimpleNamespace(phase=1, x=x, y=1, orientation=2, pending=0, previous=0, level=29, soft=0, fall=0)


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


if __name__ == '__main__':
    unittest.main()
