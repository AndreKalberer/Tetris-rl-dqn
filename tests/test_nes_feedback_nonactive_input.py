"""Pure routing checks; no native level166 recovery or crash claim."""
from types import SimpleNamespace
import unittest

from nes_original.env import Buttons
from nes_original.feedback import FeedbackController


def active_observation():
    return SimpleNamespace(phase=1, x=5, y=2, orientation=2, pending=0,
        previous=0, level=29, soft=0, fall=0)


def controller_at(nonactive_input, stage, elapsed=3, dropping=False, target_x=5):
    obs = active_observation()
    c = FeedbackController(fixed_reaction=True, tap_period=2, nonactive_input=nonactive_input)
    plan = SimpleNamespace(actions=SimpleNamespace(controls=[SimpleNamespace(x=target_x, orientation=2)]))
    c.start_plan(obs, plan, 0)
    c.stage, c.elapsed, c.dropping = stage, elapsed, dropping
    return c, obs


class NonactiveInputTests(unittest.TestCase):
    def test_only_single_legal_native_buttons_are_accepted(self):
        for value in (0, 1, 2, 32, 64, 128, Buttons.DOWN):
            c = FeedbackController(nonactive_input=value)
            self.assertEqual(c.stats()['nonactive_input'], int(value))
        for invalid in (True, False, -1, 3, 4, 8, 16, 31, 33, 96, 129, 255, 256,
                        32.0, '32', None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                FeedbackController(nonactive_input=invalid)

    def test_nonactive_phases_use_configured_byte_without_active_pose_reads(self):
        for button in (0, 1, 2, 32, 64, 128):
            c = FeedbackController(nonactive_input=button)
            calls = 0
            for stage in (None, 'active', 'settle', 'entry'):
                c.stage = stage
                for phase in (0, 2, 3, 4, 5, 6, 7, 8, 9, 10):
                    self.assertEqual(c.command(SimpleNamespace(phase=phase)), button)
                    calls += 1
                    self.assertEqual(c.stats()['nonactive_input_frames'], calls)

    def test_active_settle_reaction_navigation_and_drop_are_unchanged(self):
        cases = [('settle', 3, False, 5), ('active', 0, False, 5),
                 ('active', 2, False, 0), ('active', 3, False, 0),
                 ('active', 4, False, 0), ('active', 3, False, 5),
                 ('active', 3, True, 5), ('active', 51, False, 0)]
        for stage, elapsed, dropping, target_x in cases:
            reference, obs = controller_at(0, stage, elapsed, dropping, target_x)
            expected = reference.command(obs)
            for button in (1, 2, 32, 64, 128):
                c, obs = controller_at(button, stage, elapsed, dropping, target_x)
                self.assertEqual(c.command(obs), expected)
                self.assertEqual(c.stats()['nonactive_input_frames'], 0)
                self.assertEqual(c.elapsed, reference.elapsed)
                self.assertEqual(c.dropping, reference.dropping)

    def test_latency_idle_preserves_accounting_and_active_neutral_input(self):
        c = FeedbackController(nonactive_input=128)
        c.idle = 3
        self.assertEqual(c.command(SimpleNamespace(phase=2)), 128)
        self.assertEqual(c.command(SimpleNamespace(phase=1)), 0)
        self.assertEqual(c.command(SimpleNamespace(phase=5)), 128)
        self.assertEqual((c.idle, c.extra_idle_frames, c.nonactive_input_frames), (0, 3, 2))
        self.assertEqual(c.command(SimpleNamespace(phase=6)), 128)
        self.assertEqual(c.nonactive_input_frames, 3)

    def test_default_nonactive_commands_remain_neutral(self):
        c = FeedbackController()
        self.assertEqual(c.stats()['nonactive_input'], 0)
        self.assertEqual(c.stats()['nonactive_input_frames'], 0)
        for phase in (2, 3, 5, 6, 8, 10):
            self.assertEqual(c.command(SimpleNamespace(phase=phase)), 0)
        self.assertEqual(c.stats()['nonactive_input_frames'], 6)


if __name__ == '__main__':
    unittest.main()
