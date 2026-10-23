"""Public-state command routing; no emulator or native recovery claims."""
from types import SimpleNamespace
import unittest

import numpy as np

from nes_original.feedback import FeedbackController


def observation(frame=0, **changes):
    values = dict(frames=frame, phase=5, orientation=19, x=5, y=18,
        level=166, lines=1609, placements=4030, score=999999,
        board=np.zeros((20, 10), dtype=bool), pending=0, previous=0, soft=0, fall=0)
    values.update(changes)
    return SimpleNamespace(**values)


class StallRescueTests(unittest.TestCase):
    def test_threshold_uses_actual_frames_and_duplicate_calls_do_not_count(self):
        c = FeedbackController(stall_rescue_input=128, stall_rescue_after=120)
        self.assertEqual(c.command(observation(100)), 0)
        for _ in range(200):
            self.assertEqual(c.command(observation(100)), 0)
        self.assertEqual(c.command(observation(219)), 0)
        self.assertEqual(c.command(observation(220)), 128)
        for _ in range(200):
            self.assertEqual(c.command(observation(220)), 128)
        self.assertEqual((c.stall_rescue_attempts, c.stall_rescue_frames), (1, 1))
        self.assertEqual(c.command(observation(221)), 128)
        self.assertEqual((c.stall_rescue_attempts, c.stall_rescue_frames), (1, 2))

    def test_every_public_key_change_restarts_the_window(self):
        changes = dict(phase=6, orientation=18, x=6, y=17, level=167,
                       lines=1610, placements=4031, score=999998)
        changed_board = np.zeros((20, 10), dtype=bool)
        changed_board[19, 0] = True
        changes['board'] = changed_board
        for name, value in changes.items():
            with self.subTest(field=name):
                c = FeedbackController(stall_rescue_input=32, stall_rescue_after=3)
                self.assertEqual(c.command(observation(0)), 0)
                self.assertEqual(c.command(observation(3)), 32)
                self.assertEqual(c.command(observation(4, **{name: value})), 0)
                self.assertEqual(c.command(observation(6, **{name: value})), 0)
                self.assertEqual(c.command(observation(7, **{name: value})), 32)
                self.assertEqual((c.stall_rescue_attempts, c.stall_rescue_frames), (2, 2))

    def test_active_phase_resets_without_replacing_active_idle_or_settle(self):
        c = FeedbackController(nonactive_input=2, stall_rescue_input=128, stall_rescue_after=2)
        self.assertEqual(c.command(observation(0)), 2)
        self.assertEqual(c.command(observation(2)), 128)
        c.stage = 'settle'
        self.assertEqual(c.command(observation(3, phase=1)), 0)
        self.assertEqual(c.command(observation(4)), 2)
        self.assertEqual(c.command(observation(5)), 2)
        c.idle = 2
        self.assertEqual(c.command(observation(6)), 128)
        self.assertEqual(c.command(observation(7, phase=1)), 0)
        self.assertEqual((c.idle, c.extra_idle_frames), (0, 2))
        self.assertEqual(c.command(observation(8)), 2)

    def test_invalid_or_backward_frame_discards_accumulated_window(self):
        for invalid in (None, True, -1, 3.0, '3', 1):
            with self.subTest(frame=invalid):
                c = FeedbackController(stall_rescue_input=64, stall_rescue_after=3)
                c.command(observation(10))
                c.command(observation(13))
                self.assertEqual(c.command(observation(invalid)), 0)
                self.assertEqual(c.command(observation(20)), 0)
                self.assertEqual(c.command(observation(22)), 0)
                self.assertEqual(c.command(observation(23)), 64)
                self.assertEqual(c.stall_rescue_attempts, 2)

    def test_default_disabled_uses_only_existing_phase_routing(self):
        c = FeedbackController(nonactive_input=32)
        for _ in range(150):
            self.assertEqual(c.command(SimpleNamespace(phase=5)), 32)
        self.assertEqual((c.stall_rescue_attempts, c.stall_rescue_frames), (0, 0))
        self.assertEqual(c.stats()['stall_rescue_input'], 0)
        self.assertEqual(c.stats()['stall_rescue_after'], 120)

    def test_options_are_strict_legal_buttons_and_subwatchdog_thresholds(self):
        for button in (0, 1, 2, 32, 64, 128):
            for threshold in (1, 120, 599):
                c = FeedbackController(stall_rescue_input=button, stall_rescue_after=threshold)
                self.assertEqual(c.stats()['stall_rescue_input'], button)
                self.assertEqual(c.stats()['stall_rescue_after'], threshold)
        for invalid in (True, False, -1, 3, 4, 8, 16, 96, 129, 256, 32.0, '32', None):
            with self.subTest(button=invalid), self.assertRaises(ValueError):
                FeedbackController(stall_rescue_input=invalid)
        for invalid in (True, False, 0, -1, 600, 120.0, '120', None):
            with self.subTest(threshold=invalid), self.assertRaises(ValueError):
                FeedbackController(stall_rescue_after=invalid)

    def test_active_reaction_navigation_and_drop_match_disabled_profile(self):
        for elapsed in (0, 2, 3, 4, 51):
            for target in (0, 5):
                choices = []
                for enabled in (0, 128):
                    c = FeedbackController(fixed_reaction=True, tap_period=2, stall_rescue_input=enabled)
                    obs = observation(10000, phase=1, orientation=2, y=2)
                    plan = SimpleNamespace(actions=SimpleNamespace(controls=[SimpleNamespace(x=target, orientation=2)]))
                    c.start_plan(obs, plan, 0)
                    c.elapsed = elapsed
                    choices.append(c.command(obs))
                    self.assertEqual((c.stall_rescue_attempts, c.stall_rescue_frames), (0, 0))
                self.assertEqual(choices[0], choices[1])


if __name__ == '__main__':
    unittest.main()
