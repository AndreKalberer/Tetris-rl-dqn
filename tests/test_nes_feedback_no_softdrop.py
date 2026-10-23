"""Pure controller/model checks; no native state injection or gameplay claims."""
from types import SimpleNamespace
import unittest

import numpy as np

from nes.engine import Buttons as ModelButtons, Control, SPAWN, control_frame
from nes_original.env import Buttons
from nes_original.feedback import FeedbackController


def observation(level=29, soft=0, fall=0):
    return SimpleNamespace(phase=1, x=5, y=2, orientation=2, pending=0,
        previous=0, level=level, soft=soft, fall=fall)


def prepared(obs, flag=False, route='dropping'):
    controller = FeedbackController(fixed_reaction=True, tap_period=2,
        no_softdrop_at_1g=flag)
    plan = None if route == 'dropping' else SimpleNamespace(actions=SimpleNamespace(
        controls=[SimpleNamespace(x=obs.x, orientation=obs.orientation)]))
    controller.start_plan(obs, plan, 0)
    controller.elapsed = 51 if route == 'navigation_limit' else 3
    return controller


class NoSoftDropTests(unittest.TestCase):
    def test_opt_in_boundaries_preserve_defaults_and_initial_negative_timer(self):
        for route in ('dropping', 'navigation_limit', 'target_reached'):
            for level in (0, 18, 28, 29, 138, 255):
                for soft in (-96, 0, 1, 2, 127, 128, 160, 255):
                    with self.subTest(route=route, level=level, soft=soft):
                        obs = observation(level, soft)
                        self.assertEqual(prepared(obs, False, route).command(obs), int(Buttons.DOWN))
                        expected = 0 if level >= 29 and 0 <= soft < 128 else int(Buttons.DOWN)
                        self.assertEqual(prepared(obs, True, route).command(obs), expected)
        self.assertFalse(FeedbackController().stats()['no_softdrop_at_1g'])
        self.assertTrue(FeedbackController(no_softdrop_at_1g=True).stats()['no_softdrop_at_1g'])

    def test_navigation_and_reaction_commands_are_unchanged(self):
        obs = observation()
        for frame in range(12):
            controllers = []
            for flag in (False, True):
                c = FeedbackController(fixed_reaction=True, tap_period=2, no_softdrop_at_1g=flag)
                plan = SimpleNamespace(actions=SimpleNamespace(controls=[SimpleNamespace(x=0, orientation=1)]))
                c.start_plan(obs, plan, 0)
                c.elapsed = frame
                controllers.append(c)
            self.assertEqual(controllers[0].command(obs), controllers[1].command(obs))

    def test_ordinary_one_g_drop_has_same_geometry_but_different_soft_points(self):
        board = np.zeros((20, 10), dtype=bool)
        for level in (29, 138, 255):
            for pose in SPAWN:
                for soft in (0, 1, 2):
                    held = Control(y=2, orientation=pose, soft=soft, previous=4 if soft else 0)
                    released = held
                    for _ in range(25):
                        held, held_lock = control_frame(board, held, int(ModelButtons.DOWN), level)
                        released, released_lock = control_frame(board, released, 0, level)
                        self.assertEqual((held.x, held.y, held.orientation, held_lock),
                            (released.x, released.y, released.orientation, released_lock))
                        if held_lock:
                            break
                    self.assertTrue(held_lock)
                    self.assertGreater(held.soft_points, 0)
                    self.assertEqual(released.soft_points, 0)

    def test_initial_timer_and_slower_gravity_are_not_motion_equivalent(self):
        board = np.zeros((20, 10), dtype=bool)
        initial = Control(y=2, soft=0xA0)
        down, _ = control_frame(board, initial, int(ModelButtons.DOWN), 29)
        idle, _ = control_frame(board, initial, 0, 29)
        self.assertEqual((down.y, idle.y), (3, 2))
        self.assertEqual(prepared(observation(29, 0xA0), True).command(observation(29, 0xA0)), int(Buttons.DOWN))
        slow = Control(y=2, soft=2, previous=4)
        down, _ = control_frame(board, slow, int(ModelButtons.DOWN), 28)
        idle, _ = control_frame(board, slow, 0, 28)
        self.assertEqual((down.y, idle.y), (3, 2))
        self.assertEqual(prepared(observation(28, 2), True).command(observation(28, 2)), int(Buttons.DOWN))

    def test_one_g_byte_wrap_is_not_a_universal_geometry_guarantee(self):
        # A non-steady fall byte can wrap before the normal1G comparison.
        # Forced soft-drop can still move in that frame, idle cannot. This
        # constructed model case is not an observed native late-game state.
        board = np.zeros((20, 10), dtype=bool)
        unusual = Control(y=2, fall=255, soft=2, previous=4)
        down, _ = control_frame(board, unusual, int(ModelButtons.DOWN), 29)
        idle, _ = control_frame(board, unusual, 0, 29)
        self.assertEqual((down.y, idle.y), (3, 2))
        for fall in (-1, 128, 255):
            for route in ('dropping', 'navigation_limit', 'target_reached'):
                obs = observation(29, 2, fall)
                self.assertEqual(prepared(obs, True, route).command(obs), int(Buttons.DOWN))

    def test_entire_guarded_fall_range_preserves_one_g_geometry(self):
        board = np.zeros((20, 10), dtype=bool)
        for level in (29, 138, 255):
            for pose in SPAWN:
                for soft in (0, 1, 2):
                    for fall in range(128):
                        state = Control(y=2, orientation=pose, soft=soft, fall=fall,
                            previous=4 if soft else 0)
                        down, down_lock = control_frame(board, state, int(ModelButtons.DOWN), level)
                        idle, idle_lock = control_frame(board, state, 0, level)
                        self.assertEqual((down.x, down.y, down.orientation, down_lock),
                            (idle.x, idle.y, idle.orientation, idle_lock))
                        self.assertEqual(idle.fall, 0)
        # In particular127 increments to128; (128-1)&255 ==127 still passes
        # the original signed comparison, so natural gravity moves that frame.
        state = Control(y=2, soft=0, fall=127)
        idle, _ = control_frame(board, state, 0, 29)
        self.assertEqual((idle.y, idle.fall), (3, 0))
        obs = observation(29, 0, 127)
        self.assertEqual(prepared(obs, True).command(obs), 0)


if __name__ == '__main__':
    unittest.main()
