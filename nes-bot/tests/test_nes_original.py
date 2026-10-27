"""Run with unittest or pytest in the separate original-game environment."""
from dataclasses import fields
import importlib.util
import json
import unittest

import numpy as np

from nes_original.env import Buttons, OriginalGame, Observation, bcd, from_game_bits


class EncodingTests(unittest.TestCase):
    def test_explicit_button_order_and_rejected_menu_inputs(self):
        for game, native in ((1,128),(2,64),(4,32),(64,2),(128,1),(133,161)):
            self.assertEqual(from_game_bits(game), native)
        for invalid in (-1, 8, 16, 32, 256):
            with self.assertRaises(ValueError):
                from_game_bits(invalid)
        self.assertEqual(bcd([0x56,0x34,0x12]), 123456)
        with self.assertRaises(ValueError):
            bcd([0xAF])

    def test_agent_interface_excludes_rng_and_future_pieces(self):
        names = {field.name for field in fields(Observation)}
        self.assertFalse(names & {'ram','rng','seed','cpu','future_pieces','snapshot'})
        self.assertIn('next_orientation', names)


@unittest.skipUnless(importlib.util.find_spec('nes_py'), 'Use requirements-nes-original.txt')
class EmulatorTests(unittest.TestCase):
    def test_genuine_menu_starts_and_deterministic_reset(self):
        with OriginalGame() as game:
            for level in (0,9,10,18,19):
                obs = game.reset(start_level=level, seed=17)
                self.assertEqual(obs.level, level)
                self.assertEqual((obs.lines,obs.score,obs.placements,obs.frames), (0,0,0,1))
                self.assertFalse(obs.board.any())
                self.assertFalse(obs.board.flags.writeable)
                first = game.replay()
                game.reset(start_level=level, seed=17)
                self.assertEqual(first, game.replay())

    def test_native_snapshot_and_portable_replay_continue_exactly(self):
        with OriginalGame() as game, OriginalGame() as other:
            game.reset(start_level=18, seed=23)
            rng = np.random.default_rng(19)
            inputs = [int(value) for value in rng.choice([0,1,2,32,64,128,129,66], 240)]
            for action in inputs[:120]:
                game.step(action)
            snap = game.snapshot()
            before = game.replay()
            for action in inputs[120:]:
                game.step(action)
            expected = game.replay()
            game.restore(snap)
            self.assertEqual(game.replay(), before)
            for action in inputs[120:]:
                game.step(action)
            self.assertEqual(game.replay(), expected)
            other.load_replay(json.loads(json.dumps(expected)))
            for action in inputs[:20]:
                game.step(action)
                other.step(action)
            self.assertEqual(game.replay(), other.replay())

    def test_frame_clock_top_out_and_lock_count(self):
        with OriginalGame() as game:
            game.reset(start_level=19, seed=3)
            previous = int.from_bytes(bytes(game._env.ram[0xB1:0xB3]), 'little')
            for frame in range(1500):
                obs = game.step(Buttons.DOWN)
                current = int.from_bytes(bytes(game._env.ram[0xB1:0xB3]), 'little')
                self.assertEqual((current-previous) & 65535, 1)
                previous = current
                if obs.over:
                    break
            self.assertTrue(obs.over)
            self.assertGreater(obs.placements, 3)
            # Game statistics count the failed spawned piece as well.
            r = game._env.ram
            spawned = sum(bcd(r[address:address+2]) for address in range(0x3F0,0x3FE,2))
            self.assertEqual(spawned, obs.placements+1)
            self.assertEqual(obs.lines, 0)
            with self.assertRaises(RuntimeError):
                game.step(0)
            with OriginalGame() as other:
                other.load_replay(game.replay())
                self.assertTrue(other.over)

    def test_no_ram_writes_or_gym_tetris_reset_helpers(self):
        # Writable dependency RAM is deliberately made read-only at the adapter
        # boundary; native emulator code can still evolve the real memory.
        from unittest.mock import patch
        import gym_tetris.tetris_env
        with patch.object(gym_tetris.tetris_env.TetrisEnv, '_did_reset',
                          side_effect=AssertionError('Upstream reset hook invoked')):
            with OriginalGame() as game:
                game.reset(start_level=18, seed=3)
                self.assertFalse(game._env.ram.flags.writeable)
                for _ in range(120):
                    game.step(Buttons.DOWN)
                self.assertGreater(game.observe().frames, 0)

    def test_active_controls_include_native_input_pipeline(self):
        from nes.engine import Control, control_frame
        from nes_original.planner import control
        checked = 0
        with OriginalGame() as game:
            for level, seed in ((0,5),(18,17),(19,19)):
                game.reset(start_level=level,seed=seed)
                rng = np.random.default_rng(8)
                for _ in range(900):
                    before = game.observe()
                    if before.phase == 1:
                        expected, failed = control_frame(before.board,control(before),before.pending,before.level)
                    action = int(rng.choice([0,1,2,32,64,128,129,66]))
                    after = game.step(action)
                    if before.phase == 1:
                        checked += 1
                        self.assertEqual(control(after),expected)
                        self.assertEqual(after.phase,2 if failed else 1)
                    if after.over:
                        break
        self.assertGreater(checked,1500)

    def test_all_proposed_paths_match_original_game_at_three_speeds(self):
        from nes_original.planner import proposals, execute
        checked = 0
        with OriginalGame() as game:
            for level in (0,18,19):
                game.reset(start_level=level,seed=23)
                snapshot = game.snapshot()
                plan = proposals(game.observe())
                self.assertGreater(len(plan.paths),8)
                for selected in range(len(plan.paths)):
                    obs = execute(game,plan,selected)
                    self.assertEqual(obs.placements,1)
                    checked += 1
                    game.restore(snapshot)
                    self.assertEqual(game.snapshot().inputs,snapshot.inputs)
        self.assertGreater(checked,30)

    def test_runtime_handles_observed_late_game_main_loop_sampling(self):
        # Regression: at frame 6370 of this original-game trajectory, strict
        # predicted lock timing stopped a live game at 102 placements/40 lines.
        # Actual Down + observed lock must continue, with exact replay intact.
        from nes_original.run import play
        result = play(level=18,seed=17,max_placements=105,fixed_reaction=True)
        self.assertIsNone(result['error'])
        self.assertEqual(result['placements'],105)
        self.assertEqual(result['stopped_by'],'placement_limit')
        self.assertTrue(result['exact_ram_rgb_replay'])
        self.assertEqual(result['controller']['path_mismatches'],0)
        self.assertGreater(result['controller']['drop_timing_adjustments'],0)

    def test_runtime_discards_late_plan_and_advances_real_game(self):
        from unittest.mock import patch
        from nes_original.run import Controller
        with OriginalGame() as game:
            game.reset(start_level=18,seed=17)
            controller = Controller()
            frame = game.observe().frames
            with patch('nes_original.run.time.perf_counter',side_effect=[0.0,0.2]):
                action = controller.command(game.observe())
            self.assertEqual(action,0)
            self.assertFalse(controller.queue)
            self.assertIsNone(controller.expected)
            game.step(action)
            while controller.idle:
                game.step(controller.command(game.observe()))
            self.assertEqual(controller.extra_idle_frames,13)
            self.assertEqual(game.observe().frames,frame+13)
            self.assertEqual(bytes(game.inputs[-13:]),bytes(13))


if __name__ == '__main__':
    unittest.main()
