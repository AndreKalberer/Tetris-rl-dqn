"""Reference bridge tests; real-core tests opt in via NES_MESEN_EXE.

Watchdog unit cases use standalone arrays, never injected game/emulator RAM.
No emulator, ROM or portable settings are installed/modified by these tests.
"""
from dataclasses import fields
import os
from pathlib import Path
import time
import unittest
from unittest.mock import Mock

import numpy as np

from nes_original.env import Buttons, ROM_SHA256
from nes_original.mesen import MesenFault, MesenGame, MesenObservation, PROTOCOL, SuspectedCrash


class WatchdogTests(unittest.TestCase):
    def game(self):
        game = MesenGame.__new__(MesenGame)
        game._ram = np.zeros(2048,dtype=np.uint8)
        game._ram[0xC0],game._ram[0x68] = 4,1
        game.started = True
        game.watchdog_frames = 60
        game._nmi_stalled = game._semantic_stalled = 0
        game._fault_record = Mock()
        game.session_dir = 'unit-array-only'
        return game

    def test_frozen_nmi_is_fault_not_topout(self):
        game = self.game()
        for _ in range(59): game._watchdog(game._ram.copy())
        with self.assertRaises(SuspectedCrash): game._watchdog(game._ram.copy())
        game._fault_record.assert_called_once_with('rom_nmi_counter_stalled')
        self.assertFalse(game.over)

    def test_nmi_running_with_frozen_game_is_distinct_fault(self):
        game = self.game()
        for i in range(60):
            before = game._ram.copy()
            game._ram[0xB1] = i+1
            if i<59: game._watchdog(before)
            else:
                with self.assertRaises(SuspectedCrash): game._watchdog(before)
        game._fault_record.assert_called_once_with('gameplay_state_stalled')

    def test_genuine_phase_ten_is_normal_over_and_other_mode_exit_is_fault(self):
        game = self.game()
        game._ram[0x68] = 10
        game._nmi_stalled = 99
        game._watchdog(game._ram.copy())
        self.assertTrue(game.over)
        game._ram[0x68],game._ram[0xC0] = 1,1
        with self.assertRaises(SuspectedCrash): game._watchdog(game._ram.copy())
        game._fault_record.assert_called_once_with('unexpected_exit_from_gameplay')

    def test_observation_exposes_pipeline_without_hidden_diagnostics(self):
        names = {f.name for f in fields(MesenObservation)}
        self.assertIn('input_pipeline',names)
        self.assertFalse(names & {'ram','cpu','rng','seed','future_pieces'})

    def test_reject_replay_core_mismatch_before_starting_process(self):
        game = self.game()
        game.identity = {'core':'expected'}
        game.rom_sha256 = ROM_SHA256
        with self.assertRaises(ValueError):
            game.load_replay(dict(protocol=PROTOCOL,rom_sha256=ROM_SHA256,identity={'core':'other'}))


@unittest.skipUnless(os.name=='nt' and os.environ.get('NES_MESEN_EXE'),
    'Set NES_MESEN_EXE to an initialized dedicated portable MesenCE build')
class ReferenceCoreTests(unittest.TestCase):
    def game(self, **kwargs):
        return MesenGame(executable=os.environ['NES_MESEN_EXE'],transport='pipe',
            session_root=Path(os.environ.get('NES_MESEN_TEST_ROOT','runs/mesen-test-sessions')),
            process_timeout=kwargs.pop('process_timeout',120),**kwargs)

    def test_legal_menu_immediate_input_and_fresh_process_exact_replay(self):
        with self.game() as game:
            obs = game.reset(start_level=18,seed=0)
            self.assertEqual((obs.level,obs.lines,obs.frames,obs.input_pipeline),(18,0,1,0))
            self.assertFalse(obs.board.flags.writeable)
            self.assertEqual(game.step(Buttons.RIGHT).x,obs.x+1)
            game.step(0)
            self.assertEqual(game.step(Buttons.LEFT).x,obs.x)
            game.step(0)
            for i in range(176): game.step(Buttons.DOWN if i%40<15 else 0)
            expected = game.replay()
            self.assertEqual(expected['result']['frames'],181)
            self.assertFalse(expected['result']['over'])
            game.load_replay(expected)
            self.assertEqual(game.replay(),expected)

    def test_host_idle_timeout_kills_only_owned_child(self):
        with self.game(action_timeout=2) as game:
            game.reset()
            process = game._process
            deadline = time.monotonic()+4
            while process.poll() is None and time.monotonic()<deadline: time.sleep(.05)
            self.assertIsNotNone(process.poll())
            self.assertTrue(game._idle_expired)
            with self.assertRaises((MesenFault,OSError)): game.step(0)

    def test_pipe_disconnect_unblocks_lua_and_exits_cleanly(self):
        with self.game() as game:
            game.reset()
            process = game._process
            game._pipes.close()
            game._pipes = None
            process.wait(timeout=3)
            self.assertEqual(process.returncode,2)
            error = (game.session_dir/'error.json').read_text()
            self.assertIn('host_pipe_closed',error)

    def test_process_wall_deadline_holds_even_with_active_commands(self):
        with self.game(action_timeout=2,process_timeout=3) as game:
            try:
                # The total process budget includes menu frames and startup;
                # under CPU contention it may expire before reset completes.
                game.reset()
                while True: game.step(0)
            except MesenFault:
                self.assertTrue(game._process_expired)
                self.assertFalse(game._idle_expired)
                self.assertGreater(len(game.inputs),0)
                game._process.wait(timeout=2)


if __name__=='__main__': unittest.main()
