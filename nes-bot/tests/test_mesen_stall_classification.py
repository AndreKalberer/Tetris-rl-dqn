"""Opt-in semantic-stall classification and typed observation faults.

Standalone arrays/mocks only: no emulator process, and no game RAM is written.
"""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from nes_original.env import ObservationFault, bcd
from nes_original.feedback import ControllerObservationError
from nes_original.mesen import MesenGame, SuspectedCrash


def fake_rom(folder):
    prg = bytearray(32768)
    prg[0x8123-0x8000] = 0x4C  # JMP abs, a plausible ROM loop opcode
    path = Path(folder)/'fixture.nes'
    path.write_bytes(b'NES\x1a'+bytes([2,2,0x10])+bytes(9)+bytes(prg)+bytes(16384))
    return path


class StallClassificationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def game(self, *, grace=None, pc=0x8123):
        game = MesenGame.__new__(MesenGame)
        game._ram = np.zeros(2048,dtype=np.uint8)
        game._ram[0xC0],game._ram[0x68] = 4,1
        game.started = True
        game.watchdog_frames = 60
        game._nmi_stalled = game._semantic_stalled = 0
        game._fault_record = Mock()
        game.session_dir = 'unit-array-only'
        game.inputs, game.start_frame = bytearray(), 0
        game._request = {'pc':pc}
        game.rom_path, game._prg = fake_rom(self.folder.name), None
        game.stall_events = []
        if grace is not None: game.stall_grace_frames = grace
        return game

    def run_frames(self, game, frames, *, nmi=True):
        """Frozen gameplay RAM; optionally advancing NMI counter. Returns frames run."""
        for i in range(frames):
            before = game._ram.copy()
            if nmi:
                counter = (int(game._ram[0xB1])+1) & 0xFF
                game._ram[0xB1] = counter
            game.inputs.append(0)
            try: game._watchdog(before)
            except SuspectedCrash: return i+1
        return None

    def test_default_semantic_stall_still_faults_at_watchdog(self):
        game = self.game()
        self.assertEqual(self.run_frames(game,200),60)
        game._fault_record.assert_called_once_with('gameplay_state_stalled')

    def test_recoverable_loop_gets_bounded_grace_then_faults_with_diagnostics(self):
        game = self.game(grace=30)
        self.assertEqual(self.run_frames(game,200),90)
        reason,stall = game._fault_record.call_args.args
        self.assertEqual(reason,'gameplay_state_stalled')
        self.assertEqual((stall['classification'],stall['pc'],stall['pc_region'],stall['opcode']),
                         ('recoverable_loop',0x8123,'rom',0x4C))
        self.assertTrue(stall['grace_exhausted'])
        self.assertEqual(len(game.stall_events),1)

    def test_rescue_within_grace_clears_stall_without_fault(self):
        game = self.game(grace=30)
        self.assertIsNone(self.run_frames(game,75))
        before = game._ram.copy()
        game._ram[0x60] += 1  # Standalone array: models a rescued piece moving.
        game._ram[0xB1] += 1
        game._watchdog(before)
        self.assertEqual(game._semantic_stalled,0)
        self.assertIsNone(self.run_frames(game,80))
        game._fault_record.assert_not_called()

    def test_ram_pc_is_hard_crash_immediately_with_ram_opcode(self):
        game = self.game(grace=600,pc=0x0041)
        game._ram[0x41] = 0x12  # JAM
        self.assertEqual(self.run_frames(game,200),60)
        reason,stall = game._fault_record.call_args.args
        self.assertEqual(reason,'gameplay_state_stalled')
        self.assertEqual((stall['classification'],stall['pc_region'],stall['opcode']),('hard_crash','ram',0x12))

    def test_stopped_nmi_is_hard_crash_even_with_rom_pc(self):
        game = self.game(grace=600)
        self.assertEqual(self.run_frames(game,200,nmi=False),60)
        reason,stall = game._fault_record.call_args.args
        self.assertEqual(reason,'rom_nmi_counter_stalled')
        self.assertEqual(stall['classification'],'hard_crash')
        self.assertFalse(stall['nmi_advancing'])

    def test_nmi_stopping_during_grace_ends_immediately(self):
        game = self.game(grace=300)
        self.assertIsNone(self.run_frames(game,70))
        self.assertEqual(self.run_frames(game,10,nmi=False),2)
        self.assertEqual(game._fault_record.call_args.args[1]['classification'],'hard_crash')

    def test_missing_pc_is_hard_crash(self):
        game = self.game(grace=300,pc=None)
        self.assertEqual(self.run_frames(game,100),60)
        stall = game._fault_record.call_args.args[1]
        self.assertEqual((stall['classification'],stall['pc_region'],stall['opcode']),('hard_crash','unknown',None))

    @unittest.skipUnless(os.name=='nt','Mesen launcher is Windows-only')
    def test_constructor_rejects_invalid_grace_before_process(self):
        for value in (0,-1,True,1.5,'30',36001):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'stall_grace_frames'):
                MesenGame(stall_grace_frames=value)


class ObservationFaultTests(unittest.TestCase):
    def test_invalid_bcd_is_typed_and_routes_as_runtime_and_value_error(self):
        with self.assertRaises(ObservationFault) as caught:
            bcd([0x12,0xAF])
        self.assertIsInstance(caught.exception,RuntimeError)
        self.assertIsInstance(caught.exception,ValueError)
        self.assertEqual(bcd([0x56,0x34,0x12]),123456)

    def test_orientation_decode_error_is_an_observation_fault(self):
        self.assertTrue(issubclass(ControllerObservationError,ObservationFault))
        self.assertTrue(issubclass(ControllerObservationError,RuntimeError))


if __name__ == '__main__':
    unittest.main()
