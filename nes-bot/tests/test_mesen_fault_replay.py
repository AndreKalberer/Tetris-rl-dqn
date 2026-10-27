"""Fault-replay verifier control-flow fixtures; no emulator RAM is modified."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

from nes_original.env import ROM_SHA256
from nes_original.mesen import MesenFault,MesenGame,PROTOCOL,SuspectedCrash


EXPECTED = {'type':'SuspectedCrash','reason':'rom_nmi_counter_stalled'}
INPUTS = bytes([0,128,0,64,0])


class ReplayFixture(MesenGame):
    """Standalone verifier fixture, not a native ROM or crash reproduction."""
    def __init__(self,*,fault_at=5,reason='rom_nmi_counter_stalled',fault_type=SuspectedCrash,
                 mutate_input=False):
        self.identity = {'fixture':'no-native-core'}
        self.rom_sha256 = ROM_SHA256
        self.fault_at,self.reason,self.fault_type = fault_at,reason,fault_type
        self.mutate_input = mutate_input
        self.reset_calls = 0
        self.expected_result = {'level':157,'lines':1512,'placements':7,'frames':3,'over':False}
        self.expected_fingerprint = {'ram_sha256':'ram','screen_sha256':'rgb',
                                     'clock':{'cpu_cycle':42,'video_frame':5}}

    def reset(self,**kwargs):
        self.reset_calls += 1
        self.inputs = bytearray(INPUTS[:2])
        self.start_frame = 2
        self.placements = 7

    def step(self,buttons):
        self.inputs.append(buttons if not self.mutate_input else buttons^1)
        if len(self.inputs)==self.fault_at:
            raise self.fault_type(self.reason+'; diagnostics=fresh-fixture-path')

    def replay(self):
        return {'fingerprint':self.expected_fingerprint,'result':self.expected_result,
                'input_hex':self.inputs.hex()}

    def observe(self): return SimpleNamespace(**self.expected_result)

    def record(self):
        return dict(protocol=PROTOCOL,rom_sha256=ROM_SHA256,identity=self.identity,
            start_level=18,menu_wait_seed=33,start_frame=2,input_hex=INPUTS.hex(),
            placements=7,fingerprint=deepcopy(self.expected_fingerprint),
            result=deepcopy(self.expected_result))


class ExpectedFaultReplayTests(unittest.TestCase):
    def test_explicit_final_matching_fault_requires_full_state_and_input_comparison(self):
        game = ReplayFixture()
        obs = game.load_replay(game.record(),expected_fault=EXPECTED)
        self.assertEqual(bytes(game.inputs),INPUTS)
        self.assertEqual((obs.level,obs.lines),(157,1512))
        self.assertFalse(obs.over)

    def test_default_preserves_raised_watchdog_and_normal_replay_behavior(self):
        game = ReplayFixture()
        with self.assertRaises(SuspectedCrash): game.load_replay(game.record())
        normal = ReplayFixture(fault_at=None)
        normal.load_replay(normal.record())

    def test_wrong_reason_class_or_earlier_fault_is_rejected(self):
        class OtherSuspectedCrash(SuspectedCrash): pass
        for kwargs in ({'reason':'gameplay_state_stalled'},{'fault_at':4},
                       {'fault_type':OtherSuspectedCrash},{'fault_type':MesenFault}):
            with self.subTest(kwargs=kwargs):
                game = ReplayFixture(**kwargs)
                with self.assertRaises((ValueError,MesenFault)):
                    game.load_replay(game.record(),expected_fault=EXPECTED)

    def test_missing_expected_fault_is_not_a_verified_fault_trace(self):
        game = ReplayFixture(fault_at=None)
        with self.assertRaisesRegex(ValueError,'did not occur'):
            game.load_replay(game.record(),expected_fault=EXPECTED)

    def test_ram_rgb_clock_result_journal_and_placement_mismatches_are_rejected(self):
        for field in ('ram_sha256','screen_sha256','clock','result','input_hex','placements'):
            with self.subTest(field=field):
                game = ReplayFixture(mutate_input=field=='input_hex')
                record = game.record()
                if field=='result': record['result']['lines'] += 1
                elif field=='placements': record['placements'] += 1
                elif field!='input_hex': record['fingerprint'][field] = 'wrong'
                with self.assertRaises(ValueError):
                    game.load_replay(record,expected_fault=EXPECTED)

    def test_invalid_expected_fault_contract_fails_before_reset(self):
        for expected in (True,'rom_nmi_counter_stalled',{},
                         {'type':'MesenFault','reason':'rom_nmi_counter_stalled'},
                         {'type':'SuspectedCrash','reason':'unknown'},
                         {'type':'SuspectedCrash','reason':[]},dict(EXPECTED,extra=True)):
            with self.subTest(expected=expected):
                game = ReplayFixture()
                with self.assertRaises(ValueError):
                    game.load_replay(game.record(),expected_fault=expected)
                self.assertEqual(game.reset_calls,0)

    def test_callback_watchdog_is_never_accepted_as_an_emulator_fault(self):
        game = ReplayFixture()
        def callback(_): raise SuspectedCrash('rom_nmi_counter_stalled')
        with self.assertRaises(SuspectedCrash):
            game.load_replay(game.record(),expected_fault=EXPECTED,callback=callback)


if __name__=='__main__': unittest.main()
