"""Evaluator exception routing fixtures; not evidence of a native-game crash."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('nes_py') and importlib.util.find_spec('torch'),
                     'Requires isolated original-game dependencies')
class FaultFlowTests(unittest.TestCase):
    def test_expected_native_fault_keeps_error_and_continues_next_declared_case(self):
        from nes_original.learn import evaluate
        from nes_original.mesen import SuspectedCrash
        expected = dict(type='SuspectedCrash',reason='rom_nmi_counter_stalled')
        checked = []
        class Controller:
            def __init__(self,**kwargs): self.stage = None
            def command(self,obs): return 0
            def transition(self,before,after): pass
            def stats(self): return dict(fixture=True)
        class Game:
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def reset(self,**kwargs):
                self.seed = kwargs['seed']
                return self.observe()
            def observe(self):
                return SimpleNamespace(level=18,lines=0,placements=0,frames=2,
                    score=0,over=self.seed==7,phase=1)
            def step(self,buttons):
                raise SuspectedCrash('rom_nmi_counter_stalled; diagnostics=fixture')
            def replay(self): return dict(seed=self.seed,fixture=True)
            def load_replay(self,record,**kwargs): checked.append((record,kwargs))
        base = Path.cwd()/'.pytest-work/fault-flow'
        base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temporary, patch('nes_original.learn.FeedbackController',Controller):
            rows = evaluate(None,temporary,seeds=[3,7],game_factory=Game,target_level=255)
            self.assertEqual(len(rows),2)
            self.assertTrue(rows[0]['fault_reproduced'])
            self.assertEqual(rows[0]['observed_fault'],expected)
            self.assertFalse(rows[0]['success'])
            self.assertFalse(rows[0]['censored'])
            self.assertEqual(rows[0]['termination_kind'],'suspected_original_crash')
            self.assertIn('rom_nmi_counter_stalled',rows[0]['error'])
            self.assertEqual(checked[0][1],dict(expected_fault=expected))
            self.assertEqual(checked[0][0]['observed_fault'],expected)
            self.assertEqual(checked[1][1],{})
            self.assertEqual(rows[1]['termination_kind'],'top_out')
            self.assertFalse(rows[1]['fault_reproduced'])
            self.assertEqual(json.loads((Path(temporary)/'results.json').read_text()),rows)
