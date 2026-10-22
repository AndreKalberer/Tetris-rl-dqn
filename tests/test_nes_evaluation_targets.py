"""Target255 evaluation plumbing; short native proofs do not establish a255 clear."""
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from unittest.mock import patch
from types import SimpleNamespace


@unittest.skipUnless(importlib.util.find_spec('nes_py') and importlib.util.find_spec('torch'),
                     'Requires isolated original-game learning environment')
class EvaluationTargetTests(unittest.TestCase):
    def setUp(self):
        base = Path.cwd()/'.pytest-work/evaluation-targets'
        base.mkdir(parents=True,exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)

    def test_invalid_targets_and_frame_limits_fail_before_starting_or_writing(self):
        from nes_original.learn import evaluate
        from nes_original.parallel_evaluate import evaluate_parallel
        factory = Mock()
        for target in (None,False,True,18,0,-1,256,30.0,'255'):
            with self.subTest(target=target):
                with self.assertRaisesRegex(ValueError,'target_level'):
                    evaluate(None,self.output/'invalid',target_level=target,game_factory=factory)
                with self.assertRaisesRegex(ValueError,'target_level'):
                    evaluate_parallel(None,self.output/'invalid',seeds=[3],config={},target_level=target)
        for maximum in (None,False,True,0,-1,90.0,float('inf')):
            with self.subTest(max_frames=maximum):
                with self.assertRaisesRegex(ValueError,'max_frames'):
                    evaluate(None,self.output/'invalid',max_frames=maximum,game_factory=factory)
                with self.assertRaisesRegex(ValueError,'max_frames'):
                    evaluate_parallel(None,self.output/'invalid',seeds=[3],config={},max_frames=maximum)
        factory.assert_not_called()
        self.assertFalse((self.output/'invalid').exists())

    def test_custom_selector_rejects_model_and_noncallable_before_core_start(self):
        from nes_original.learn import evaluate
        factory = Mock()
        with self.assertRaisesRegex(ValueError,'context_selector'):
            evaluate(Mock(),self.output/'invalid',game_factory=factory,
                     context_selector=lambda obs,plan:0)
        with self.assertRaisesRegex(ValueError,'context_selector'):
            evaluate(None,self.output/'invalid',game_factory=factory,context_selector=0)
        factory.assert_not_called()
        self.assertFalse((self.output/'invalid').exists())

    def test_custom_selector_receives_native_context_without_feature_encoder(self):
        from nes_original.learn import Config,evaluate
        def select(obs,plan):
            self.assertEqual(obs.start_level,18)
            self.assertGreater(len(plan.paths),0)
            return int(plan.actions.heuristic.argmax())
        selector = Mock(side_effect=select)
        with patch('nes_original.learn.FeatureEncoder',side_effect=AssertionError('encoder must be skipped')):
            row, = evaluate(None,self.output/'custom',seeds=[3],max_frames=90,
                target_level=255,config=Config(tap_period=2),context_selector=selector)
        selector.assert_called()
        self.assertTrue(row['exact_ram_rgb_replay'])
        self.assertEqual(row['target_level'],255)

    def test_target255_and_default30_preserve_the_same_short_native_inputs(self):
        from nes_original.learn import Config,evaluate
        outcomes = []
        for label,kwargs in (('default',{}),('explicit30',dict(target_level=30)),
                             ('target255',dict(target_level=255))):
            folder = self.output/label
            row, = evaluate(None,folder,seeds=[3],max_frames=90,config=Config(tap_period=2),**kwargs)
            outcomes.append((row,json.loads((folder/'seed-3.json').read_text())))
            self.assertEqual(row['start_level'],18)
            self.assertEqual(row['target_level'],kwargs.get('target_level',30))
            self.assertEqual(row['max_frames'],90)
            self.assertEqual(row['frames'],90)
            self.assertEqual(row['level'],18)
            self.assertEqual(row['peak_level'],18)
            self.assertFalse(row['target_reached'])
            self.assertFalse(row['success'])
            self.assertTrue(row['censored'])
            self.assertTrue(row['exact_ram_rgb_replay'])
            self.assertEqual(row['case_state'],'verified')
            self.assertIn('settled',row['target_criterion'])
            self.assertEqual(json.loads((folder/'seed-3-case.json').read_text()),row)
        self.assertEqual(outcomes[0][1],outcomes[1][1])
        self.assertEqual(outcomes[0][1],outcomes[2][1])

    def test_synthetic_counter_copy_cannot_certify_target_before_settled_boundary(self):
        # Pure evaluator control-flow fixture, not a native-game255 proof or a
        # RAM patch. Actual ROM/counter fidelity belongs to the native tests.
        from nes_original.learn import evaluate
        class SyntheticController:
            def __init__(self,**kwargs): self.stage = None
            def command(self,obs): return 0
            def transition(self,before,after): self.stage = after.stage
            def stats(self): return {'synthetic_control_flow_fixture':True}
        class SyntheticGame:
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def reset(self,**kwargs):
                self.index = 0
                return SimpleNamespace(level=18,frames=1,phase=1,stage=None,
                    lines=0,score=0,placements=0,over=False)
            def step(self,buttons):
                # Active action, then entry/counter-copy phase, then a corrupt
                # settled255. None of these observations are NES RAM.
                level,phase,stage = [(255,1,'active'),(18,1,None),
                    (255,3,None),(18,1,None),(255,1,None)][self.index]
                self.index += 1
                return SimpleNamespace(level=level,frames=self.index+1,phase=phase,stage=stage,
                    lines=0,score=0,placements=0,over=False)
            def replay(self): return {'synthetic_control_flow_fixture':True}
            def load_replay(self,record): pass
        with patch('nes_original.learn.FeedbackController',SyntheticController):
            short, = evaluate(None,self.output/'short-synthetic',seeds=[3],max_frames=5,
                              target_level=255,game_factory=SyntheticGame)
            settled, = evaluate(None,self.output/'settled-synthetic',seeds=[3],max_frames=6,
                                target_level=255,game_factory=SyntheticGame)
        self.assertFalse(short['target_reached'])
        self.assertFalse(short['success'])
        self.assertEqual(short['peak_level'],18)
        self.assertFalse(settled['target_reached'])
        self.assertFalse(settled['success'])
        self.assertEqual(settled['frames'],6)
        self.assertEqual(settled['peak_level'],18)
        self.assertIn('Progress counter audit failed',settled['error'])
        self.assertEqual(settled['progress']['raw_level'],18)

    def test_synthetic_legal_progression_reaches255_at_first_cycle_line3292(self):
        # Counter/control-flow fixture generated by the independently tested
        # original counter helper, not native gameplay or an emulator RAM edit.
        from nes_original.learn import evaluate
        from nes_original.progression import after_clear
        class SyntheticController:
            def __init__(self,**kwargs): self.stage = None
            def command(self,obs): return 0
            def transition(self,before,after): pass
            def stats(self): return {'synthetic_control_flow_fixture':True}
        class SyntheticGame:
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def reset(self,**kwargs):
                self.level,self.lines,self.placements,self.frames = 18,0,0,1
                return self.observe()
            def observe(self):
                return SimpleNamespace(level=self.level,lines=self.lines,placements=self.placements,
                    frames=self.frames,phase=1,over=False,score=0)
            def step(self,buttons):
                self.level,self.lines = after_clear(self.level,self.lines,4)
                self.placements += 1
                self.frames += 1
                return self.observe()
            def replay(self): return {'synthetic_control_flow_fixture':True}
            def load_replay(self,record): pass
        with patch('nes_original.learn.FeedbackController',SyntheticController):
            row, = evaluate(None,self.output/'legal-synthetic',seeds=[3],max_frames=1000,
                            target_level=255,game_factory=SyntheticGame)
        self.assertTrue(row['success'])
        self.assertTrue(row['target_reached'])
        self.assertEqual(row['progress']['raw_level'],255)
        self.assertEqual(row['progress']['absolute_lines'],3292)
        self.assertEqual(row['progress']['level_cycles'],0)
        self.assertIsNone(row['error'])

    @unittest.skipUnless(os.name=='nt' and (os.cpu_count() or 0)>=9,
                         'Requires isolated Windows CPU assignments0/8')
    def test_parallel_target255_matches_serial_native_proof_and_records_protocol(self):
        from nes_original.learn import Config,evaluate
        from nes_original.parallel_evaluate import evaluate_parallel
        config = Config(tap_period=2)
        row, = evaluate(None,self.output/'serial',seeds=[3],max_frames=90,
                        target_level=255,config=config)
        parallel, = evaluate_parallel(None,self.output/'parallel',seeds=[3],
            config=asdict(config),max_frames=90,target_level=255,
            runtime=False,workers=1,masks=(1,))
        self.assertEqual(parallel['target_level'],255)
        self.assertEqual(parallel['target_reached'],row['target_reached'])
        self.assertTrue(parallel['exact_ram_rgb_replay'])
        first = json.loads((self.output/'serial/seed-3.json').read_text())
        second = json.loads((self.output/'parallel'/parallel['proof']).read_text())
        self.assertEqual(first,second)
        protocol = json.loads((self.output/'parallel/parallel-protocol.json').read_text())
        self.assertEqual(protocol['target_level'],255)
        self.assertEqual(protocol['start_level'],18)
        self.assertEqual(protocol['max_frames'],90)


if __name__=='__main__':
    unittest.main()
