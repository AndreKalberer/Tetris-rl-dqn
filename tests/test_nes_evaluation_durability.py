"""Actual original-game evaluations preserve per-case evidence before replay."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('nes_py') and importlib.util.find_spec('torch'),
                     'Requires isolated original-game learning environment')
class EvaluationDurabilityTests(unittest.TestCase):
    def setUp(self):
        base = Path.cwd()/'.pytest-work/evaluation-durability'
        base.mkdir(parents=True,exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)

    def read(self,name):
        return json.loads((self.output/name).read_text())

    def test_verified_cases_publish_stats_before_replay_and_results_after_each_case(self):
        from nes_original.env import OriginalGame
        from nes_original.learn import evaluate
        test = self
        class InspectingGame(OriginalGame):
            def load_replay(self,record,**kwargs):
                seed = record['menu_wait_seed']
                pending = test.read(f'seed-{seed}-case.json')
                test.assertEqual(pending['case_state'],'verification_pending')
                test.assertFalse(pending['exact_ram_rgb_replay'])
                test.assertFalse(pending['success'])
                test.assertGreater(pending['controller']['plans'],0)
                test.assertEqual(test.read(f'seed-{seed}.json'),record)
                if seed == 7:
                    first = test.read('results.json')
                    test.assertEqual([r['seed'] for r in first],[3])
                    test.assertTrue(first[0]['exact_ram_rgb_replay'])
                return super().load_replay(record,**kwargs)
        rows = evaluate(None,self.output,seeds=[3,7],max_frames=90,game_factory=InspectingGame)
        self.assertEqual(self.read('results.json'),rows)
        for row in rows:
            self.assertEqual(row['case_state'],'verified')
            self.assertTrue(row['exact_ram_rgb_replay'])
            self.assertEqual(self.read(f'seed-{row["seed"]}-case.json'),row)

    def test_verifier_failure_keeps_input_journal_controller_stats_and_prior_results(self):
        from nes_original.env import OriginalGame
        from nes_original.learn import evaluate
        class FailingVerifier(OriginalGame):
            def load_replay(self,record,**kwargs):
                if record['menu_wait_seed'] == 7:
                    raise RuntimeError('forced fresh replay failure')
                return super().load_replay(record,**kwargs)
        with self.assertRaisesRegex(RuntimeError,'forced fresh replay failure'):
            evaluate(None,self.output,seeds=[3,7],max_frames=90,game_factory=FailingVerifier)
        failed = self.read('seed-7-case.json')
        self.assertEqual(failed['case_state'],'verification_failed')
        self.assertFalse(failed['exact_ram_rgb_replay'])
        self.assertFalse(failed['success'])
        self.assertIsNone(failed['error'])
        self.assertIn('forced fresh replay failure',failed['verification_error'])
        self.assertGreater(failed['controller']['plans'],0)
        self.assertGreater(len(self.read('seed-7.json')['input_hex']),0)
        self.assertEqual([r['seed'] for r in self.read('results.json')],[3])

    def test_gameplay_error_and_fresh_observation_survive_verifier_failure(self):
        from nes_original.env import OriginalGame
        from nes_original.learn import evaluate
        class FailingVerifier(OriginalGame):
            def load_replay(self,record,**kwargs):
                raise RuntimeError('independent verifier error')
        with patch('nes_original.learn.FeedbackController.transition',
                   side_effect=RuntimeError('original transition error')):
            with self.assertRaisesRegex(RuntimeError,'independent verifier error'):
                evaluate(None,self.output,seeds=[3],max_frames=90,game_factory=FailingVerifier)
        failed = self.read('seed-3-case.json')
        self.assertEqual(failed['error'],'original transition error')
        self.assertIn('independent verifier error',failed['verification_error'])
        self.assertEqual(failed['case_state'],'verification_failed')
        self.assertFalse(failed['success'])
        self.assertFalse(failed['exact_ram_rgb_replay'])
        self.assertEqual(failed['controller']['plans'],1)
        self.assertEqual(self.read('seed-3.json')['result']['frames'],2)
        self.assertFalse((self.output/'results.json').exists())

    def test_capture_failure_keeps_pending_telemetry_and_raw_input_fallback(self):
        from nes_original.env import OriginalGame
        from nes_original.learn import evaluate
        class FailingCapture(OriginalGame):
            def replay(self):
                raise RuntimeError('forced capture failure')
        with self.assertRaisesRegex(RuntimeError,'forced capture failure'):
            evaluate(None,self.output,seeds=[3],max_frames=90,game_factory=FailingCapture)
        failed = self.read('seed-3-case.json')
        self.assertEqual(failed['case_state'],'verification_failed')
        self.assertEqual(failed['verification_failure_phase'],'replay_capture')
        self.assertGreater(failed['controller']['plans'],0)
        self.assertFalse(failed['exact_ram_rgb_replay'])
        raw = self.read('seed-3-raw-inputs.json')
        self.assertGreater(len(raw['input_hex']),0)
        self.assertFalse(raw['exact_replay_verified'])


if __name__ == '__main__':
    unittest.main()
