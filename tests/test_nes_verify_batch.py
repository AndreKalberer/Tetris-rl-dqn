"""Parallel proof checks must still reject incorrect expected game states."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

from nes_original.verify_batch import verify_many


class BatchLimitsTests(unittest.TestCase):
    def test_overlapping_or_empty_worker_cpu_assignments_are_rejected(self):
        for masks in ((1,1),(0,256)):
            with self.assertRaises(ValueError): verify_many(['unused'],workers=2,masks=masks)
        with self.assertRaises(ValueError): verify_many([],workers=2)
        with self.assertRaises(ValueError): verify_many(['unused'],workers=5,masks=(1,2,4,8,16))


@unittest.skipUnless(os.name=='nt' and (os.cpu_count() or 0)>=16 and importlib.util.find_spec('nes_py'),
    'Actual emulator and the workstation spare CPU assignments required')
class ActualBatchTests(unittest.TestCase):
    def test_parallel_fresh_core_checks_pass_and_corrupted_proof_fails(self):
        from nes_original.env import OriginalGame
        base=Path.cwd()/'.pytest-work/batch-proofs'
        base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temporary:
            folder=Path(temporary)
            with OriginalGame() as game:
                game.reset(start_level=18,seed=3)
                for _ in range(120): game.step(0)
                record=game.replay()
            good=folder/'good.json'
            good.write_text(json.dumps(record))
            bad=folder/'bad.json'
            # Change only the expected proof, never emulator RAM or controller input.
            record['result']['score']+=1
            bad.write_text(json.dumps(record))
            result=verify_many([good,bad],workers=2,masks=(1,256))
            self.assertFalse(result['verified'])
            self.assertEqual([row['verified'] for row in result['rows']],[True,False])
            self.assertIn('Replay diverged',result['rows'][1]['error'])
            self.assertNotEqual(result['rows'][0]['pid'],result['rows'][1]['pid'])
            self.assertEqual({row['cpu_mask'] for row in result['rows']},{1,256})


if __name__=='__main__': unittest.main()
