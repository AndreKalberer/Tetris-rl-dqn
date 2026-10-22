"""Parallel logical evaluation must reproduce the serial policy's native inputs."""
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

from nes_original.parallel_evaluate import evaluate_parallel


class ParallelLimitsTests(unittest.TestCase):
    def test_duplicate_games_and_overlapping_cpu_assignments_fail_before_start(self):
        with self.assertRaises(ValueError):
            evaluate_parallel(None,'unused',seeds=[3,3],config={})
        with self.assertRaises(ValueError):
            evaluate_parallel(None,'unused',seeds=[3,7],config={},masks=[1,1])


@unittest.skipUnless(os.name=='nt' and (os.cpu_count() or 0)>=16 and
    importlib.util.find_spec('nes_py') and importlib.util.find_spec('torch'),
    'Requires the isolated native training dependencies and spare CPU assignments')
class NativeParallelTests(unittest.TestCase):
    def test_serial_and_parallel_loaded_policy_have_identical_native_proofs(self):
        self.check_policy()

    @unittest.skipUnless(os.environ.get('NES_MESEN_EXE'),'Opt in with an existing dedicated portable reference build')
    def test_reference_core_parallel_loaded_policy_has_identical_serial_proofs(self):
        self.check_policy(os.environ['NES_MESEN_EXE'])

    def check_policy(self,mesen=None):
        import torch
        from nes_original.env import ROM_SHA256
        from nes_original.learn import Config,KIND,Network,evaluate,load_policy
        config=Config(seed=7,tap_period=2)
        torch.manual_seed(7)
        model=Network(288)
        with torch.no_grad(): model.residual[-1].weight.fill_(.001)
        base=Path.cwd()/'.pytest-work/parallel-evaluation'
        base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temporary:
            folder=Path(temporary)
            policy=folder/'policy.pt'
            torch.save(dict(kind=KIND,rom_sha256=ROM_SHA256,dim=288,config=asdict(config),
                online=model.state_dict(),torch_version=str(torch.__version__),decisions=0,updates=0),policy)
            loaded,_=load_policy(policy)
            from nes_original.env import OriginalGame
            factory=OriginalGame
            if mesen:
                from nes_original.mesen import MesenGame
                factory=lambda:MesenGame(executable=mesen,session_root=folder/'serial-sessions')
            serial=evaluate(loaded,folder/'serial',seeds=[3,7],runtime=False,max_frames=120,game_factory=factory)
            parallel=evaluate_parallel(policy,folder/'parallel',seeds=[3,7],config=asdict(config),
                runtime=False,max_frames=120,workers=2,masks=(1,256),mesen=mesen)
            self.assertEqual([r['seed'] for r in parallel],[3,7])
            for left,right in zip(serial,parallel):
                self.assertTrue(left['exact_ram_rgb_replay'] and right['exact_ram_rgb_replay'])
                a=json.loads((folder/f'serial/seed-{left["seed"]}.json').read_text())
                b=json.loads((folder/'parallel'/right['proof']).read_text())
                self.assertEqual(a,b)
                self.assertIsNone(right['error'])
            self.assertEqual({r['worker_cpu_mask'] for r in parallel},{1,256})
            self.assertTrue((folder/'parallel/parallel-protocol.json').is_file())


if __name__=='__main__': unittest.main()
