"""Opt-in actual-reference-core learning, evaluation parking and exact resume."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(os.name=='nt' and os.environ.get('NES_MESEN_EXE') and
                     importlib.util.find_spec('torch') and importlib.util.find_spec('nes_py'),
                     'Requires isolated training dependencies and an existing portable MesenCE')
class ReferenceLearningTests(unittest.TestCase):
    def test_optimizer_replay_resume_and_evaluation_preserve_training_state(self):
        import torch
        from nes_original.learn import Config,Trainer,load_policy
        from nes_original.mesen import MesenGame
        base=Path.cwd()/'.pytest-work/reference-learning'
        base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temporary:
            folder=Path(temporary)
            factory=lambda:MesenGame(executable=os.environ['NES_MESEN_EXE'],
                session_root=folder/'sessions',process_timeout=180)
            config=Config(seed=2,capacity=32,batch_size=4,learning_starts=4,
                teacher_decisions=2,train_every=1,target_every=3,tap_period=2)
            a=Trainer(config,game_factory=factory)
            b=None
            try:
                for _ in range(6): a.advance()
                before=a.game.replay()
                rows=a.evaluate(a.online,folder/'bounded-validation',seeds=[3],max_frames=60)
                self.assertEqual(before,a.game.replay())
                self.assertTrue(rows[0]['exact_ram_rgb_replay'])
                self.assertEqual(rows[0]['backend'],'MesenGame')
                self.assertTrue(rows[0]['censored'])
                a.save(folder/'resume.pt')
                a.export(folder/'policy.pt')
                _,metadata=load_policy(folder/'policy.pt')
                self.assertEqual(metadata['training_backend'],'MesenGame')
                for _ in range(3): a.advance()
                expected=a.game.replay()
                weights=[p.detach().clone() for p in a.online.parameters()]
                a.close()
                b=Trainer.load(folder/'resume.pt',game_factory=factory)
                for _ in range(3): b.advance()
                self.assertEqual(expected,b.game.replay())
                self.assertEqual((a.step,a.updates,a.loss),(b.step,b.updates,b.loss))
                self.assertTrue(all(torch.equal(x,y) for x,y in zip(weights,b.online.parameters())))
                self.assertEqual(b.audit_totals['board_audit_failures'],0)
                b.close()
                # Full trainer continuation cannot silently switch core; weights
                # alone may be evaluated in another backend as a separate trial.
                with self.assertRaises(ValueError): Trainer.load(folder/'resume.pt')
            finally:
                a.close()
                if b: b.close()


if __name__=='__main__': unittest.main()
