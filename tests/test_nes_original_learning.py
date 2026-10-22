"""Actual-emulator learning and full original-game checkpoint regressions."""
import importlib.util
import tempfile
from pathlib import Path
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('nes_py') and importlib.util.find_spec('torch'),
                     'Requires isolated original-game learning environment')
class OriginalLearningTests(unittest.TestCase):
    def test_cross_piece_rotation_is_rejected_instead_of_spinning_forever(self):
        from nes.actions import rotate_button
        with self.assertRaises(ValueError):
            rotate_button(8,11)
    def test_original_game_checkpoint_continues_optimizer_rng_replay_and_inputs(self):
        import torch
        from nes_original.learn import Config,Trainer,load_policy
        c = Config(seed=2,batch_size=8,capacity=64,learning_starts=8,
                   teacher_decisions=3,target_every=5,train_every=1)
        a = Trainer(c)
        b = None
        try:
            for _ in range(12):
                a.advance()
            base = (Path.cwd()/'.pytest-work/original-learning').resolve()
            self.assertTrue(base.is_relative_to(Path.cwd().resolve()))
            base.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(dir=base) as temp:
                checkpoint = Path(temp)/'resume.pt'
                a.save(checkpoint)
                for _ in range(3):
                    a.advance()
                expected = a.game.replay()
                weights = [p.detach().clone() for p in a.online.parameters()]
                b = Trainer.load(checkpoint)
                for _ in range(3):
                    b.advance()
                self.assertEqual(b.game.replay(),expected)
                self.assertEqual(b.step,a.step)
                self.assertEqual(b.updates,a.updates)
                self.assertEqual(b.loss,a.loss)
                self.assertEqual(b.rng.bit_generator.state,a.rng.bit_generator.state)
                self.assertTrue(all(torch.equal(x,y) for x,y in zip(weights,b.online.parameters())))
                self.assertEqual(len(a.replay),len(b.replay))
                for x,y in zip(a.replay,b.replay):
                    self.assertTrue(np.array_equal(x[0],y[0]))
                    self.assertEqual((x[1],x[3]),(y[1],y[3]))
                    self.assertTrue(np.array_equal(x[2],y[2]))
                policy = Path(temp)/'policy.pt'
                b.export(policy)
                model,metadata = load_policy(policy)
                self.assertEqual(metadata['decisions'],15)
                self.assertTrue(all(torch.equal(x,y) for x,y in zip(weights,model.parameters())))
        finally:
            a.close()
            if b:
                b.close()

    def test_actual_emulator_reward_and_neutral_prior_initialization(self):
        from nes_original.learn import Config,Trainer,features,values,reward
        t = Trainer(Config(seed=3))
        try:
            x = features(t.plan)
            self.assertEqual(x.shape[1],288)
            self.assertTrue(np.array_equal(values(t.online,x),x[:,-1]))
            before = t.game.observe()
            from nes_original.learn import execute
            after,_ = execute(t.game,t.plan,int(x[:,-1].argmax()))
            self.assertGreater(after.placements,before.placements)
            self.assertGreater(after.frames,before.frames)
            self.assertTrue(np.isfinite(reward(before,after)))
            # Actual locked board differs from the empty start; no simulator
            # Game.tick is involved in advancing this episode.
            self.assertEqual(int(after.board.sum()),4)
        finally:
            t.close()


if __name__ == '__main__':
    unittest.main()
