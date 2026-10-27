import unittest

import numpy as np

try:
    from nes_original import sim
except ImportError:  # numba missing
    sim = None


@unittest.skipIf(sim is None, 'numba not installed')
class PlayoutTests(unittest.TestCase):
    def test_sequence_odds_sum_to_one(self):
        for last in range(7):
            seqs, prob = sim.playout_sequences(last, 2)
            self.assertEqual(seqs.shape, (49, 2))
            self.assertAlmostEqual(float(prob.sum()), 1.0, places=9)

    def test_skipped_roots_stay_dead(self):
        roots = np.zeros((2, 20, 10), np.bool_)
        part = np.asarray([0., sim.DEAD])
        das = np.zeros(2, np.int64)
        scores = sim.playout_scores(roots, part, das, 2, 29, 2, 0, 9, sim.DELLACHERIE, 4, 2, 1, -1000.)
        self.assertGreater(scores[0], sim.DEAD)
        self.assertEqual(scores[1], sim.DEAD)

    def test_play_game_with_playouts(self):
        out = sim.play_game(sim.DELLACHERIE, sim.DELLACHERIE, seed=3, start_level=29, initial_score=sim.MAXED,
                            max_lines=20, beam=3, well=-1, well_survive=-1, alt_weights=sim.DELLACHERIE, alt_below=7,
                            reaction=0, playout_top=2, playout_cap=1, playout_length=1)
        self.assertTrue(out['dead'] or out['lines'] >= 20)

    def test_off_by_default_matches_beam(self):
        args = dict(seed=5, start_level=29, initial_score=sim.MAXED, max_lines=30, beam=3, well=-1, well_survive=-1,
                    alt_weights=sim.DELLACHERIE, alt_below=7, reaction=0)
        self.assertEqual(sim.play_game(sim.DELLACHERIE, sim.DELLACHERIE, **args),
                         sim.play_game(sim.DELLACHERIE, sim.DELLACHERIE, playout_top=0, **args))


if __name__ == '__main__':
    unittest.main()
