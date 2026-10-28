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

    def test_first_piece_reuse_and_threads_keep_the_reference_result(self):
        # Reference: every sequence placed from scratch, serially (the original loop).
        rng = np.random.default_rng(4)
        board = np.zeros((20, 10), np.bool_)
        board[14:, :9] = rng.random((6, 9)) < .8
        w = np.asarray(sim.DELLACHERIE, float)
        state, failed = sim.spawn_state(2, 0, board, 29)
        states = np.zeros((sim.MAX_CANDIDATES, 9), np.int64)
        n = sim.candidates(board, state, 29, 2, states, 0)
        roots = np.zeros((n, 20, 10), np.bool_); part = np.zeros(n); das = np.zeros(n, np.int64)
        scratch = np.zeros(sim.N_FEATURES)
        for i in range(n):
            roots[i], dead, _, part[i] = sim._scored_lock(board, states[i, 0], states[i, 1], states[i, 2], 9, w, scratch)
            if dead:
                part[i] = sim.DEAD
            das[i] = states[i, 4]
        chosen_root, chosen_pose, reward = sim.playout_rank(roots, part, das, 7, 29, 2, 0, 9, w, 6, 2)
        seqs, prob = sim.playout_sequences(sim.PIECE_TYPE[7], 2)
        deep = np.zeros((sim.MAX_CANDIDATES, 9), np.int64)
        for e in range(len(chosen_root)):
            start, _, _ = sim.lock(roots[chosen_root[e]], *chosen_pose[e, :3])
            total = 0.
            for s in range(len(seqs)):
                b, d, reward_so_far, value = start, chosen_pose[e, 3], 0., -1000.
                for j in range(2):
                    b, d, dead, v, p = sim._greedy(b, seqs[s, j], d, 29, 2, 0, 9, w, deep, scratch)
                    if dead:
                        value = -1000.+reward_so_far
                        break
                    if j == 1:
                        value = reward_so_far+v
                    else:
                        reward_so_far += p
                total += prob[s]*value
            fast = sim.playout_expand(roots, chosen_root, chosen_pose, 7, 29, 2, 0, 9, w, 2, -1000., e, e+1)[0]
            self.assertAlmostEqual(fast, total, places=9)

    def test_weights_file_turns_on_playouts(self):
        import json, tempfile, os
        data = dict(features=list(sim.FEATURES), alt_weights=list(sim.DELLACHERIE), alt_below=8,
                    burn_weights=list(sim.DELLACHERIE), playout_top=6, playout_cap=2)
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
            json.dump(data, f)
        try:
            selector = sim.tuned_selector('dellacherie', alt_weights=f.name, reaction=0)
        finally:
            os.unlink(f.name)
        self.assertEqual((selector.playout_top, selector.playout_cap, selector.playout_length), (6, 2, 2))


if __name__ == '__main__':
    unittest.main()
