import json
import unittest
from pathlib import Path

import numpy as np

try:
    from nes_original import sim
except ImportError:  # numba missing
    sim = None

WEIGHTS = Path(__file__).resolve().parents[1]/'nes_original'/'weights'


def reference_net(board, W1, b1, W2, b2, W3, b3):
    """Plain numpy forward pass: 200 cells (row-major) + 10 column heights / 20, two ReLU layers."""
    cells = board.reshape(-1).astype(float)
    filled = board.any(0)
    heights = np.where(filled, 20-board.argmax(0), 0)/20.
    x = np.concatenate([cells, heights])
    h1 = np.maximum(W1 @ x+b1, 0.)
    h2 = np.maximum(W2 @ h1+b2, 0.)
    return float((W3 @ h2+b3)[0])


@unittest.skipIf(sim is None, 'numba not installed')
class ValueNetTests(unittest.TestCase):
    def test_numba_forward_pass_matches_numpy(self):
        net = sim.load_value_net(WEIGHTS/'value-net-v3.npz')
        rng = np.random.default_rng(4)
        for fill in (0., .2, .5):
            board = rng.random((20, 10)) < fill
            board[:5] = False
            self.assertAlmostEqual(sim.value_net(board, *net), reference_net(board, *net), places=9)

    def test_zero_weight_matches_plain_playouts(self):
        net = sim.load_value_net(WEIGHTS/'value-net-v3.npz')
        roots = np.zeros((3, 20, 10), np.bool_)
        roots[1, 19, :9] = True
        roots[2, 18:, :8] = True
        part = np.zeros(3)
        das = np.zeros(3, np.int64)
        w = sim.DELLACHERIE
        plain = sim.playout_scores(roots, part, das, 2, 29, 2, 0, 9, w, 4, 2, 2, -1000.)
        netted = sim.playout_scores_net(roots, part, das, 2, 29, 2, 0, 9, w, 4, 2, 2, -1000., 0., *net)
        np.testing.assert_allclose(netted, plain)

    def test_weights_file_turns_on_the_gated_net(self):
        path = WEIGHTS/'tetris-playout12-b8-vnet.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        selector = sim.tuned_selector('dellacherie', alt_weights=str(path), reaction=0)
        self.assertIsNotNone(selector.value_net)
        self.assertEqual(selector.value_lam, data['value_lam'])
        self.assertEqual(selector.value_levels, ((176, 243),))
        self.assertEqual(selector.value_max_height, 6)
        self.assertEqual(selector.playout_top, 12)


if __name__ == '__main__':
    unittest.main()
