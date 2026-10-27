"""Regression tests for the fast simulator and the crash-zone simulator (need numpy + numba).

Two kinds of checks:

* Exact: seeded games, v8c crash-zone runs and 1G reach limits must match the values
  recorded in tests/data/sim_golden.json for the current spawn reaction wait. Any change
  to piece movement, the evaluator, the randomizer or the crash model shows up here.
  After an intended change, check the survival tests below, then re-record with
  `python tools/sim_golden.py --write`.
* Survival: invariants and pass-rate floors that must hold across intended changes too
  (faster reaction may only widen reach; 1G Dellacherie play stays alive; the v8c bot
  keeps its crash-zone pass rate). The full 1,000-game v8c baseline is slow and runs only
  with SIM_FULL_BASELINE=1.
"""
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
try:
    from nes_original import sim
    import sim_golden as golden
except ImportError:  # numba is optional; see requirements-nes-search.txt
    sim = golden = None

FULL = os.environ.get('SIM_FULL_BASELINE') == '1'


def _zone_pass(job):
    zone, seed = job
    return golden.zone_game(zone, seed)['result'] == 'pass'


def zone_passes(zone, seeds, workers=None):
    jobs = [(zone, s) for s in seeds]
    workers = workers or min(4, os.cpu_count() or 1)
    if workers == 1:
        return sum(map(_zone_pass, jobs))
    with mp.get_context('spawn').Pool(workers) as pool:
        return sum(pool.map(_zone_pass, jobs, chunksize=1))


@unittest.skipIf(sim is None, 'numba not installed')
class RecordedOutcomeTest(unittest.TestCase):
    """Exact values recorded for the current reaction wait."""

    @classmethod
    def setUpClass(cls):
        cls.key = f'reaction_{golden.reaction()}'
        cls.data = golden.load().get(cls.key)

    def recorded(self, part):
        if not self.data or part not in self.data:
            self.fail(f'No recorded {part} for {self.key}. If the change is intended and the survival tests pass, '
                      'run: python tools/sim_golden.py --write')
        return self.data[part]

    def test_seeded_games_match_recording(self):
        recorded = self.recorded('games')
        self.assertEqual(set(recorded), set(golden.GAMES))
        for name in golden.GAMES:
            with self.subTest(game=name):
                self.assertEqual(golden.play_named(name), recorded[name])

    def test_v8c_zone_games_match_recording(self):
        recorded = self.recorded('zone_games')
        for zone, seed in golden.ZONE_GAMES:
            with self.subTest(zone=zone, seed=seed):
                self.assertEqual(golden.zone_game(zone, seed), recorded[f'{zone}-{seed}'])

    def test_crash_model_matches_recording(self):
        self.assertEqual(golden.crash_fingerprint(), self.recorded('crash'))

    def test_well_reach_limits_match_recording(self):
        recorded = self.recorded('reach')
        for level in golden.REACH_LEVELS:
            with self.subTest(level=level):
                self.assertEqual(golden.reach_table(level), recorded[str(level)])

    @unittest.skipUnless(sim is not None and golden.reaction_argument(), 'sim has a fixed reaction wait')
    def test_every_recorded_reaction_reach_matches(self):
        for key, values in golden.load().items():
            frames = int(key.split('_')[1])
            for level in golden.REACH_LEVELS:
                with self.subTest(reaction=frames, level=level):
                    self.assertEqual(golden.reach_table(level, frames), values['reach'][str(level)])


@unittest.skipIf(sim is None, 'numba not installed')
class ReachInvariantTest(unittest.TestCase):
    """Movement physics that hold for any reaction wait."""

    def test_empty_board_every_piece_reaches_every_column(self):
        board = np.zeros((20, 10), np.bool_)
        for level in (18, 19, 29, 39, 255):
            for piece, orientation in golden.PIECES.items():
                with self.subTest(level=level, piece=piece):
                    self.assertEqual(golden.placed_columns(board, orientation, min(level, 29)), set(range(10)))

    def test_faster_gravity_never_widens_reach(self):
        tables = [golden.reach_table(level) for level in (18, 19, 29)]
        for slow, fast in zip(tables, tables[1:]):
            for piece in golden.PIECES:
                self.assertTrue(all(a >= b for a, b in zip(slow[piece], fast[piece])), piece)

    def test_1g_edges_are_hardest_and_i_reaches_furthest(self):
        table = golden.reach_table(29)
        for piece, limits in table.items():
            with self.subTest(piece=piece):
                self.assertEqual(min(limits), min(limits[0], limits[9]))
                self.assertTrue(all(i >= p for i, p in zip(table['I'], limits)))
                # Left side climbs towards spawn (x=5), right side falls away from it.
                self.assertEqual(limits[:4], sorted(limits[:4]))
                self.assertEqual(limits[6:], sorted(limits[6:], reverse=True))

    def test_shorter_reaction_wait_never_loses_reach(self):
        """Removing spawn wait frames may only raise 1G reach, by at most one row per frame."""
        current = golden.reaction()
        recorded = golden.load()
        for other in range(4):
            key = f'reaction_{other}'
            if other == current or key not in recorded or 'reach' not in recorded[key]:
                continue
            for level in golden.REACH_LEVELS:
                ours = golden.reach_table(level)
                theirs = recorded[key]['reach'][str(level)]
                for piece in golden.PIECES:
                    with self.subTest(reaction=other, level=level, piece=piece):
                        for a, b in zip(ours[piece], theirs[piece]):
                            if current < other:
                                self.assertGreaterEqual(a, b)
                            else:
                                self.assertLessEqual(a, b)
                            self.assertLessEqual(abs(a-b), abs(current-other))

    def test_1g_edge_reach_floor(self):
        """A 1G well in column 0 or 9 must stay usable over at least 8 rows (reaction 3 gives 8-11)."""
        table = golden.reach_table(29)
        for piece, limits in table.items():
            with self.subTest(piece=piece):
                self.assertGreaterEqual(min(limits[0], limits[9]), 8)


@unittest.skipIf(sim is None, 'numba not installed')
class SurvivalFloorTest(unittest.TestCase):
    """Pass-rate floors that intended bot changes must keep (survival first)."""

    def test_1g_dellacherie_stays_alive(self):
        # v8c-style Dellacherie beam 5 at 1G dies about 1 in 200 games over 2,700 lines.
        deaths = lines = 0
        for seed in range(30):
            game = sim.play_game(sim.DELLACHERIE, sim.DELLACHERIE, well=-1, well_survive=-1, seed=100+seed,
                                 start_level=29, max_lines=400, beam=5)
            deaths += game['dead']
            lines += game['lines']
        self.assertLessEqual(deaths, 1)
        self.assertGreaterEqual(lines, 29*400)

    def test_v8c_early_zone_quick_floor(self):
        # Baseline 95.7%; 23 of 24 here. A floor of 21 only catches large breaks.
        self.assertGreaterEqual(zone_passes('early', range(1000, 1024)), 21)

    @unittest.skipUnless(FULL, 'set SIM_FULL_BASELINE=1 for the 1,000-game v8c baseline (several minutes)')
    def test_v8c_full_baseline(self):
        # v8c (reaction 3): early 150-175 seeds 1000-1599 pass 574/600, late 246-255 seeds 2000-2399 pass 396/400.
        early = zone_passes('early', range(1000, 1600))
        late = zone_passes('late', range(2000, 2400))
        print(f'\nv8c baseline: early {early}/600, late {late}/400')
        self.assertGreaterEqual(early, 570)   # 95%
        self.assertGreaterEqual(late, 390)    # 97.5%


if __name__ == '__main__':
    unittest.main()
