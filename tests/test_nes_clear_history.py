import unittest

from nes_original.clear_history import ClearHistory, encode_clear_events
from nes_original.progression import after_clear


class ClearHistoryTests(unittest.TestCase):
    def advance(self, tracker, cleared):
        level, lines, placements = tracker._boundary
        level, lines = after_clear(level, lines, cleared)
        return tracker.update(level, lines, placements+1)

    def new_game(self):
        tracker = ClearHistory()
        self.assertTrue(tracker.reset(18, 0, 0)['known'])
        return tracker

    def test_c_is_pre_event_units_carry_not_post_ten(self):
        tracker = self.new_game()
        for _ in range(9):
            self.advance(tracker, 1)
        self.assertEqual(tracker.counts, (9, 0, 0, 0))
        self.assertTrue(tracker.candidate_c(1))
        self.assertFalse(tracker.candidate_c(2))
        result = self.advance(tracker, 1)
        self.assertTrue(result['last_candidate_c'])
        self.assertEqual(tracker.raw_counters[0], 0x10)
        self.assertFalse(tracker.candidate_c(1))

    def test_clear_statistics_wrap_at160_events_not100(self):
        tracker = self.new_game()
        for _ in range(159):
            self.advance(tracker, 1)
        self.assertEqual(tracker.raw_counters[0], 0xF9)
        self.assertTrue(tracker.candidate_c(1))
        self.advance(tracker, 1)
        self.assertEqual(tracker.counts, (160, 0, 0, 0))
        self.assertEqual(tracker.raw_counters[0], 0)
        self.assertEqual(encode_clear_events(1119), 0xF9)
        self.assertEqual(encode_clear_events(1120), 0)
        for count in range(320):
            self.assertEqual(encode_clear_events(count), encode_clear_events(count+160))

    def test_four_counts_are_events_and_zero_clear_is_not_an_event(self):
        tracker = self.new_game()
        for cleared in (0, 4, 2, 1, 3, 4, 0):
            result = self.advance(tracker, cleared)
            self.assertTrue(result['known'])
        self.assertEqual(tracker.counts, (1, 1, 1, 2))
        self.assertEqual(tracker._boundary[1], 14)
        self.assertEqual(result['last_clear'], 0)
        self.assertFalse(result['last_candidate_c'])

    def test_duplicate_planning_or_cpu_overrun_observation_does_not_increment(self):
        tracker = self.new_game()
        for _ in range(10):
            self.advance(tracker, 2)
        before = tracker.counts
        for _ in range(50):
            result = tracker.update(*tracker._boundary)
            self.assertEqual(result['last_update'], 'duplicate')
            self.assertIsNone(result['last_clear'])
            self.assertIsNone(result['last_candidate_c'])
        self.assertEqual(tracker.counts, before)

    def test_missing_unsettled_and_skipped_history_is_sticky_unknown(self):
        tracker = ClearHistory()
        self.assertFalse(tracker.update(18, 0, 0)['known'])
        self.assertIsNone(tracker.candidate_c(1))
        self.assertFalse(tracker.candidate_c(0))
        for args, settled in (((18, 0, 2), True), ((18, 5, 1), True),
                              ((19, 0, 1), True), ((18, 0, 1), False),
                              ((18, 1, 0), True), ((18, 0, -1), True)):
            tracker = self.new_game()
            self.assertFalse(tracker.update(*args, settled=settled)['known'])
            self.assertIsNone(tracker.counts)
            self.assertIsNone(tracker.raw_counters)
            self.assertFalse(tracker.update(18, 0, 0)['known'])
            self.assertTrue(tracker.reset(18, 0, 0)['known'])
        for start in ((0, 0, 0), (18, 1, 0), (18, 0, 1), (235, 2290, 1000)):
            self.assertFalse(ClearHistory().reset(*start)['known'])
        self.assertFalse(ClearHistory().reset(18, 0, 0, settled=False)['known'])

    def test_original_long_level235_transition_is_not_linear(self):
        tracker = self.new_game()
        for absolute in range(1, 3101):
            self.assertTrue(self.advance(tracker, 1)['known'])
            if absolute in (2290, 2300, 3099):
                self.assertEqual(tracker._boundary[0], 235)
        self.assertEqual(tracker._boundary[:2], (236, 3100))
        self.assertEqual(tracker.counts, (3100, 0, 0, 0))
        # A naive +1 at2300 must invalidate otherwise complete history.
        tracker = self.new_game()
        for _ in range(2299):
            self.advance(tracker, 1)
        self.assertFalse(tracker.update(236, 2300, 2300)['known'])

    def test_raw_line_counter_wrap_keeps_complete_history(self):
        tracker = self.new_game()
        for _ in range(6400):
            self.assertTrue(self.advance(tracker, 4)['known'])
        self.assertEqual(tracker._boundary[1:], (0, 6400))
        self.assertEqual(tracker.counts, (0, 0, 0, 6400))
        self.assertEqual(tracker.raw_counters, (0, 0, 0, 0))
        self.advance(tracker, 1)
        self.assertEqual(tracker._boundary[1], 1)
        self.assertEqual(tracker.counts, (1, 0, 0, 6400))


if __name__ == '__main__':
    unittest.main()
