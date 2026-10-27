"""Synthetic proposal tests, not a native timing oracle or gameplay result."""
import hashlib
import json
from types import SimpleNamespace
import unittest

import numpy as np

from nes_original.crash_alerts import (DocumentedHazardPrioritizer,
    Observed167TriplePrioritizer, SIGNATURES, candidate_alerts, observed167_candidate_alerts)


def observation(level=167, lines=1610, score=999999):
    return SimpleNamespace(level=level, lines=lines, score=score)


def plan(clears, terminal=None):
    return SimpleNamespace(actions=SimpleNamespace(lines=np.asarray(clears, dtype=np.int64),
        terminal=np.zeros(len(clears), dtype=bool) if terminal is None else np.asarray(terminal, dtype=bool)))


class FixedScores:
    def __init__(self, values): self.values = np.asarray(values, dtype=float)
    def score_actions(self, observation, plan): return self.values


class Observed167AlertTests(unittest.TestCase):
    def test_cap_and_projected_within167_tuple_only(self):
        for level, lines, clear, score, expected in (
                (167, 1610, 3, 999999, True), (167, 1616, 3, 999999, True),
                (167, 1617, 3, 999999, False), (166, 1609, 3, 999999, False),
                (168, 1621, 3, 999999, False), (167, 1610, 2, 999999, False),
                (167, 1610, 4, 999999, False), (167, 1610, 3, 999998, False),
                (167, 1610, 3, 1000000, False)):
            with self.subTest(level=level, lines=lines, clear=clear, score=score):
                row = observed167_candidate_alerts(observation(level, lines, score), plan([clear]))[0]
                self.assertEqual(row['alert'] == 'native_observed_triple_within_167', expected)
                if level == 167 and lines == 1617:
                    self.assertEqual(row['predicted_level'], 168)

    def test_known_start_proxy_is_separate_from_documented_signatures(self):
        self.assertNotIn((167, 167, 3), SIGNATURES)
        self.assertIsNone(candidate_alerts(observation(), plan([3]))[0]['alert'])
        row = observed167_candidate_alerts(observation(157, 1511), plan([1]))[0]
        self.assertEqual(row['alert'], 'single_within_157')
        self.assertEqual(row['alert_origin'], 'documented_conditional_signature')

    def test_unalerted_survivor_precedes_proxy_then_first_tie_wins(self):
        selector = Observed167TriplePrioritizer(FixedScores([100, 4, 4]))
        self.assertEqual(selector(observation(), plan([3, 1, 0])), 1)
        decision = selector.last_decision
        self.assertEqual(decision['selection_tier'], 'unalerted_nonterminal')
        self.assertEqual(decision['candidates'][0]['alert_origin'], 'native_observed_public_counter_proxy')
        self.assertTrue(decision['unalerted_is_not_safe'])

    def test_alerted_survivor_precedes_unalerted_terminal_and_all_terminal_fallback_survives(self):
        selector = Observed167TriplePrioritizer(FixedScores([1e9, -5, -5]))
        self.assertEqual(selector(observation(), plan([0, 3, 3], [True, False, False])), 1)
        self.assertEqual(selector.last_decision['selection_tier'], 'alerted_nonterminal_fallback')
        selector = Observed167TriplePrioritizer(FixedScores([2, 9]))
        self.assertEqual(selector(observation(), plan([3, 0], [True, True])), 1)
        self.assertEqual(selector.last_decision['selection_tier'], 'all_terminal_fallback')

    def test_stats_origin_verified_fault_evidence_and_no_input_or_export_mutation(self):
        obs, proposals = observation(), plan([3, 1])
        original = proposals.actions.lines.copy()
        base = FixedScores([10, 2])
        selector = Observed167TriplePrioritizer(base)
        selector(obs, proposals)
        exported = selector.stats()
        self.assertEqual(exported['alert_candidates_by_origin'], dict(
            documented_conditional_signature=0, native_observed_public_counter_proxy=1))
        self.assertEqual(exported['observed_evidence']['known_menu_wait'], 33)
        self.assertFalse(exported['observed_evidence']['raw_pushdown_or_native_timing_used_by_selector'])
        self.assertEqual(exported['observed_evidence']['original_fault_replay_status'], 'verified_original_fault_replay')
        self.assertEqual(exported['last_decision']['observed_fault_replay_status'], 'verified_original_fault_replay')
        exported['last_decision']['candidates'][0]['alert_origin'] = 'mutated'
        self.assertEqual(selector.last_decision['candidates'][0]['alert_origin'], 'native_observed_public_counter_proxy')
        np.testing.assert_array_equal(proposals.actions.lines, original)
        np.testing.assert_array_equal(base.values, [10, 2])
        json.dumps(selector.stats())

    def test_default_documented_choices_and_all_stats_match_prechange_golden(self):
        # Captured before adding the optional extension. No RNG or game needed.
        fixtures = [(154, 1489), (157, 1511), (167, 1610), (172, 1661), (173, 1671), (254, 3289)]
        output = []
        for i in range(72):
            level, lines = fixtures[i % 6]
            terminal = ([True]*5 if i % 9 == 0 else [False]*5 if i % 9 == 1
                        else [(i+2*j) % 4 == 0 for j in range(5)])
            proposals = plan([(i+3*j) % 5 for j in range(5)], terminal)
            obs = observation(level, lines, 999999 if i % 3 else 999998)
            selector = DocumentedHazardPrioritizer(FixedScores([(i*7+j*3) % 21-10 for j in range(5)]))
            output.append(dict(choice=selector(obs, proposals), stats=selector.stats()))
        actual = hashlib.sha256(json.dumps(output, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(actual, '95332f22eeee383439a5aad0cc54fbcaa9adbfcab86bd22ee69386004f684bd6')


if __name__ == '__main__': unittest.main()
