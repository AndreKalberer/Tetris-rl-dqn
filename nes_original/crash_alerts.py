"""Optional prioritization of four documented original-ROM crash signatures.

This is NOT a crash oracle or a safe-action table. Alerts use visible counters
and approximate proposal line counts; real transitions remain in Mesen.
Pushdown, row/carry/render/CPU timing modifiers and other crashes are unmodelled.
An unalerted action is unknown, never certified safe. No RNG or hidden state is
read, and no controls, proposal arrays or game state are changed here.

Primary setup list (conditional baseline hazards, not unconditional rules):
https://docs.google.com/spreadsheets/d/1zAQIo_mnkk0c9e4-hpeDvVxrl9r_HvLSx8V4h4ttmrs/edit#gid=1013692687
"""
import operator

import numpy as np

from nes_original.dellacherie import Dellacherie
from nes_original.progression import after_clear


SIGNATURES = {
    (154,155,1): 'single_entering_155',
    (157,157,1): 'single_within_157',
    (172,172,3): 'triple_within_172',
    (173,173,2): 'double_within_173',
}
SCOPE = ('Four documented conditional hazard alerts only; visible counters and approximate proposal clears; '
         'unalerted candidates and timing modifiers remain unknown, not safe')


def candidate_alerts(observation, plan):
    """Return fresh JSON-compatible metadata; never mutate the input proposals.

    Projection uses the byte-correct original counter routine. Alerts apply
    only at the original score cap999999, and only to the four exact tuples.
    A projected level255/3290-line threshold is not an alert or goal claim.
    """
    actions = plan.actions
    lines = np.asarray(actions.lines)
    terminal = np.asarray(actions.terminal)
    if lines.ndim!=1 or terminal.shape!=lines.shape or not len(lines):
        raise ValueError('Expected a nonempty line/terminal vector for the same proposals')
    level,raw_lines = operator.index(observation.level),operator.index(observation.lines)
    capped = observation.score==999999
    result = []
    for index,(cleared,dead) in enumerate(zip(lines,terminal)):
        cleared = operator.index(cleared)
        post_level,post_lines = after_clear(level,raw_lines,cleared)
        signature = SIGNATURES.get((level,post_level,cleared)) if capped else None
        result.append(dict(index=index,pre_level=level,predicted_level=post_level,
            predicted_lines=post_lines,predicted_cleared_lines=cleared,
            predicted_terminal=bool(dead),alert=signature))
    return result


class DocumentedHazardPrioritizer:
    """Prefer an unalerted predicted survivor, then rank by existing D score.

    If every predicted survivor is alerted, choose the best alerted survivor.
    Only when ALL proposals predict termination do we fall back to the best
    terminal proposal. These are proposal preferences, not game guarantees.
    """
    def __init__(self,base_selector=None):
        self.base_selector = Dellacherie() if base_selector is None else base_selector
        if not callable(getattr(self.base_selector,'score_actions',None)):
            raise TypeError('Base selector must expose score_actions(observation, plan)')
        self.calls = self.candidate_evaluations = self.capped_calls = 0
        self.calls_with_candidate_alerts = 0
        self.alert_candidates = {name:0 for name in SIGNATURES.values()}
        self.selections = dict(unalerted_nonterminal=0,alerted_nonterminal_fallback=0,
            all_terminal_fallback=0)
        self.last_decision = None

    def _candidate_alerts(self, observation, plan):
        return candidate_alerts(observation, plan)

    def __call__(self,observation,plan):
        candidates = self._candidate_alerts(observation,plan)
        scores = np.array(self.base_selector.score_actions(observation,plan),dtype=float,copy=True)
        if scores.shape!=(len(candidates),) or not np.isfinite(scores).all():
            raise ValueError('Expected one finite base score per proposal')
        nonterminal = np.array([not row['predicted_terminal'] for row in candidates],dtype=bool)
        unalerted = np.array([row['alert'] is None for row in candidates],dtype=bool)
        if (nonterminal & unalerted).any():
            eligible = np.flatnonzero(nonterminal & unalerted)
            tier = 'unalerted_nonterminal'
        elif nonterminal.any():
            eligible = np.flatnonzero(nonterminal)
            tier = 'alerted_nonterminal_fallback'
        else:
            eligible = np.arange(len(candidates))
            tier = 'all_terminal_fallback'
        # np.argmax preserves the original first-index tie break in each tier.
        selected = int(eligible[np.argmax(scores[eligible])])
        self.calls += 1
        self.candidate_evaluations += len(candidates)
        self.capped_calls += int(observation.score==999999)
        has_alert = False
        for row,score in zip(candidates,scores):
            row['base_score'] = float(score)
            if row['alert'] is not None:
                has_alert = True
                self.alert_candidates[row['alert']] += 1
        self.calls_with_candidate_alerts += int(has_alert)
        self.selections[tier] += 1
        self.last_decision = dict(selected=selected,selection_tier=tier,
            score_capped=bool(observation.score==999999),candidates=candidates,
            unalerted_is_not_safe=True,scope=SCOPE)
        return selected

    def stats(self):
        """Copies of counters and latest candidate log, bounded to one choice."""
        import copy
        return dict(calls=self.calls,candidate_evaluations=self.candidate_evaluations,
            capped_calls=self.capped_calls,calls_with_candidate_alerts=self.calls_with_candidate_alerts,
            alert_candidates=dict(self.alert_candidates),selection_tiers=dict(self.selections),
            last_decision=copy.deepcopy(self.last_decision),scope=SCOPE)


_OBSERVED_167_ALERT = 'native_observed_triple_within_167'
_OBSERVED_167_SCOPE = (
    'Experimental four-documented-plus-native-observed167 proxy; one known-start33 development event; '
    'capped public counters and approximate clear projection only, not an exact pushdown/timing crash oracle; '
    'the original observed fault has an exact verified replay; unalerted remains unknown, not safe')


def observed167_candidate_alerts(observation, plan):
    """Separate experimental extension; candidate_alerts/SIGNATURES stay intact.

    One recorded known-start33 development event was a capped triple within167
    with raw pushdown6. This broader counter proxy does NOT read pushdown, infer
    native CPU timing, fix the ambiguous primary table label, or certify risk.
    Its original fault was reproduced by an exact full native input replay.
    That proof establishes repeatability of that event, not this proxy's efficacy.
    """
    rows = candidate_alerts(observation, plan)
    for row in rows:
        if (observation.score == 999999 and
                (row['pre_level'], row['predicted_level'], row['predicted_cleared_lines']) == (167, 167, 3)):
            row['alert'] = _OBSERVED_167_ALERT
            row['alert_origin'] = 'native_observed_public_counter_proxy'
        else:
            row['alert_origin'] = 'documented_conditional_signature' if row['alert'] is not None else None
    return rows


class Observed167TriplePrioritizer(DocumentedHazardPrioritizer):
    """Optional development profile; retained four-alert defaults are unchanged."""
    def __init__(self, base_selector=None):
        super().__init__(base_selector)
        self.alert_candidates[_OBSERVED_167_ALERT] = 0

    def _candidate_alerts(self, observation, plan):
        return observed167_candidate_alerts(observation, plan)

    def __call__(self, observation, plan):
        selected = super().__call__(observation, plan)
        self.last_decision.update(scope=_OBSERVED_167_SCOPE,
            alert_profile='documented_four_plus_native_observed167',
            observed_alert_is_public_counter_proxy=True,
            observed_fault_replay_status='verified_original_fault_replay')
        return selected

    def stats(self):
        result = super().stats()
        result.update(scope=_OBSERVED_167_SCOPE,
            alert_profile='documented_four_plus_native_observed167',
            alert_candidates_by_origin=dict(
                documented_conditional_signature=sum(self.alert_candidates[name] for name in SIGNATURES.values()),
                native_observed_public_counter_proxy=self.alert_candidates[_OBSERVED_167_ALERT]),
            observed_evidence=dict(known_menu_wait=33, development_only=True, pre_level=167,
                post_level=167, cleared_lines=3, score_cap=999999, recorded_raw_pushdown=6,
                raw_pushdown_or_native_timing_used_by_selector=False,
                original_fault_replay_status='verified_original_fault_replay'))
        return result
