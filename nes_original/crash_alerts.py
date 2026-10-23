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

    def __call__(self,observation,plan):
        candidates = candidate_alerts(observation,plan)
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
