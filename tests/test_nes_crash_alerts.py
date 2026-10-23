"""Synthetic proposal fixtures only; no emulator, ROM or gameplay is started."""
from types import SimpleNamespace
import json

import numpy as np
import pytest

from nes.engine import Control
from nes_original.crash_alerts import DocumentedHazardPrioritizer,candidate_alerts
from nes_original.dellacherie import Dellacherie


def observation(level=157,lines=1511,score=999999):
    return SimpleNamespace(level=level,lines=lines,score=score,board=np.zeros((20,10),dtype=bool))


def proposals(lines,terminal=None):
    n=len(lines)
    return SimpleNamespace(actions=SimpleNamespace(lines=np.array(lines,dtype=np.int64),
        terminal=np.zeros(n,dtype=bool) if terminal is None else np.array(terminal,dtype=bool),
        boards=np.zeros((n,20,10),dtype=bool),
        controls=tuple(Control(x=i%8+1,y=18,orientation=18) for i in range(n))))


class FixedScores:
    def __init__(self,scores): self.scores=np.asarray(scores,dtype=float)
    def score_actions(self,observation,plan): return self.scores


@pytest.mark.parametrize('level,lines,clear,expected',[
    (154,1489,1,'single_entering_155'),
    (154,1488,1,None),(154,1488,2,None),(155,1491,1,None),
    (157,1511,1,'single_within_157'),(157,1519,1,None),
    (172,1661,3,'triple_within_172'),(171,1659,3,None),
    (173,1671,2,'double_within_173'),(172,1669,2,None),
    (254,3289,1,None),(255,3299,1,None),(250,3241,0,None),
])
def test_exact_four_signatures_and_source_counter_projection(level,lines,clear,expected):
    row=candidate_alerts(observation(level,lines),proposals([clear]))[0]
    assert row['alert']==expected
    if level==254:
        assert (row['predicted_level'],row['predicted_lines'])==(255,3290)
    if level==255:
        assert (row['predicted_level'],row['predicted_lines'])==(0,3300)


@pytest.mark.parametrize('score',[999998,0,1000000])
def test_alerts_only_at_original_score_cap(score):
    assert candidate_alerts(observation(score=score),proposals([1]))[0]['alert'] is None


def test_unalerted_survivors_precede_higher_scored_alerts_then_base_score_breaks_ties():
    selector=DocumentedHazardPrioritizer(FixedScores([100,3,9,9]))
    assert selector(observation(),proposals([1,2,0,0]))==2
    assert selector.last_decision['selection_tier']=='unalerted_nonterminal'
    assert selector.last_decision['candidates'][0]['alert']=='single_within_157'
    stats=selector.stats()
    assert stats['alert_candidates']['single_within_157']==1
    assert stats['calls_with_candidate_alerts']==stats['capped_calls']==stats['calls']==1
    assert stats['candidate_evaluations']==4
    json.dumps(stats)


def test_terminal_unalerted_candidate_never_beats_alerted_survivor_even_with_huge_score():
    selector=DocumentedHazardPrioritizer(FixedScores([1e12,-50,-10]))
    assert selector(observation(),proposals([2,1,1],[True,False,False]))==2
    assert selector.last_decision['selection_tier']=='alerted_nonterminal_fallback'
    assert selector.stats()['selection_tiers']['alerted_nonterminal_fallback']==1


def test_all_terminal_fallback_is_explicit_and_ranks_existing_base_scores():
    selector=DocumentedHazardPrioritizer(FixedScores([-9,-2]))
    assert selector(observation(),proposals([1,0],[True,True]))==1
    assert selector.last_decision['selection_tier']=='all_terminal_fallback'
    assert selector.last_decision['candidates'][1]['predicted_terminal'] is True
    assert selector.last_decision['unalerted_is_not_safe'] is True


def test_uncapped_score_preserves_surviving_base_ranking_without_alert_preference():
    selector=DocumentedHazardPrioritizer(FixedScores([9,1]))
    assert selector(observation(score=999998),proposals([1,2]))==0
    assert selector.stats()['calls_with_candidate_alerts']==0


def test_no_observation_proposal_score_or_stats_mutation():
    obs=observation();plan=proposals([1,2],[False,False]);base=FixedScores([9,1])
    before=(obs.board.tobytes(),plan.actions.boards.tobytes(),plan.actions.lines.copy(),
        plan.actions.terminal.copy(),plan.actions.controls,base.scores.copy())
    selector=DocumentedHazardPrioritizer(base)
    selector(obs,plan)
    assert (obs.board.tobytes(),plan.actions.boards.tobytes())==before[:2]
    assert np.array_equal(plan.actions.lines,before[2])
    assert np.array_equal(plan.actions.terminal,before[3])
    assert plan.actions.controls==before[4]
    assert np.array_equal(base.scores,before[5])
    exported=selector.stats()
    exported['last_decision']['candidates'][0]['alert']='mutated copy'
    exported['alert_candidates'].clear()
    assert selector.last_decision['candidates'][0]['alert']=='single_within_157'
    assert selector.stats()['alert_candidates']['single_within_157']==1


def test_dellacherie_score_method_and_default_first_argmax_choices_are_unchanged():
    obs=observation();plan=proposals([0,1,2],[False,False,True])
    plan.actions.boards[1,19,:3]=True
    plan.actions.controls=(plan.actions.controls[0],Control(x=2,y=19,orientation=18),plan.actions.controls[2])
    # Hand-counted:40 row transitions,10 column transitions, no holes/wells or
    # eroded full rows. Landing heights2,1,2; third retains the old1m penalty.
    expected=np.array([-52.,-51.,-1000052.])
    base=Dellacherie()
    assert np.array_equal(base.score_actions(obs,plan),expected)
    assert base.calls==1
    assert base(obs,plan)==int(expected.argmax())
    assert base.calls==2
    tied=proposals([0,0])
    tied.actions.controls=(tied.actions.controls[0],)*2
    assert base(obs,tied)==0


@pytest.mark.parametrize('scores',[[1],[float('nan'),2],[1,float('inf')]])
def test_malformed_scores_are_rejected_instead_of_silently_selecting(scores):
    with pytest.raises(ValueError):
        DocumentedHazardPrioritizer(FixedScores(scores))(observation(),proposals([1,2]))


def test_nonintegral_or_out_of_range_clear_predictions_are_rejected():
    plan=proposals([1]);plan.actions.lines=np.array([1.5])
    with pytest.raises(TypeError): candidate_alerts(observation(),plan)
    with pytest.raises(ValueError): candidate_alerts(observation(),proposals([5]))
