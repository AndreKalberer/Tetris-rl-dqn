import pytest
from dqn.line_challenge import stop_reason


@pytest.mark.parametrize('changes,expected',[
    ({'lines':10000},'line_target_reached'),
    ({'lines':10001,'terminated':True},'line_target_reached'),
    ({'terminated':True},'top_out'),
    ({'placements':100000},'placement_budget'),
    ({'seconds':900},'time_budget'),
    ({'truncated':True},'environment_truncated'),
    ({},None),
])
def test_line_goal_and_resource_cutoffs_have_distinct_outcomes(changes,expected):
    state=dict(lines=9999,target=10000,terminated=False,truncated=False,
               placements=25000,max_placements=100000,seconds=100,max_seconds=900)
    state.update(changes)
    assert stop_reason(**state)==expected
