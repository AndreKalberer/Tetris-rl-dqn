from dataclasses import replace
import numpy as np

from nes.actions import ActionRules,enumerate_actions,execute
from nes.engine import Game,Rules


def test_every_candidate_replays_real_nes_inputs_at_multiple_speeds():
    for level in (0,18,19):
        base=Game(20,Rules(start_level=level))
        snapshot=base.get_state()
        actions=enumerate_actions(base.observe())
        assert len(actions.paths)>8
        for index,path in enumerate(actions.paths):
            game=Game.from_state(snapshot)
            execute(game,actions,index)
            assert path[:3]==(0,0,0)
            assert all(buttons&3!=3 for buttons in path)
        assert base.get_state()==snapshot


def test_candidates_do_not_access_hidden_rng_or_new_preview():
    a=Game(10)
    b=Game.from_state(a.get_state())
    b._randomizer.state=0x1234
    b._randomizer.count=190
    b._randomizer.last=18
    x,y=enumerate_actions(a.observe()),enumerate_actions(b.observe())
    assert np.array_equal(x.features,y.features) and x.paths==y.paths


def test_reachable_heuristic_executes_complete_games_without_preview_errors():
    for seed in (20,21):
        game=Game(seed,Rules(start_level=19))
        for _ in range(100):
            actions=enumerate_actions(game.observe())
            execute(game,actions,int(actions.heuristic.argmax()))
            if game.over: break
        assert game.placements>5


def test_nes_training_checkpoint_continues_exactly(tmp_path):
    import torch
    from nes.learn import Config,Trainer,load_policy
    config=Config(seed=5,levels=(19,),capacity=64,batch_size=8,learning_starts=8,train_every=1,target_every=5)
    trainer=Trainer(config)
    for _ in range(15): trainer.advance()
    checkpoint=tmp_path/'nes.pt'
    trainer.save(checkpoint)
    for _ in range(4): trainer.advance()
    expected=trainer.game.signature()
    parameters=[parameter.detach().clone() for parameter in trainer.online.parameters()]
    restored=Trainer.load(checkpoint)
    for _ in range(4): restored.advance()
    assert restored.step==trainer.step and restored.game.signature()==expected
    assert all(torch.equal(a,b) for a,b in zip(parameters,restored.online.parameters()))
    policy=tmp_path/'nes-policy.pt'
    restored.export(policy,'double-dqn')
    _,model,metadata=load_policy(policy)
    assert metadata['training_decisions']==19
    assert all(torch.equal(a,b) for a,b in zip(parameters,model.parameters()))


def test_teacher_training_is_recorded_separately_from_rl():
    from nes.learn import Config,Trainer
    trainer=Trainer(Config(seed=3,levels=(19,),capacity=64,batch_size=8,learning_starts=8))
    trainer.warm_start(8,10)
    assert trainer.step==0 and trainer.teacher_decisions==8 and trainer.supervised_updates==10
    assert np.isfinite(trainer.last_loss) and len(trainer.replay.items)==8


def test_learned_controller_records_a_replayable_timed_episode(tmp_path):
    from nes.learn import Config,Trainer
    from nes.agent import episode
    trainer=Trainer(Config(seed=1,reaction_frames=12))
    policy=tmp_path/'policy.pt'
    trainer.export(policy,'test-only')
    row,game=episode(policy,seed=4,start_level=19,max_frames=800)
    assert row['replay_verified'] and row['path_mismatches']==0
    assert row['frames']==game.frames


def test_nes_policy_loader_rejects_other_model_formats(tmp_path):
    import pytest,torch
    from nes.learn import load_policy
    path=tmp_path/'modern.pt'
    torch.save({'encoding_version':1,'kind':'modern'},path)
    with pytest.raises(ValueError,match='NES-specific'):
        load_policy(path)


def test_late_plan_is_discarded_and_its_time_advances_real_game_frames(monkeypatch):
    from nes.agent import CandidateController,record_episode
    from nes.learn import Config
    import nes.agent
    counter=iter(i*0.1 for i in range(10000))
    monkeypatch.setattr(nes.agent.time,'perf_counter',lambda:next(counter))
    controller=CandidateController(Config(),None,'heuristic')
    row,game=record_episode(controller,seed=0,max_frames=30)
    assert row['planning_overruns']>0 and row['extra_idle_frames']==30
    assert game.frames==30 and row['replay_verified'] and row['valid_timing']
    assert controller.executed_plans==0


def test_logical_model_selection_does_not_change_actions_with_cpu_jitter(monkeypatch):
    from nes.agent import CandidateController,record_episode
    from nes.learn import Config
    import nes.agent
    baseline,_=record_episode(CandidateController(Config(),None,'heuristic',account_latency=False),seed=4,max_frames=200)
    counter=iter(i*0.1 for i in range(10000))
    monkeypatch.setattr(nes.agent.time,'perf_counter',lambda:next(counter))
    delayed,_=record_episode(CandidateController(Config(),None,'heuristic',account_latency=False),seed=4,max_frames=200)
    assert delayed['final_state_sha256']==baseline['final_state_sha256']
    assert delayed['planning_overruns']>0 and delayed['extra_idle_frames']==0
    assert not delayed['real_time_accounted']
