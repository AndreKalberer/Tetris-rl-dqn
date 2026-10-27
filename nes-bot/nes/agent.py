"""Timed playback and replay recording for NES-specific neural policies."""
from collections import deque
from dataclasses import asdict
import time

import numpy as np

from nes.actions import enumerate_actions
from nes.engine import FPS,Game,Phase,Rules
from nes.learn import load_policy,values


class PlanningBudgetExceeded(RuntimeError):
    pass


class CandidateController:
    def __init__(self,config,model=None,policy='learned',seed=0,account_latency=True):
        self.config,self.model,self.policy=config,model,policy
        self.rng=np.random.default_rng(seed)
        self.account_latency=account_latency
        self.name='NES '+policy
        self.pending=deque()
        self.expected_board=None
        self.expected_lines=None
        self.delay=0
        self.extra_idle_frames=self.executed_plans=0
        self.planning_seconds=0.0
        self.plans=self.overruns=self.path_mismatches=0
        # Initialize CPU inference before the first per-piece planning clock.
        if model is not None:
            dim=next(model.parameters()).shape[1]
            values(self.model,np.zeros((1,dim),dtype=np.float32))

    def command(self,observation):
        if self.delay:
            self.delay-=1
            self.extra_idle_frames+=1
            return 0
        if observation.phase!=Phase.ACTIVE:
            if self.pending: raise AssertionError('Learned action ended early')
            return 0
        if self.pending:
            return self.pending.popleft()
        if self.expected_board is not None:
            if not np.array_equal(observation.board!=0,self.expected_board!=0) or observation.lines!=self.expected_lines:
                self.path_mismatches+=1
                raise AssertionError('Learned path differs from its preview')
            self.expected_board=None
        start=time.perf_counter()
        actions=enumerate_actions(observation,self.config.actions,board_input=self.config.board_input)
        if self.policy=='learned': selected=int(values(self.model,actions.features).argmax())
        elif self.policy=='heuristic': selected=int(actions.heuristic.argmax())
        elif self.policy=='random': selected=int(self.rng.integers(len(actions.paths)))
        else: raise ValueError('Unknown NES candidate policy')
        elapsed=time.perf_counter()-start
        self.planning_seconds+=elapsed
        self.plans+=1
        if elapsed>self.config.reaction_frames/FPS:
            self.overruns+=1
            if self.account_latency:
                self.delay=max(0,int(np.ceil(elapsed*FPS))-1)
                self.extra_idle_frames+=1
                return 0
        self.pending=deque(actions.paths[selected])
        self.expected_board=actions.boards[selected]
        self.expected_lines=observation.lines+int(actions.lines[selected])
        self.executed_plans+=1
        return self.pending.popleft()


class LearnedController(CandidateController):
    def __init__(self,checkpoint):
        config,model,self.metadata=load_policy(checkpoint)
        super().__init__(config,model)
        self.name=f"NES {self.metadata['stage']} / {self.metadata['training_decisions']:,} RL decisions"


def record_episode(controller,seed=0,start_level=0,max_frames=80000,wall_seconds=120,callback=None,max_placements=1000):
    from nes.evaluate import replay
    game=Game(seed,Rules(start_level=start_level))
    inputs,locks=[],[]
    previous=-1
    started=time.perf_counter()
    reason=None
    while not game.over and game.frames<max_frames and game.placements<max_placements and time.perf_counter()-started<wall_seconds:
        buttons=controller.command(game.observe())
        if buttons!=previous:
            inputs.append(dict(frame=game.frames+1,buttons=buttons))
            previous=buttons
        event=game.tick(buttons)
        if event['locked']: locks.append(dict(frame=game.frames,state_sha256=game.signature()))
        if callback: callback(game)
    reason=reason or ('natural_top_out' if game.over else 'simulation_frame_limit' if game.frames>=max_frames else
                     'placement_limit' if game.placements>=max_placements else 'wall_time_limit')
    row=dict(seed=seed,rules=asdict(game.rules),controller=controller.name,placements=game.placements,
        lines=game.lines,score=game.score,level=game.level,frames=game.frames,inputs=inputs,locks=locks,
        wall_seconds=time.perf_counter()-started,planning_seconds=controller.planning_seconds,
        stopped_by=reason,completed=game.over,alive_at_stop=not game.over,
        valid_timing=controller.account_latency or not controller.overruns,
        real_time_accounted=controller.account_latency,
        planning_overruns=controller.overruns,path_mismatches=controller.path_mismatches,
        extra_idle_frames=controller.extra_idle_frames,executed_plans=controller.executed_plans,
        latency_accounting=('Minimum reaction delay; over-budget proposals discarded and elapsed frames charged as idle before replanning'
                           if controller.account_latency else 'Fixed logical reaction only; CPU timing recorded, not part of this model-selection simulation'),
        final_state_sha256=game.signature(),scope='NES-trained policy in provisional simulator; whole-ROM parity pending')
    assert replay(row).signature()==game.signature()
    row['replay_verified']=True
    return row,game


def episode(checkpoint,seed=0,start_level=0,max_frames=120000,wall_seconds=60,callback=None):
    return record_episode(LearnedController(checkpoint),seed,start_level,max_frames,wall_seconds,callback)
