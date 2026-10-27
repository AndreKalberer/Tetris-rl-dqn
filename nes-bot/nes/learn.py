"""NES-specific candidate Double DQN and an explicitly labeled teacher warm-start."""
import argparse
from dataclasses import asdict,dataclass
import hashlib
import json
import os
import re
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from dqn.train_candidates import Replay,double_targets,network,values
from nes.actions import ActionRules,ENCODING_VERSION,enumerate_actions,execute,structural
from nes.engine import FPS,Game,Rules


KIND='nes-timed-candidate-double-dqn'
VALIDATION_SEEDS=(10101,13007)
RESERVED_SEEDS=(20003,24007,30011)
EVAL_LEVELS=(0,18,19)


@dataclass
class Config:
    seed: int=0
    levels: tuple=(0,9,18,19)
    board_input: bool=False
    reaction_frames: int=3
    navigation_frames: int=48
    rotation_period: int=4
    gamma: float=0.99
    learning_rate: float=0.0001
    capacity: int=5000
    batch_size: int=64
    learning_starts: int=256
    train_every: int=4
    target_every: int=500
    epsilon_start: float=0.15
    epsilon_final: float=0.03
    epsilon_steps: int=20000
    threads: int=2

    def __post_init__(self):
        self.levels=tuple(self.levels)
        if not self.levels or any(not 0<=level<=19 for level in self.levels):
            raise ValueError('Invalid NES training starting levels')
        if min(self.batch_size,self.capacity,self.train_every,self.target_every,self.threads,self.epsilon_steps)<1:
            raise ValueError('Invalid replay/update configuration')
        if self.capacity<max(self.learning_starts,self.batch_size):
            raise ValueError('Replay is too small')
        if not 0<=self.epsilon_final<=self.epsilon_start<=1 or not 0<self.gamma<=1:
            raise ValueError('Invalid exploration/discount configuration')

    @property
    def actions(self):
        return ActionRules(self.reaction_frames,self.navigation_frames,self.rotation_period)


def write_json(path,data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2),encoding='utf-8')
    temporary.replace(path)


def publish_progress(status):
    root=Path(__file__).resolve().parents[2]  # repository root
    task=root.parents[1]
    output=task/'outputs/progress.md'
    if not output.exists(): return
    text=output.read_text(encoding='utf-8')
    description=(f"NES pilot {status['state']}; PID {status.get('pid')}; phase {status.get('phase')}; "
        f"{status.get('rl_decisions',0):,} RL decisions, {status.get('teacher_decisions',0):,} teacher decisions; "
        f"loss {status.get('loss')}. Training metrics are development output, not held-out strength.")
    text=re.sub(r'^Active work:.*$','Active work: '+description,text,flags=re.M)
    for target in (output,root/'progress.md'):
        temporary=target.with_suffix('.md.tmp')
        temporary.write_text(text,encoding='utf-8',newline='\n')
        temporary.replace(target)
    write_json(task/'outputs/nes-training-status.json',status)


def load_policy(path):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state.get('kind')!=KIND or state.get('encoding_version')!=ENCODING_VERSION:
        raise ValueError('Expected a NES-specific policy, not a modern Tetris checkpoint')
    config=Config(**state['config'])
    torch.set_num_threads(config.threads)
    model=network(state['dim'])
    model.load_state_dict(state['online'])
    return config,model,state


def shaped_reward(before,after,lines,done):
    a,b=structural(before[None])[0],structural(after[None])[0]
    return float(0.1+lines**2-10*done-10*(b[0]-a[0])-60*(b[1]-a[1])-4*(b[2]-a[2]))


class Trainer:
    def __init__(self,config):
        self.config=config
        torch.set_num_threads(config.threads)
        torch.manual_seed(config.seed)
        self.rng=np.random.default_rng(config.seed)
        self.replay=Replay(config.capacity)
        self.step=self.teacher_decisions=self.supervised_updates=self.episodes=self.placements=0
        self.last_loss=None
        self.history=[]
        self.reset_game()
        self.dim=self.actions.features.shape[1]
        self.online,self.target=network(self.dim),network(self.dim)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer=torch.optim.Adam(self.online.parameters(),lr=config.learning_rate)

    def reset_game(self):
        seed=int(self.rng.integers(0,10000))
        level=int(self.rng.choice(self.config.levels))
        self.game=Game(seed,Rules(start_level=level))
        self.actions=enumerate_actions(self.game.observe(),self.config.actions,board_input=self.config.board_input)
        self.episode_decisions=0

    def transition(self,index):
        before=self.game.board.copy()
        features=self.actions.features[index]
        placed=self.game.placements
        lines,done=execute(self.game,self.actions,index)
        self.placements+=self.game.placements-placed
        reward=shaped_reward(before,self.game.board,lines,done)
        next_actions=None if done else enumerate_actions(self.game.observe(),self.config.actions,board_input=self.config.board_input)
        next_features=np.empty((0,self.dim),dtype=np.float32) if done else next_actions.features
        self.replay.add(features,reward,next_features,done)
        self.episode_decisions+=1
        if done:
            self.episodes+=1
            self.history.append(dict(seed=self.game.seed,start_level=self.game.rules.start_level,
                lines=self.game.lines,level=self.game.level,score=self.game.score,decisions=self.episode_decisions))
            self.reset_game()
        else:
            self.actions=next_actions
        return done

    def warm_start(self,decisions,updates,callback=None):
        features,targets=[],[]
        for index in range(decisions):
            mask=~self.actions.terminal
            if mask.any():
                features.append(self.actions.features[mask])
                targets.append(np.clip(self.actions.heuristic[mask]/10,-30,10))
            selected=int(self.actions.heuristic.argmax())
            if self.rng.random()<0.05:
                selected=int(self.rng.integers(len(self.actions.paths)))
            self.transition(selected)
            self.teacher_decisions+=1
            if callback and (index+1)%100==0: callback('teacher_collect',index+1)
        xs,ys=np.concatenate(features),np.concatenate(targets)
        optimizer=torch.optim.Adam(self.online.parameters(),lr=0.001)
        for update in range(updates):
            indices=self.rng.integers(len(xs),size=256)
            predicted=self.online(torch.from_numpy(xs[indices])).squeeze(-1)
            loss=nn.functional.mse_loss(predicted,torch.from_numpy(ys[indices]))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.online.parameters(),10)
            optimizer.step()
            self.last_loss=float(loss.detach())
            self.supervised_updates+=1
            if callback and (update+1)%1000==0: callback('teacher_fit',update+1)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer=torch.optim.Adam(self.online.parameters(),lr=self.config.learning_rate)
        self.reset_game()
        self.history=[]

    def advance(self):
        c=self.config
        epsilon=c.epsilon_start+min(1,self.step/c.epsilon_steps)*(c.epsilon_final-c.epsilon_start)
        selected=int(self.rng.integers(len(self.actions.paths))) if self.rng.random()<epsilon else int(values(self.online,self.actions.features).argmax())
        self.transition(selected)
        self.step+=1
        if len(self.replay.items)>=max(c.learning_starts,c.batch_size) and self.step%c.train_every==0:
            batch=self.replay.sample(c.batch_size,self.rng)
            targets=double_targets(self.online,self.target,batch,c.gamma)
            predicted=self.online(torch.from_numpy(np.stack([item[0] for item in batch]).astype(np.float32))).squeeze(-1)
            loss=nn.functional.smooth_l1_loss(predicted,targets)
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite NES loss')
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.online.parameters(),10)
            self.optimizer.step()
            self.last_loss=float(loss.detach())
        if self.step%c.target_every==0:
            self.target.load_state_dict(self.online.state_dict())

    def save(self,path):
        payload=dict(kind=KIND,encoding_version=ENCODING_VERSION,config=asdict(self.config),dim=self.dim,
            online=self.online.state_dict(),target=self.target.state_dict(),optimizer=self.optimizer.state_dict(),
            replay=dict(capacity=self.replay.capacity,items=self.replay.items,position=self.replay.position),
            step=self.step,teacher_decisions=self.teacher_decisions,supervised_updates=self.supervised_updates,
            episodes=self.episodes,placements=self.placements,last_loss=self.last_loss,history=self.history,
            episode_decisions=self.episode_decisions,game=self.game.get_state(),
            rng=self.rng.bit_generator.state,torch_rng=torch.get_rng_state())
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix(path.suffix+'.tmp')
        torch.save(payload,temporary)
        temporary.replace(path)

    @classmethod
    def load(cls,path):
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state.get('kind')!=KIND or state.get('encoding_version')!=ENCODING_VERSION:
            raise ValueError('Unsupported NES training checkpoint')
        trainer=cls(Config(**state['config']))
        for name in ('online','target','optimizer'):
            getattr(trainer,name).load_state_dict(state[name])
        trainer.replay.items=state['replay']['items']
        trainer.replay.position=state['replay']['position']
        for name in ('step','teacher_decisions','supervised_updates','episodes','placements','last_loss','history','episode_decisions'):
            setattr(trainer,name,state[name])
        trainer.game=Game.from_state(state['game'])
        trainer.actions=enumerate_actions(trainer.game.observe(),trainer.config.actions,board_input=trainer.config.board_input)
        trainer.rng.bit_generator.state=state['rng']
        torch.set_rng_state(state['torch_rng'])
        return trainer

    def export(self,path,stage):
        torch.save(dict(kind=KIND,encoding_version=ENCODING_VERSION,config=asdict(self.config),dim=self.dim,
            online=self.online.state_dict(),training_decisions=self.step,teacher_decisions=self.teacher_decisions,
            supervised_updates=self.supervised_updates,stage=stage),path)


def evaluate(config,model=None,policy='learned',seeds=VALIDATION_SEEDS,levels=EVAL_LEVELS,max_decisions=1000,trace_dir=None):
    from nes.evaluate import replay
    rows=[]
    for level in levels:
        for seed in seeds:
            game=Game(int(seed),Rules(start_level=int(level)))
            rng=np.random.default_rng(seed)
            started=time.perf_counter()
            inputs,locks=[],[]
            previous=-1
            max_planning=0.0
            reason=None
            decisions=0
            def record(current,buttons,event):
                nonlocal previous
                if buttons!=previous:
                    inputs.append(dict(frame=current.frames,buttons=buttons))
                    previous=buttons
                if event['locked']: locks.append(dict(frame=current.frames,state_sha256=current.signature()))
            for _ in range(max_decisions):
                planned=time.perf_counter()
                actions=enumerate_actions(game.observe(),config.actions,board_input=config.board_input)
                if policy=='heuristic': index=int(actions.heuristic.argmax())
                elif policy=='random': index=int(rng.integers(len(actions.paths)))
                elif policy=='learned' and model is not None: index=int(values(model,actions.features).argmax())
                else: raise ValueError('Missing NES model or unknown policy')
                latency=time.perf_counter()-planned
                max_planning=max(max_planning,latency)
                if latency>config.reaction_frames/FPS:
                    reason='planning_budget'
                    break
                execute(game,actions,index,callback=record)
                decisions+=1
                if game.over:
                    reason='natural_top_out'
                    break
                if game.frames>=80000 or time.perf_counter()-started>=120:
                    reason='resource_budget'
                    break
            reason=reason or 'decision_limit'
            row=dict(seed=int(seed),rules=asdict(game.rules),policy=policy,decisions=decisions,
                placements=game.placements,lines=game.lines,score=game.score,level=game.level,frames=game.frames,
                stopped_by=reason,completed=game.over,alive_at_stop=not game.over,
                valid_timing=reason!='planning_budget',max_planning_seconds=max_planning,
                wall_seconds=time.perf_counter()-started,final_state_sha256=game.signature(),inputs=inputs,locks=locks,
                scope='Small provisional-simulator sample, fixed three-frame reaction; not original-ROM performance')
            assert replay(row).signature()==game.signature()
            row['replay_verified']=True
            if trace_dir:
                write_json(Path(trace_dir)/f'level{level}-seed{seed}.json',row)
            rows.append({k:v for k,v in row.items() if k not in ('inputs','locks')})
    valid=[row for row in rows if row['valid_timing']]
    return dict(policy=policy,games=rows,all_timing_valid=len(valid)==len(rows),
        mean_observed_lines=float(np.mean([row['lines'] for row in valid])) if valid else None,
        selection_score=float(np.mean([row['lines'] for row in valid])) if len(valid)==len(rows) else -1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=20000)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--teacher-decisions',type=int,default=1000)
    parser.add_argument('--teacher-updates',type=int,default=5000)
    parser.add_argument('--wall-seconds',type=float,default=900)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--prior-policy',type=Path,help='Retain an earlier validation-selected policy as a candidate')
    parser.add_argument('--reserved-seeds',type=int,nargs='+',default=list(RESERVED_SEEDS))
    args=parser.parse_args()
    if args.steps<1 or args.wall_seconds<=0 or (not args.resume and min(args.teacher_decisions,args.teacher_updates)<1):
        parser.error('Positive budgets required')
    args.output.mkdir(parents=True,exist_ok=False)
    trainer=Trainer.load(args.resume) if args.resume else Trainer(Config(seed=args.seed))
    protocol=dict(config=asdict(trainer.config),rl_target_decisions=args.steps,teacher_decisions=args.teacher_decisions,
        teacher_updates=args.teacher_updates,validation_seeds=VALIDATION_SEEDS,reserved_seeds=args.reserved_seeds,
        resumed_from=str(args.resume) if args.resume else None,prior_policy=str(args.prior_policy) if args.prior_policy else None,
        evaluation_start_levels=EVAL_LEVELS,training_seed_range=[0,10000],selection='Mean observed lines across all six timing-valid validation games; cutoffs remain censored',
        action_scope='Bounded held-DAS navigation programs; three-frame reaction; no RNG access or pose setters',
        reward='0.1 + lines^2 - 10*death - 0.05*delta aggregate height - 0.3*delta holes - 0.02*delta bumpiness',
        scope='First single-training-seed NES pilot on provisional simulator, not a historical or original-ROM strength claim',
        source_sha256={name:hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest() for name in ('nes/engine.py','nes/actions.py','nes/learn.py')})
    write_json(args.output/'protocol.json',protocol)
    started=time.perf_counter()
    def status(phase,count=None):
        row=dict(state='running',pid=os.getpid(),phase=phase,progress=count,rl_decisions=trainer.step,
            actual_placements=trainer.placements,teacher_decisions=trainer.teacher_decisions,
            supervised_updates=trainer.supervised_updates,elapsed_seconds=time.perf_counter()-started,
            loss=trainer.last_loss,replay_mib=trainer.replay.bytes/2**20,
            recent_mean_training_lines=float(np.mean([r['lines'] for r in trainer.history[-100:]])) if trainer.history else None)
        write_json(args.output/'status.json',row)
        publish_progress(row)
        with (args.output/'training.jsonl').open('a',encoding='utf-8') as stream: stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    evaluations=[]
    if not args.resume:
        trainer.warm_start(args.teacher_decisions,args.teacher_updates,status)
        trainer.export(args.output/'policy-warm.pt','teacher-imitation')
        trainer.save(args.output/'checkpoint-warm.pt')
        status('validation-warm')
        evaluations.append(dict(snapshot='policy-warm.pt',result=evaluate(trainer.config,trainer.online)))
    else:
        trainer.export(args.output/'policy-resume.pt','double-dqn')
        status('validation-resume')
        evaluations.append(dict(snapshot='policy-resume.pt',result=evaluate(trainer.config,trainer.online)))
    if args.prior_policy:
        import shutil
        prior_config,prior_model,_=load_policy(args.prior_policy)
        if asdict(prior_config)!=asdict(trainer.config): raise ValueError('Prior policy configuration differs')
        shutil.copy2(args.prior_policy,args.output/'policy-prior-best.pt')
        evaluations.append(dict(snapshot='policy-prior-best.pt',result=evaluate(prior_config,prior_model)))
    write_json(args.output/'validation.json',evaluations)
    training_started=time.perf_counter()
    start_step=trainer.step
    try:
        while trainer.step<args.steps and time.perf_counter()-training_started<args.wall_seconds and not (args.output/'STOP').exists():
            trainer.advance()
            if trainer.step%500==0: status('double-dqn')
            if trainer.step%5000==0:
                name=f'policy-{trainer.step}.pt'
                trainer.save(args.output/f'checkpoint-{trainer.step}.pt')
                trainer.export(args.output/name,'double-dqn')
                status('validation',trainer.step)
                evaluations.append(dict(snapshot=name,result=evaluate(trainer.config,trainer.online)))
                write_json(args.output/'validation.json',evaluations)
    except KeyboardInterrupt:
        status('interrupted')
    trainer.save(args.output/'latest.pt')
    trainer.export(args.output/'policy-final.pt','double-dqn')
    if not evaluations:
        evaluations.append(dict(snapshot='policy-final.pt',result=evaluate(trainer.config,trainer.online)))
    best=max(evaluations,key=lambda row:row['result']['selection_score'])
    config,model,metadata=load_policy(args.output/best['snapshot'])
    import shutil
    shutil.copy2(args.output/best['snapshot'],args.output/'policy-best.pt')
    status('reserved-comparison')
    comparisons={}
    for policy in ('random','heuristic','learned'):
        comparisons[policy]=evaluate(config,model,policy,seeds=args.reserved_seeds,trace_dir=args.output/'reserved'/policy)
    result=dict(state='finished',rl_decisions=trainer.step,additional_decisions=trainer.step-start_step,
        teacher_decisions=trainer.teacher_decisions,supervised_updates=trainer.supervised_updates,
        actual_placements=trainer.placements,selected_snapshot=best['snapshot'],selected_stage=metadata['stage'],
        validation=evaluations,reserved=comparisons,elapsed_seconds=time.perf_counter()-started,
        stopped_by='target_decisions' if trainer.step>=args.steps else 'resource_or_interrupt',
        scope=protocol['scope'])
    write_json(args.output/'results.json',result)
    write_json(args.output/'status.json',{k:v for k,v in result.items() if k not in ('validation','reserved')})
    publish_progress(dict(state='finished',pid=os.getpid(),phase='pilot-complete',rl_decisions=trainer.step,
        teacher_decisions=trainer.teacher_decisions,loss=trainer.last_loss))
    print(json.dumps({k:v for k,v in result.items() if k not in ('validation','reserved')}),flush=True)


if __name__=='__main__': main()
