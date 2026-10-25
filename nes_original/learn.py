"""Original-game candidate Double DQN; native transitions in the selected core."""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from nes.actions import ActionRules,structural
from nes_original.env import OriginalGame, ROM_SHA256
from nes_original.planner import proposals
from nes_original.feedback import FeedbackController

LEGACY_KIND = 'original-ntsc-a-feedback-double-dqn-v2'
KIND = 'original-ntsc-a-feedback-double-dqn-v3'
VALIDATION_SEEDS = (1001,1013)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)


def save_failure(game,folder,error,decisions):
    """Preserve the original error even when a dead core cannot take a capture."""
    folder=Path(folder)
    try:
        game.save_replay(folder/'failure-replay.json')
    except Exception as capture_error:
        diagnostic=dict(rom_sha256=ROM_SHA256,input_hex=game.inputs.hex(),
            backend=type(game).__name__,capture_error=repr(capture_error),
            exact_replay_verified=False)
        try:
            diagnostic['fingerprint']=game.fingerprint()
        except Exception as fingerprint_error:
            diagnostic['fingerprint_error']=repr(fingerprint_error)
        write_json(folder/'failure-raw-inputs.json',diagnostic)
    write_json(folder/'failure.json',dict(error=repr(error),decisions=decisions))


@dataclass
class Config:
    seed: int = 0
    capacity: int = 3000
    batch_size: int = 64
    learning_starts: int = 128
    teacher_decisions: int = 256
    train_every: int = 4
    target_every: int = 500
    gamma: float = 0.99
    learning_rate: float = 0.00005
    epsilon_start: float = 0.05
    epsilon_final: float = 0.01
    epsilon_steps: int = 10000
    threads: int = 2
    tap_period: int = 0
    prior: str = 'heuristic'
    exploration_top_k: int = 0

    def __post_init__(self):
        if self.capacity < max(self.batch_size,self.learning_starts) or min(
                self.batch_size,self.train_every,self.target_every,self.epsilon_steps,self.threads) < 1:
            raise ValueError('Invalid replay/update settings')
        ActionRules(horizontal_period=self.tap_period)
        if self.prior not in ('heuristic','lookahead') or self.exploration_top_k<0:
            raise ValueError('Invalid prior/exploration mode')


def features(plan,prior_values=None):
    # Full predicted board + visible context + fixed heuristic prior. These
    # describe proposals; the reward and next state always come from original code.
    targets = np.zeros((len(plan.paths),20),dtype=np.float32)
    for i,state in enumerate(plan.actions.controls):
        targets[i,0] = state.x/9
        targets[i,1+state.orientation] = 1
    prior=plan.actions.heuristic if prior_values is None else prior_values
    return np.column_stack((plan.actions.features,targets,
        np.clip(prior/10,-10,2))).astype(np.float32)


class FeatureEncoder:
    def __init__(self,config):
        self.lookahead=None
        if config.prior=='lookahead':
            from nes_original.lookahead import NextPiece
            self.lookahead=NextPiece(34,config.tap_period,jit=True)

    def __call__(self,obs,plan):
        prior=self.lookahead.action_values(obs,plan) if self.lookahead else None
        return features(plan,prior)


class Network(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.residual = nn.Sequential(nn.Linear(dim,128),nn.ReLU(),
            nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, x):
        return x[:,-1] + self.residual(x).squeeze(-1)


@torch.no_grad()
def values(model, x):
    return model(torch.from_numpy(np.asarray(x,dtype=np.float32))).numpy()


def potential(board):
    s = structural(board[None])[0]
    return float(-4*s[0]-50*s[1]-4*s[2])


def reward(before, after, gamma=0.99):
    lines = after.lines-before.lines
    # Dense survival/line/Tetris/score reward plus discounted board potential.
    # Explicit experimental shaping; not a guarantee of optimal NES play.
    won = after.level >= 30
    raw = .02 + .1*lines**2 + max(0,after.score-before.score)/100000 - 5*after.over + 5*won
    return float(raw + (0 if after.over or won else gamma*potential(after.board))-potential(before.board))


def execute(game, plan, selected, *, tap_period=0):
    """Native target feedback, not an assertion of exact predicted afterstates."""
    controller = FeedbackController(fixed_reaction=True,tap_period=tap_period)
    controller.start_plan(game.observe(),plan,selected)
    for _ in range(600):
        before = game.observe()
        after = game.step(controller.command(before))
        controller.transition(before,after)
        if controller.stage is None:
            return after,controller.stats()
    raise RuntimeError('Original-game action exceeded 600 native frames')


class Trainer:
    def __init__(self, config, *, game_factory=OriginalGame):
        self.config = config
        self.game_factory = game_factory
        self.rules=ActionRules(horizontal_period=config.tap_period)
        self.encoder=FeatureEncoder(config)
        torch.set_num_threads(config.threads)
        torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)
        self.game = game_factory()
        self.step = self.updates = self.episodes = self.placements = 0
        self.position = 0
        self.replay = []
        self.history = []
        self.audit_totals = {key:0 for key in ('target_misses','proposal_board_differences',
                                               'board_audit_failures','top_boundary_locks','piece_type_differences')}
        self.loss = None
        self.reset_game()
        self.dim = self.encoder(self.game.observe(),self.plan).shape[1]
        self.online,self.target = Network(self.dim),Network(self.dim)
        self.online.feature_config=config
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(),lr=config.learning_rate)

    def reset_game(self):
        self.game.reset(start_level=18,seed=int(self.rng.integers(0,512)))
        self.plan = proposals(self.game.observe(),self.rules,board_input=True)
        self.episode_decisions = 0

    def execute(self,plan,selected):
        after,stats = execute(self.game,plan,selected,tap_period=self.config.tap_period)
        for key in self.audit_totals:
            self.audit_totals[key] += stats[key]
        return after,stats

    def ready(self,replay_dir=None):
        # A queued input can already make lock inevitable. Execute it without
        # inventing an agent choice or adding a synthetic transition to replay.
        for _ in range(10):
            if self.plan is not None:
                return
            old_placed = self.game.observe().placements
            after,_ = self.execute(None,0)
            self.placements += after.placements-old_placed
            if after.over:
                self.finish_episode(replay_dir)
                self.reset_game()
            else:
                self.plan = proposals(after,self.rules,board_input=True)
        raise RuntimeError('Repeated forced-lock states')

    def finish_episode(self,replay_dir=None):
        o = self.game.observe()
        if replay_dir:
            record = self.game.replay()
            # The reference core has a bounded host-idle watchdog. End the
            # completed episode before spending time reconstructing its proof.
            self.game.close()
            with self.game_factory() as other:
                other.load_replay(record)
            write_json(Path(replay_dir)/f'episode-{self.episodes:05d}.json',record)
        self.history.append(dict(seed=self.game.seed,level=o.level,lines=o.lines,
            score=o.score,placements=o.placements,decisions=self.episode_decisions,
            completed_level29=o.level>=30,over=o.over,exact_ram_rgb_replay=bool(replay_dir)))
        self.episodes += 1

    def advance(self, *, replay_dir=None):
        self.ready(replay_dir)
        c = self.config
        x = self.encoder(self.game.observe(),self.plan)
        epsilon = c.epsilon_start+min(1,self.step/c.epsilon_steps)*(c.epsilon_final-c.epsilon_start)
        if self.step < c.teacher_decisions:
            selected = int(x[:,-1].argmax())
        elif self.rng.random() < epsilon:
            pool=np.argsort(-x[:,-1],kind='stable')[:c.exploration_top_k] if c.exploration_top_k else np.arange(len(x))
            selected = int(self.rng.choice(pool))
        else:
            selected = int(values(self.online,x).argmax())
        before = self.game.observe()
        after,stats = self.execute(self.plan,selected)
        self.episode_decisions += 1
        self.plan = None if after.over else proposals(after,self.rules,board_input=True)
        # Resolve forced states in the same real transition until a new genuine
        # choice exists; reward uses the resulting actual counters and board.
        for _ in range(10):
            if after.over or self.plan is not None:
                break
            after,_ = self.execute(None,0)
            self.plan = None if after.over else proposals(after,self.rules,board_input=True)
        else:
            raise RuntimeError('Forced-state continuation exceeded bound')
        self.placements += after.placements-before.placements
        done = after.over or after.level >= 30
        next_x = np.empty((0,self.dim),dtype=np.float16) if done else self.encoder(after,self.plan).astype(np.float16)
        item = (x[selected].astype(np.float16),reward(before,after,c.gamma),next_x,done)
        if len(self.replay) < c.capacity:
            self.replay.append(item)
        else:
            self.replay[self.position] = item
        self.position = (self.position+1)%c.capacity
        self.step += 1
        if len(self.replay) >= max(c.learning_starts,c.batch_size) and self.step%c.train_every == 0:
            self.update()
        if self.step%c.target_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        if done or after.level >= 30:
            self.finish_episode(replay_dir)
            self.reset_game()
        return stats

    def update(self):
        c = self.config
        batch = [self.replay[int(i)] for i in self.rng.choice(len(self.replay),c.batch_size,replace=False)]
        targets = torch.tensor([item[1] for item in batch],dtype=torch.float32)
        with torch.no_grad():
            active = [(i,item[2]) for i,item in enumerate(batch) if not item[3]]
            if active:
                nx = torch.from_numpy(np.concatenate([x for _,x in active]).astype(np.float32))
                online,target = self.online(nx),self.target(nx)
                offset = 0
                for i,x in active:
                    chosen = online[offset:offset+len(x)].argmax()+offset
                    targets[i] += c.gamma*target[chosen]
                    offset += len(x)
            targets.clamp_(-20,20)
        predicted = self.online(torch.from_numpy(np.stack([item[0] for item in batch]).astype(np.float32)))
        loss = nn.functional.smooth_l1_loss(predicted,targets)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite original-game TD loss')
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(),1)
        self.optimizer.step()
        self.loss = float(loss.detach())
        self.updates += 1

    def save(self,path):
        payload = dict(kind=KIND,rom_sha256=ROM_SHA256,config=asdict(self.config),dim=self.dim,
            torch_version=str(torch.__version__),
            online=self.online.state_dict(),target=self.target.state_dict(),optimizer=self.optimizer.state_dict(),
            replay=self.replay,position=self.position,step=self.step,updates=self.updates,
            episodes=self.episodes,placements=self.placements,history=self.history,loss=self.loss,
            episode_decisions=self.episode_decisions,rng=self.rng.bit_generator.state,
            audit_totals=self.audit_totals,torch_rng=torch.get_rng_state(),game=self.game.replay())
        path = Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        temp = path.with_suffix(path.suffix+'.tmp')
        torch.save(payload,temp)
        temp.replace(path)

    @classmethod
    def load(cls,path, *, game_factory=OriginalGame):
        data = torch.load(path,map_location='cpu',weights_only=False)
        if data['kind'] not in (KIND,LEGACY_KIND) or data['rom_sha256'] != ROM_SHA256:
            raise ValueError('Not an original-game training checkpoint')
        if data.get('torch_version',str(torch.__version__)) != str(torch.__version__):
            raise ValueError('Use the recorded PyTorch version for exact trainer resume')
        obj = cls(Config(**data['config']),game_factory=game_factory)
        for name in ('online','target','optimizer'):
            getattr(obj,name).load_state_dict(data[name])
        for name in ('replay','position','step','updates','episodes','placements','history','loss','episode_decisions'):
            setattr(obj,name,data[name])
        try:
            obj.game.load_replay(data['game'])
        except Exception:
            obj.close()
            raise
        obj.audit_totals = data.get('audit_totals',obj.audit_totals)
        obj.plan = proposals(obj.game.observe(),obj.rules,board_input=True)
        obj.rng.bit_generator.state = data['rng']
        torch.set_rng_state(data['torch_rng'])
        return obj

    def export(self,path):
        torch.save(dict(kind=KIND,rom_sha256=ROM_SHA256,dim=self.dim,online=self.online.state_dict(),
            config=asdict(self.config),torch_version=str(torch.__version__),decisions=self.step,updates=self.updates,
            training_backend=type(self.game).__name__,game_identity=getattr(self.game,'identity',None)),path)

    def close(self):
        self.game.close()

    def evaluate(self,model,output,**kwargs):
        """Park the training core while a separate core evaluates the policy."""
        record=self.game.replay()
        self.game.close()
        try:
            return evaluate(model,output,config=self.config,game_factory=self.game_factory,**kwargs)
        finally:
            self.game.load_replay(record)


def load_policy(path):
    data = torch.load(path,map_location='cpu',weights_only=True)
    if data['kind'] not in (KIND,LEGACY_KIND) or data['rom_sha256'] != ROM_SHA256:
        raise ValueError('Expected an original-game policy')
    torch.set_num_threads(data['config']['threads'])
    model = Network(data['dim'])
    model.feature_config=Config(**data['config'])
    model.load_state_dict(data['online'])
    model.eval()
    values(model,np.zeros((34,data['dim']),dtype=np.float32))
    return model,data


ENDLESS = 10**9  # target level that is never reached: play until the game ends


def evaluate(model, output, *, seeds=VALIDATION_SEEDS, runtime=False, max_frames=120000,config=None,
             game_factory=OriginalGame,target_level=30,context_selector=None,no_softdrop_at_1g=False,
             stop_on_success=False,no_softdrop_levels=(),nonactive_input=0,
             stall_rescue_input=0,stall_rescue_after=120,
             stall_rescue_fallback_input=0,stall_rescue_primary_frames=8):
    """Reach a native level from18 at a settled controller boundary and replay.

    target_level255 means reaching255, not clearing it or certifying rebirth.
    It requires the first-cycle line window3290..3293 and valid per-piece
    original counter transitions; replaying corrupted RAM cannot certify it.
    Targets above 255 are absolute levels across rebirths (level 255 -> 0):
    256*cycles+level from audited wraps; 511 = level 255 on the second lap.
    ENDLESS (a huge target) plays until the game ends or the frame budget runs out.
    max_frames bounds gameplay frames after menu entry, not reconstruction time.
    context_selector optionally replaces feature/model selection, using only
    the same observation and approximate proposal interface as the defaults.
    """
    if isinstance(target_level,bool) or not isinstance(target_level,int) or not 19<=target_level<=ENDLESS:
        raise ValueError('target_level must be an integer from19 (absolute levels count rebirths; '
                         f'{ENDLESS} = endless) for a level18 start')
    if isinstance(max_frames,bool) or not isinstance(max_frames,int) or max_frames<1:
        raise ValueError('max_frames must be a positive integer gameplay-frame budget')
    if context_selector is not None and (model is not None or not callable(context_selector)):
        raise ValueError('context_selector must be callable and cannot be combined with a model')
    if not isinstance(no_softdrop_at_1g,bool):
        raise ValueError('no_softdrop_at_1g must be a boolean control-profile choice')
    if not isinstance(stop_on_success,bool):
        raise ValueError('stop_on_success must be a boolean')
    try:
        no_softdrop_levels=tuple(no_softdrop_levels)
    except TypeError as exc:
        raise ValueError('no_softdrop_levels must be an iterable of levels') from exc
    from nes_original.mesen import SuspectedCrash
    from nes_original.progression import MarathonProgress
    config=config or getattr(model,'feature_config',None) or Config()
    encoder=FeatureEncoder(config) if context_selector is None else None
    rows = []
    for seed in seeds:
        controller = FeedbackController(fixed_reaction=not runtime,tap_period=config.tap_period,
            no_softdrop_at_1g=no_softdrop_at_1g,
            no_softdrop_levels=no_softdrop_levels,
            nonactive_input=nonactive_input,
            stall_rescue_input=stall_rescue_input,stall_rescue_after=stall_rescue_after,
            stall_rescue_fallback_input=stall_rescue_fallback_input,
            stall_rescue_primary_frames=stall_rescue_primary_frames,
            context_selector=context_selector if context_selector is not None else
                ((lambda obs,plan: values(model,encoder(obs,plan)).argmax()) if model else
                 (lambda obs,plan: encoder(obs,plan)[:,-1].argmax())))
        with game_factory() as game:
            o = game.reset(start_level=18,seed=int(seed))
            error = None
            observed_fault = None
            peak_level = o.level
            target_reached = False
            progress = MarathonProgress(18,0,o.placements)
            progress_counters = progress.update(o.level,o.lines,o.placements)
            while not o.over and not target_reached and o.frames < max_frames:
                before = o
                try:
                    o = game.step(controller.command(o))
                    controller.transition(before,o)
                    if controller.stage is None and o.phase==1 and not o.over:
                        try:
                            if o.placements-progress.placements>1:
                                raise ValueError('Missed a settled piece boundary')
                            progress_counters = progress.update(o.level,o.lines,o.placements)
                        except ValueError as exc:
                            raise RuntimeError(f'Progress counter audit failed: {exc}') from exc
                        absolute_level = 256*progress.level_cycles+progress.level
                        peak_level = max(peak_level,absolute_level)
                        if target_level<=255:
                            target_reached = progress.level>=target_level and progress.level_cycles==0 and (
                                target_level!=255 or 3290<=progress.absolute_lines<=3293)
                        else:
                            target_reached = absolute_level>=target_level
                except (AssertionError,RuntimeError) as exc:
                    error = str(exc)
                    if type(exc) is SuspectedCrash:
                        observed_fault = dict(type='SuspectedCrash',reason=str(exc).split(';',1)[0])
                    # Both native backends expose a non-advancing observation.
                    # A transition may fail after a successful frame; retain
                    # that latest cached/native state without stepping again.
                    try:
                        o = game.observe()
                    except Exception:
                        pass
                    break
            path = Path(output)/f'seed-{seed}.json'
            backend=type(game).__name__
            case_path = Path(output)/f'seed-{seed}-case.json'
            row = dict(seed=int(seed),level=o.level,lines=o.lines,score=o.score,backend=backend,
                placements=o.placements,success=False,
                over=o.over,error=error,censored=not o.over and not target_reached and error is None,
                observed_fault=observed_fault,fault_reproduced=False,
                termination_kind=('suspected_original_crash' if observed_fault else
                    'controller_error' if error else 'top_out' if o.over else
                    'target_reached' if target_reached else 'frame_limit'),
                exact_ram_rgb_replay=False,controller=controller.stats(),
                case_state='verification_pending',start_level=18,target_level=target_level,
                target_reached=target_reached,peak_level=peak_level,
                frames=o.frames,max_frames=max_frames,
                progress=progress_counters,
                target_criterion='Audited native counters at a settled active-piece controller boundary')
            # Write gameplay telemetry before screen capture/core shutdown or
            # fresh-process reconstruction can fail or be interrupted.
            write_json(case_path,row)
            try:
                record = game.replay()
                if observed_fault:
                    record['observed_fault'] = observed_fault
                write_json(path,record)
            except Exception as exc:
                row.update(case_state='verification_failed',verification_error=repr(exc),
                           verification_failure_phase='replay_capture')
                write_json(case_path,row)
                write_json(Path(output)/f'seed-{seed}-raw-inputs.json',
                    dict(input_hex=game.inputs.hex(),backend=backend,
                         exact_replay_verified=False,error=error,capture_error=repr(exc)))
                raise
        try:
            with game_factory() as other:
                if observed_fault:
                    other.load_replay(record,expected_fault=observed_fault)
                else:
                    other.load_replay(record)
        except Exception as exc:
            row.update(case_state='verification_failed',verification_error=repr(exc),
                       verification_failure_phase='fresh_replay')
            write_json(case_path,row)
            raise
        row.update(case_state='verified',exact_ram_rgb_replay=True,
                   success=target_reached and not error,fault_reproduced=observed_fault is not None)
        write_json(case_path,row)
        rows.append(row)
        write_json(Path(output)/'results.json',rows)
        if stop_on_success and row['success']:
            break
    write_json(Path(output)/'results.json',rows)
    return rows


def rank(rows):
    if any(row['error'] for row in rows):
        return (-1,-1,-1)
    return (sum(row['success'] for row in rows),float(np.median([r['lines'] for r in rows])),
            float(np.mean([r['score'] for r in rows])))


def cpu_limits(mask=160):
    if os.name == 'nt':
        import ctypes
        kernel = ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        handle = kernel.GetCurrentProcess()
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p,ctypes.c_size_t]
        kernel.SetPriorityClass.argtypes = [ctypes.c_void_p,ctypes.c_uint32]
        if os.cpu_count() < 8:
            mask = 3
        if not kernel.SetProcessAffinityMask(handle,mask) or not kernel.SetPriorityClass(handle,0x4000):
            raise ctypes.WinError(ctypes.get_last_error())
    torch.set_num_interop_threads(1)
    return mask


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--status',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--check-every',type=int,default=2500)
    p.add_argument('--resume',type=Path)
    p.add_argument('--mesen',type=Path,help='Use an existing dedicated portable MesenCE build')
    p.add_argument('--prior',choices=['heuristic','lookahead'],default='heuristic')
    p.add_argument('--tap-period',type=int,default=0)
    p.add_argument('--jit-deps',type=Path)
    p.add_argument('--cpu-mask',type=int,default=160)
    args = p.parse_args()
    if args.jit_deps:
        import sys
        sys.path.insert(0,str(args.jit_deps.resolve()))
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        p.error('Output is not empty; use a new directory or explicitly resume a checkpoint')
    args.output.mkdir(parents=True,exist_ok=True)
    game_factory=OriginalGame
    if args.mesen:
        from nes_original.mesen import MesenGame
        game_factory=lambda:MesenGame(executable=args.mesen,
            session_root=args.output/'reference-sessions',transport='pipe')
    mask = cpu_limits(args.cpu_mask)
    lock = args.output/'RUNNING.lock'
    try:
        descriptor = os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        p.error('Job already has RUNNING.lock; verify its process before resuming')
    with os.fdopen(descriptor,'w',encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    started = time.perf_counter()
    current = None
    finished = []
    def status(state,phase):
        try:
            o = current.game.observe() if current else None
        except Exception:
            o = None
        data = dict(state=state,phase=phase,pid=os.getpid(),time=time.time(),
            wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time(),
            affinity_mask=mask,priority='BelowNormal',threads=2,job=str(args.output),
            seed=current.config.seed if current else None,decisions=current.step if current else 0,
            prior=current.config.prior if current else args.prior,
            horizontal_pulse_period=current.config.tap_period if current else args.tap_period,
            teacher_decisions=min(current.step,current.config.teacher_decisions) if current else 0,
            updates=current.updates if current else 0,loss=current.loss if current else None,
            action_audits=current.audit_totals if current else {},
            total_decisions=sum(row['decisions'] for row in finished)+(current.step if current else 0),
            completed_seeds=finished,
            episodes=current.episodes if current else 0,
            best_training_lines=max((r['lines'] for r in current.history),default=0) if current else 0,
            best_training_level=max((r['level'] for r in current.history),default=18) if current else 18,
            completed_level29_in_training=any(r['completed_level29'] for r in current.history) if current else False,
            current_lines=o.lines if o else None,target='Start18; clear29 and reach30 (240lines)',
            goal_achieved=False,scope='Original NES game training; development metrics, not reserved acceptance')
        write_json(args.status,data)
        write_json(args.output/'status.json',data)
    status('running','initializing')
    protocol_path = args.output/('protocol.json' if not args.resume else f'protocol-resume-{time.time_ns()}.json')
    write_json(protocol_path,dict(kind=KIND,rom_sha256=ROM_SHA256,
        backend='MesenGame' if args.mesen else 'OriginalGame/nes-py',
        training_seeds=args.seeds,steps_per_seed=args.steps,training_menu_wait_range=[0,511],
        validation_menu_wait_seeds=VALIDATION_SEEDS,starting_level=18,target_level=30,
        teacher_rollout_decisions=1024 if args.prior=='lookahead' else 256,
        prior=args.prior,horizontal_pulse_period=args.tap_period,
        learning_starts=1024 if args.prior=='lookahead' else 128,
        exploration_top_k=4 if args.prior=='lookahead' else 0,
        epsilon_start=.01 if args.prior=='lookahead' else .05,
        epsilon_final=.002 if args.prior=='lookahead' else .01,
        training_timing='Fixed logical reaction, actual original-game transitions',
        validation_timing='Fixed logical reaction; selected policy additionally tested with charged runtime latency',
        selected_by='Validation successes then median lines then mean score; no reserved starts used',
        reward='0.02 + 0.1*lines^2 + score_delta/100000 - 5*death + 5*complete_level29 + gamma*Phi(next)-Phi(before)',
        potential='-0.02*sum_heights -0.25*holes -0.02*bumpiness; zero at terminal',
        method='Approximate board/context + target-pose features; configured fixed prior + zero-initial residual Double DQN; actual native feedback next-state/reward',
        action_interface='3 reaction; 48 navigation; configured held-DAS or horizontal pulse period; 4frame rotations; Down to observed lock; stable spawn before next choice',
        limitations='Experimental. Strict preview audit retained separately. Native target misses/prediction differences logged; actual board conservation and exact episode replay audited. No assumed gain.',
        reserved_acceptance='Not run; freeze final protocol before 20 untouched starts'))
    try:
        for seed in args.seeds:
            folder = args.output/f'seed{seed}'
            folder.mkdir(exist_ok=True)
            config=Config(seed=seed,tap_period=args.tap_period,prior=args.prior,
                teacher_decisions=1024 if args.prior=='lookahead' else 256,
                learning_starts=1024 if args.prior=='lookahead' else 128,
                epsilon_start=.01 if args.prior=='lookahead' else .05,
                epsilon_final=.002 if args.prior=='lookahead' else .01,
                exploration_top_k=4 if args.prior=='lookahead' else 0)
            current = Trainer.load(args.resume,game_factory=game_factory) if args.resume else Trainer(config,game_factory=game_factory)
            if current.config.seed != seed:
                raise ValueError('Resume seed does not match requested seed')
            status('running','initial-validation')
            if not args.resume:
                current.export(folder/'policy-0.pt')
                initial = current.evaluate(None,folder/'baseline')
                best = rank(initial)
                current.export(folder/'best.pt')
                write_json(folder/'selection.json',dict(step=0,rank=best,rows=initial))
                current.save(folder/'latest.pt')
            else:
                best = tuple(json.loads((folder/'selection.json').read_text(encoding='utf-8'))['rank'])
            last_history = len(current.history)
            while current.step < args.steps:
                if (args.output/'STOP').exists():
                    current.save(folder/'latest.pt')
                    status('stopped','stop-file')
                    return
                current.advance(replay_dir=folder/'episodes')
                if len(current.history) > last_history:
                    with (folder/'episodes.jsonl').open('a',encoding='utf-8') as stream:
                        for row in current.history[last_history:]:
                            stream.write(json.dumps(row)+'\n')
                    last_history = len(current.history)
                if current.step%25 == 0:
                    status('running','teacher-rollout' if current.step<=current.config.teacher_decisions else 'reinforcement-learning')
                if current.step%500 == 0:
                    current.save(folder/'latest.pt')
                if current.step%args.check_every == 0 or current.step == args.steps:
                    current.save(folder/f'checkpoint-{current.step}.pt')
                    current.export(folder/f'policy-{current.step}.pt')
                    status('running','validation')
                    rows = current.evaluate(current.online,folder/f'validation-{current.step}')
                    if rank(rows) > best:
                        best = rank(rows)
                        current.export(folder/'best.pt')
                        write_json(folder/'selection.json',dict(step=current.step,rank=best,rows=rows))
                    current.save(folder/'latest.pt')
            model,metadata = load_policy(folder/'best.pt')
            status('running','charged-runtime-evaluation')
            runtime = current.evaluate(model,folder/f'selected-runtime-through-{current.step}',runtime=True)
            finished.append(dict(seed=seed,decisions=current.step,updates=current.updates,
                selected_decisions=metadata['decisions'],selected_rank=best,
                best_training_lines=max((r['lines'] for r in current.history),default=0),
                best_training_level=max((r['level'] for r in current.history),default=18),runtime=runtime))
            write_json(args.output/'summary.json',finished)
            current.close()
            current = None
        status('complete','pilot-budget-completed; level29 acceptance not established')
    except Exception as exc:
        if current:
            folder = args.output/f'seed{current.config.seed}'
            save_failure(current.game,folder,exc,current.step)
            # Failure state may be mid-action; keep the last clean training
            # checkpoint for resuming, never save a partial decision as clean.
        status('failed',repr(exc))
        raise
    finally:
        if current:
            current.close()
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
