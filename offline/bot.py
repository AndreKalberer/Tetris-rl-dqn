"""Reachable path search and timed primitive-input execution for the offline game."""
from collections import deque
from dataclasses import dataclass
import time

import numpy as np

from dqn.candidates import Candidates, VisibleState, batch_board_features, board_features, one_hot
from dqn.lookahead import choose_lookahead
from dqn.train_candidates import values
from tetris_macro_env import TetrisPlacementEnv, compute_features
from offline.engine import advance, collides, locked_board, spawn_pose


@dataclass
class Planned:
    candidates: Candidates
    paths: list
    full_boards: list
    visited_nodes: int


class RewardContext:
    placement_reward=TetrisPlacementEnv.placement_reward

    def __init__(self,board,profile):
        for name in vars(TetrisPlacementEnv):
            if name.isupper():
                setattr(self,name,getattr(TetrisPlacementEnv,name))
        TetrisPlacementEnv._apply_reward_profile(self,profile)
        self._board_h=board.shape[0]
        self._prev_agg_h,self._prev_holes,self._prev_bump,self._prev_max_h=compute_features(board)


def control_edge(board,piece,pose,action,rules):
    accepted=True
    for frame in range(rules.control_frames):
        pose,locked,ok=advance(board,piece,pose,action if frame==0 else 'none',rules)
        accepted=accepted and ok
        if locked:
            return pose,True,accepted
    return pose,False,accepted


def enumerate_paths(game,start_pose,*,board_input=True,max_commands=12):
    """Enumerate a subset of reachable locks, each with a replayable timed path.

    Earliest pose visits bound the search. It is deliberately not a proof that
    every possible timing-dependent tuck is enumerated. No future RNG is read.
    """
    rules=game.rules
    visible_queue=list(game.queue)[:5]
    previous=board_features((game.visible!=0).astype(np.uint8))
    boards,full_boards,lines,terminals,contexts,paths,states=[],[],[],[],[],[],[]
    visited_count=0
    seen_results=set()
    for hold in range(1 if game.hold_used else 2):
        piece=game.piece if not hold else (game.held if game.held>=0 else visible_queue[0])
        held=game.piece if hold else game.held
        consume=2 if hold and game.held<0 else 1
        future=visible_queue[consume-1]
        rest=visible_queue[consume:]+[-1]*consume
        root=start_pose
        prefix=()
        if hold:
            root=spawn_pose(piece,rules)
            if collides(game.board,piece,root):
                continue
            for _ in range(rules.control_frames):
                root,locked,_=advance(game.board,piece,root,'none',rules)
                if locked:
                    break
            if locked:
                continue
            prefix=('hold',)
        context=np.concatenate([one_hot(future),*[one_hot(p) for p in rest],one_hot(held)])
        frontier=deque([(root,prefix)])
        seen_poses={(root.x,root.y,root.rotation)}
        while frontier:
            pose,path=frontier.popleft()
            visited_count+=1
            landing,_,_=advance(game.board,piece,pose,'hard',rules)
            result,cleared,terminal=locked_board(game.board,piece,landing,rules)
            terminal=terminal or collides(result,future,spawn_pose(future,rules))
            key=(result.tobytes(),held,future,tuple(rest))
            if key not in seen_results:
                seen_results.add(key)
                board=(result[rules.hidden_rows:]!=0).astype(np.uint8)
                boards.append(board)
                full_boards.append(result)
                lines.append(cleared)
                terminals.append(terminal)
                contexts.append(context)
                paths.append(path+('hard',))
                states.append(VisibleState(board,future,tuple(rest),held))
            # Bound planning cost without relaxing physics. Soft-drop/tuck search
            # is optional future work; all selected locks still use legal paths.
            if len(path)-len(prefix)>=max_commands:
                continue
            for action in ('left','right','cw','ccw'):
                moved,locked,ok=control_edge(game.board,piece,pose,action,rules)
                key=(moved.x,moved.y,moved.rotation)
                if ok and not locked and key not in seen_poses:
                    seen_poses.add(key)
                    frontier.append((moved,path+(action,)))
    if not boards:
        raise RuntimeError('No reachable lock found')
    boards=np.asarray(boards)
    lines=np.asarray(lines)
    terminals=np.asarray(terminals,dtype=bool)
    parts=[batch_board_features(boards),np.broadcast_to(previous,(len(boards),len(previous))),
           np.asarray(contexts),np.column_stack((lines/4,terminals))]
    if board_input:
        parts.insert(0,boards.reshape(len(boards),-1))
    candidates=Candidates(np.arange(len(boards)),np.concatenate(parts,axis=1).astype(np.float32),
                          boards,lines,terminals,tuple(states))
    return Planned(candidates,paths,full_boards,visited_count)


class Bot:
    def __init__(self,config,model,beam=1):
        self.config,self.model,self.beam=config,model,beam
        self.pending=deque()
        self.delay=self.cooldown=0
        self.last_placement=-1
        self.plans=self.overruns=self.no_plan=0
        self.planning_seconds=0.0
        self.expected_board=None
        self.path_mismatches=0

    def command(self,game):
        if game.placements!=self.last_placement:
            if self.expected_board is not None and not np.array_equal(self.expected_board,game.board):
                self.path_mismatches+=1
            self.expected_board=None
            self.last_placement=game.placements
            self.pending.clear()
            self.delay=self.cooldown=0
        if not self.pending and self.delay==0:
            predicted=game.pose
            for _ in range(game.rules.reaction_frames):
                predicted,locked,_=advance(game.board,game.piece,predicted,'none',game.rules)
                if locked:
                    self.no_plan+=1
                    # Wait legally for automatic lock, then plan the next piece.
                    return 'none'
            started=time.perf_counter()
            planned=enumerate_paths(game,predicted,board_input=self.config.board_input)
            if self.beam==1:
                selected=int(values(self.model,planned.candidates.features).argmax())
            else:
                selected,_=choose_lookahead(RewardContext((game.visible!=0).astype(np.uint8),self.config.reward_profile),
                    planned.candidates,self.model,gamma=self.config.gamma,beam=self.beam,
                    board_input=self.config.board_input)
            elapsed=time.perf_counter()-started
            self.plans+=1
            self.planning_seconds+=elapsed
            if elapsed>game.rules.reaction_frames/game.rules.fps:
                self.overruns+=1
                # Discard stale plans instead of granting extra thinking time.
                self.delay=max(1,int(np.ceil(elapsed*game.rules.fps)))
                return 'none'
            self.pending=deque(planned.paths[selected])
            self.expected_board=planned.full_boards[selected]
            self.delay=game.rules.reaction_frames
        if self.delay:
            self.delay-=1
            return 'none'
        if self.cooldown:
            self.cooldown-=1
            return 'none'
        if not self.pending:
            return 'none'
        self.cooldown=game.rules.control_frames-1
        return self.pending.popleft()
