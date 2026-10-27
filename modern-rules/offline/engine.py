"""Deterministic 60 Hz offline game. All active-piece changes use primitive inputs.

SRS-style shape/kick conventions, seven-bag, five visible next pieces, one hold per
placement. Our exact timing/top-out rules are documented in offline/RULES.md;
this is an independent ruleset, not an official Guideline certification.
"""
from collections import deque
from dataclasses import dataclass, replace
from functools import lru_cache
import random

import numpy as np

NAMES = ('I','O','T','S','Z','J','L')  # Matches the research network encoding.
SPAWN = (
    ((0,0,0,0),(1,1,1,1),(0,0,0,0),(0,0,0,0)),
    ((1,1),(1,1)),
    ((0,1,0),(1,1,1),(0,0,0)),
    ((0,1,1),(1,1,0),(0,0,0)),
    ((1,1,0),(0,1,1),(0,0,0)),
    ((1,0,0),(1,1,1),(0,0,0)),
    ((0,0,1),(1,1,1),(0,0,0)),
)
# Kick offsets are expressed in x-right/y-up; board coordinates use y-down.
JLSTZ = {
    (0,1):((0,0),(-1,0),(-1,1),(0,-2),(-1,-2)),
    (1,0):((0,0),(1,0),(1,-1),(0,2),(1,2)),
    (1,2):((0,0),(1,0),(1,-1),(0,2),(1,2)),
    (2,1):((0,0),(-1,0),(-1,1),(0,-2),(-1,-2)),
    (2,3):((0,0),(1,0),(1,1),(0,-2),(1,-2)),
    (3,2):((0,0),(-1,0),(-1,-1),(0,2),(-1,2)),
    (3,0):((0,0),(-1,0),(-1,-1),(0,2),(-1,2)),
    (0,3):((0,0),(1,0),(1,1),(0,-2),(1,-2)),
}
I_KICKS = {
    (0,1):((0,0),(-2,0),(1,0),(-2,-1),(1,2)),
    (1,0):((0,0),(2,0),(-1,0),(2,1),(-1,-2)),
    (1,2):((0,0),(-1,0),(2,0),(-1,2),(2,-1)),
    (2,1):((0,0),(1,0),(-2,0),(1,-2),(-2,1)),
    (2,3):((0,0),(2,0),(-1,0),(2,1),(-1,-2)),
    (3,2):((0,0),(-2,0),(1,0),(-2,-1),(1,2)),
    (3,0):((0,0),(1,0),(-2,0),(1,-2),(-2,1)),
    (0,3):((0,0),(-1,0),(2,0),(-1,2),(2,-1)),
}
ACTIONS = ('none','left','right','cw','ccw','soft','hard','hold')


@dataclass(frozen=True)
class Rules:
    version: str = 'offline-v1'
    width: int = 10
    visible_rows: int = 20
    hidden_rows: int = 4
    fps: int = 60
    gravity_frames: int = 30
    lock_frames: int = 30
    max_lock_resets: int = 15
    control_frames: int = 3
    reaction_frames: int = 30
    target_lines: int | None = 40


@dataclass(frozen=True)
class Pose:
    x: int
    y: int
    rotation: int = 0
    gravity: int = 0
    lock: int = 0
    resets: int = 0


@lru_cache(maxsize=28)
def cells(piece, rotation):
    matrix = np.asarray(SPAWN[piece],dtype=np.uint8)
    if piece != 1:
        matrix = np.rot90(matrix,-rotation)
    ys,xs = np.nonzero(matrix)
    return tuple(zip(xs.tolist(),ys.tolist()))


def spawn_pose(piece, rules):
    return Pose((rules.width-len(SPAWN[piece]))//2,rules.hidden_rows-2)


def collides(board, piece, pose):
    h,w = board.shape
    return any(pose.x+dx<0 or pose.x+dx>=w or pose.y+dy<0 or pose.y+dy>=h
               or board[pose.y+dy,pose.x+dx] for dx,dy in cells(piece,pose.rotation))


def rotate(board,piece,pose,direction):
    if piece==1:
        return pose
    target=(pose.rotation+direction)%4
    for dx,dy in (I_KICKS if piece==0 else JLSTZ)[pose.rotation,target]:
        trial=replace(pose,x=pose.x+dx,y=pose.y-dy,rotation=target)
        if not collides(board,piece,trial):
            return trial
    return pose


def advance(board,piece,pose,action,rules):
    """One physics frame. Returns pose, lock-now, and input-accepted."""
    if action not in ACTIONS or action=='hold':
        raise ValueError('Hold is handled at game level; other inputs must be known')
    if action=='hard':
        while not collides(board,piece,replace(pose,y=pose.y+1)):
            pose=replace(pose,y=pose.y+1)
        return pose,True,True
    grounded=collides(board,piece,replace(pose,y=pose.y+1))
    moved=pose
    if action in ('left','right','soft'):
        trial=replace(pose,x=pose.x+(-1 if action=='left' else 1 if action=='right' else 0),
                      y=pose.y+(action=='soft'))
        if not collides(board,piece,trial):
            moved=trial
    elif action in ('cw','ccw'):
        moved=rotate(board,piece,pose,1 if action=='cw' else -1)
    accepted=action=='none' or moved!=pose
    if grounded and moved!=pose and action in ('left','right','cw','ccw') and pose.resets<rules.max_lock_resets:
        moved=replace(moved,lock=0,resets=pose.resets+1)
    gravity=moved.gravity+1
    if gravity>=rules.gravity_frames:
        gravity=0
        trial=replace(moved,y=moved.y+1)
        if not collides(board,piece,trial):
            moved=trial
    moved=replace(moved,gravity=gravity)
    if collides(board,piece,replace(moved,y=moved.y+1)):
        moved=replace(moved,lock=moved.lock+1)
    else:
        moved=replace(moved,lock=0)
    return moved,moved.lock>=rules.lock_frames,accepted


def locked_board(board,piece,pose,rules):
    result=board.copy()
    for dx,dy in cells(piece,pose.rotation):
        if result[pose.y+dy,pose.x+dx]:
            raise ValueError('Cannot lock overlapping cells')
        result[pose.y+dy,pose.x+dx]=piece+1
    full=result.all(axis=1)
    lines=int(full.sum())
    if lines:
        result=np.concatenate((np.zeros((lines,rules.width),dtype=np.uint8),result[~full]))
    return result,lines,bool(result[:rules.hidden_rows].any())


class Game:
    def __init__(self,seed=0,rules=Rules()):
        self.rules=rules
        self.rng=random.Random(seed)
        self.seed=seed
        self.bag=[]
        self.queue=deque(self.draw() for _ in range(5))
        self.board=np.zeros((rules.hidden_rows+rules.visible_rows,rules.width),dtype=np.uint8)
        self.held=-1
        self.hold_used=False
        self.frames=self.lines=self.placements=self.inputs=self.rejected_inputs=0
        self.over=self.won=False
        self.piece=-1
        self.pose=Pose(0,0)
        self.spawn(self.take_next())

    def draw(self):
        if not self.bag:
            self.bag=list(range(7))
            self.rng.shuffle(self.bag)
        return self.bag.pop()

    def take_next(self):
        piece=self.queue.popleft()
        self.queue.append(self.draw())
        return piece

    def spawn(self,piece):
        self.piece=piece
        self.pose=spawn_pose(piece,self.rules)
        self.over=bool(collides(self.board,piece,self.pose))

    @property
    def visible(self):
        return self.board[self.rules.hidden_rows:]

    def tick(self,action='none'):
        if self.over or self.won:
            return dict(accepted=False,locked=False,lines=0)
        if action not in ACTIONS:
            raise ValueError(f'Unknown input: {action}')
        self.frames+=1
        accepted=True
        if action!='none':
            self.inputs+=1
        if action=='hold':
            if self.hold_used:
                accepted=False
            else:
                previous=self.piece
                self.spawn(self.held if self.held>=0 else self.take_next())
                self.held=previous
                self.hold_used=True
            action='none'
        if self.over:
            return dict(accepted=accepted,locked=False,lines=0)
        self.pose,lock_now,movement_accepted=advance(self.board,self.piece,self.pose,action,self.rules)
        accepted=accepted and movement_accepted
        if not accepted:
            self.rejected_inputs+=1
        cleared=0
        if lock_now:
            self.board,cleared,self.over=locked_board(self.board,self.piece,self.pose,self.rules)
            self.placements+=1
            self.lines+=cleared
            self.won=self.rules.target_lines is not None and self.lines>=self.rules.target_lines
            self.hold_used=False
            if not self.over and not self.won:
                self.spawn(self.take_next())
        return dict(accepted=bool(accepted),locked=bool(lock_now),lines=cleared)
