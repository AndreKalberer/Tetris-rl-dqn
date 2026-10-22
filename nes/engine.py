"""Frame-driven NES A-Type mechanics, independently implemented from references.

Reference revision and remaining ROM-parity gaps are documented in RULES.md.
No ROM, Nintendo graphics, or disassembled program code is included here.
"""
from dataclasses import asdict, dataclass, replace
from enum import IntEnum, IntFlag
import hashlib
import json

import numpy as np


FPS=60.0988
NAMES=('T','J','Z','O','S','L','I')
# Relative (x,y) cell coordinates of the 19 Nintendo orientation IDs.
CELLS=(
    ((-1,0),(0,0),(1,0),(0,-1)),
    ((0,-1),(0,0),(1,0),(0,1)),
    ((-1,0),(0,0),(1,0),(0,1)),
    ((0,-1),(-1,0),(0,0),(0,1)),
    ((0,-1),(0,0),(-1,1),(0,1)),
    ((-1,-1),(-1,0),(0,0),(1,0)),
    ((0,-1),(1,-1),(0,0),(0,1)),
    ((-1,0),(0,0),(1,0),(1,1)),
    ((-1,0),(0,0),(0,1),(1,1)),
    ((1,-1),(0,0),(1,0),(0,1)),
    ((-1,0),(0,0),(-1,1),(0,1)),
    ((0,0),(1,0),(-1,1),(0,1)),
    ((0,-1),(0,0),(1,0),(1,1)),
    ((0,-1),(0,0),(0,1),(1,1)),
    ((-1,0),(0,0),(1,0),(-1,1)),
    ((-1,-1),(0,-1),(0,0),(0,1)),
    ((1,-1),(-1,0),(0,0),(1,0)),
    ((0,-2),(0,-1),(0,0),(0,1)),
    ((-2,0),(-1,0),(0,0),(1,0)),
)
SPAWN=(2,7,8,10,11,14,18)
TYPE=(0,0,0,0,1,1,1,1,2,2,3,4,4,5,5,5,5,6,6)
ROTATE_CCW=(3,0,1,2,7,4,5,6,9,8,10,12,11,16,13,14,15,18,17)
ROTATE_CW=(1,2,3,0,5,6,7,4,9,8,10,12,11,14,15,16,13,18,17)
GRAVITY=(48,43,38,33,28,23,18,13,8,6,5,5,5,4,4,4,3,3,3,
         2,2,2,2,2,2,2,2,2,2,1)


class Buttons(IntFlag):
    NONE=0
    RIGHT=1
    LEFT=2
    DOWN=4
    B=64
    A=128


class Phase(IntEnum):
    ACTIVE=1
    LOCK=2
    SCAN=3
    CLEAR=4
    SCORE=5
    GOAL_CHECK=6
    RECEIVE=7
    SPAWN=8
    OVER=10


@dataclass(frozen=True)
class Rules:
    version: str='nes-ntsc-a-provisional-v1'
    start_level: int=0

    def __post_init__(self):
        if not 0<=self.start_level<=19:
            raise ValueError('Nintendo menu starting levels are 0 through 19')


@dataclass(frozen=True)
class Control:
    x: int=5
    y: int=0
    orientation: int=2
    fall: int=0
    das: int=0
    soft: int=0xA0
    soft_points: int=0
    previous: int=0


@dataclass(frozen=True)
class Observation:
    board: np.ndarray
    control: Control
    next_piece: int
    phase: Phase
    level: int
    lines: int
    score: int
    placements: int
    frames: int


def gravity_frames(level):
    if level<0:
        raise ValueError('Level must be nonnegative')
    return GRAVITY[min(level,29)]


def first_transition(start_level):
    return min(start_level*10+10,max(100,start_level*10-50))


def level_at_lines(start_level,lines):
    first=first_transition(start_level)
    return start_level if lines<first else start_level+1+(lines-first)//10


def lfsr_step(state):
    return (state>>1)|((((state>>1)^(state>>9))&1)<<15)


class Randomizer:
    def __init__(self,seed=0):
        # Seed is a reproducible number of warmup frames from the boot LFSR.
        self.state=0x8988
        for _ in range(seed%32767):
            self.advance()
        self.count=0
        self.last=0

    def advance(self):
        self.state=lfsr_step(self.state)

    def draw(self):
        self.count=(self.count+1)&255
        index=((self.state>>8)+self.count)&7
        if index==7 or SPAWN[index]==self.last:
            self.advance()
            index=(((self.state>>8)&7)+self.last)%7
        self.last=SPAWN[index]
        return self.last


def collides(board,x,y,orientation):
    for dx,dy in CELLS[orientation]:
        xx,yy=x+dx,y+dy
        if xx<0 or xx>=10 or yy< -2 or yy>=20 or (yy>=0 and board[yy,xx]):
            return True
    return False


def control_frame(board,state,buttons,level):
    """One active gameplay frame. Returns new control and a failed-down flag.

    Shift, rotation and gravity/soft-drop are ordered as in the Nintendo game.
    Buttons describe held electrical states, not one-shot macro commands.
    """
    buttons=int(buttons)
    if buttons & ~0xC7 or buttons<0:
        raise ValueError('Gameplay input supports only Left/Right/Down/A/B')
    pressed=buttons & ~state.previous
    x,y,orientation=state.x,state.y,state.orientation
    das,soft,points=state.das,state.soft,state.soft_points
    fall=(state.fall+1)&255
    if not buttons & Buttons.DOWN:
        shift=False
        if pressed & 3:
            das=0
            shift=True
        elif buttons & 3:
            das=(das+1)&255
            if ((das-16)&255)<128:
                das=10
                shift=True
        if shift:
            trial=x+(1 if buttons & Buttons.RIGHT else -1)
            if collides(board,trial,y,orientation):
                das=16
            else:
                x=trial
    target=(ROTATE_CW[orientation] if pressed & Buttons.A else
            ROTATE_CCW[orientation] if pressed & Buttons.B else orientation)
    if not collides(board,x,y,target):
        orientation=target
    forced=False
    if soft>=128:
        if pressed & Buttons.DOWN:
            soft=0
        else:
            soft=(soft+1)&255
            return Control(x,y,orientation,fall,das,soft,points,buttons),False
    if soft:
        if buttons & 15==Buttons.DOWN:
            soft+=1
            if soft>=3:
                soft=1
                points=(points+1)&255
                forced=True
        else:
            soft=points=0
    elif not buttons & 3 and pressed & 15==Buttons.DOWN:
        soft=1
    failed=False
    if forced or ((fall-gravity_frames(level))&255)<128:
        fall=0
        if collides(board,x,y+1,orientation):
            failed=True
        else:
            y+=1
    return Control(x,y,orientation,fall,das,soft,points,buttons),failed


class Game:
    def __init__(self,seed=0,rules=Rules()):
        self.seed,self.rules=seed,rules
        self._randomizer=Randomizer(seed)
        first=self._randomizer.draw()
        self._randomizer.advance()
        self.next_orientation=self._randomizer.draw()
        self.board=np.zeros((20,10),dtype=np.uint8)
        self.control=Control(orientation=first)
        self.phase=Phase.ACTIVE
        self.frames=self.lines=self.placements=0
        self.level=rules.start_level
        self.score_bytes=[0,0,0]
        self.vram_row=32
        self.scan_index=self.completed=self.clear_column=0
        self.line_clear_counts=[0]*5

    @property
    def over(self):
        return self.phase==Phase.OVER

    @property
    def score(self):
        return sum(((byte>>4)*10+(byte&15))*100**i for i,byte in enumerate(self.score_bytes))

    def observe(self):
        board=self.board.copy()
        board.flags.writeable=False
        return Observation(board,self.control,TYPE[self.next_orientation],self.phase,
                           self.level,self.lines,self.score,self.placements,self.frames)

    def _dirty(self):
        self.vram_row=min(self.vram_row,max(0,self.control.y-2))

    def _render(self):
        if self.phase==Phase.CLEAR:
            if self.frames%4==0:
                self.clear_column+=1
                if self.clear_column==5:
                    self.phase=Phase.SCORE
            self.vram_row=0
        elif self.vram_row<20:
            self.vram_row+=4
            if self.vram_row>=20:
                self.vram_row=32

    def _award_score(self):
        # Preserve the original soft-drop BCD quirk, including DEC before ADC.
        s=self.score_bytes
        if self.control.soft_points>=2:
            s[0]=((s[0]-1)&255)+self.control.soft_points
            s[0]&=255
            if s[0]&15>=10:
                s[0]=(s[0]+6)&255
            if s[0]&0xF0>=0xA0:
                s[0]=((s[0]&0xF0)+0x60)&255
                s[1]=(s[1]+1)&255
        base=(0,0x0040,0x0100,0x0300,0x1200)[self.completed]
        for _ in range(self.level+1):
            s[0]=(s[0]+(base&255))&255
            if s[0]>=0xA0:
                s[0]=(s[0]+0x60)&255
                s[1]=(s[1]+1)&255
            s[1]=(s[1]+(base>>8))&255
            if s[1]&15>=10:
                s[1]=(s[1]+6)&255
            if s[1]&0xF0>=0xA0:
                s[1]=(s[1]+0x60)&255
                s[2]=(s[2]+1)&255
            if s[2]&15>=10:
                s[2]=(s[2]+6)&255
            if s[2]&0xF0>=0xA0:
                s[:]=[0x99]*3
        self.control=replace(self.control,soft_points=0)

    def tick(self,buttons=Buttons.NONE):
        buttons=int(buttons)
        if buttons<0 or buttons & ~0xC7:
            raise ValueError('Unknown gameplay buttons')
        event=dict(locked=False,spawned=False,lines=0,game_over=self.over)
        if self.over:
            return event
        self._render()
        self.frames+=1
        self._randomizer.advance()
        if self.phase==Phase.ACTIVE:
            self.control,failed=control_frame(self.board,self.control,buttons,self.level)
            if failed:
                self._dirty()
                self.phase=Phase.LOCK
        else:
            self.control=replace(self.control,fall=(self.control.fall+1)&255,previous=buttons)
            if self.phase==Phase.LOCK:
                c=self.control
                if collides(self.board,c.x,c.y,c.orientation):
                    self.phase=Phase.OVER
                elif self.vram_row==32:
                    for dx,dy in CELLS[c.orientation]:
                        if c.y+dy>=0:
                            self.board[c.y+dy,c.x+dx]=TYPE[c.orientation]+1
                    self.placements+=1
                    self.scan_index=self.completed=0
                    self._dirty()
                    self.phase=Phase.SCAN
                    event['locked']=True
            elif self.phase==Phase.SCAN and self.vram_row==32:
                row=max(0,self.control.y-2)+self.scan_index
                if row<20 and self.board[row].all():
                    self.board[1:row+1]=self.board[:row].copy()
                    self.board[0]=0
                    self.completed+=1
                self.scan_index+=1
                if self.scan_index==4:
                    self.vram_row=self.clear_column=0
                    self.phase=Phase.CLEAR if self.completed else Phase.SCORE
            elif self.phase==Phase.SCORE:
                self.lines+=self.completed
                self.level=level_at_lines(self.rules.start_level,self.lines)
                self.line_clear_counts[self.completed]+=1
                self._award_score()
                event['lines']=self.completed
                self.completed=0
                self.phase=Phase.GOAL_CHECK
            elif self.phase in (Phase.GOAL_CHECK,Phase.RECEIVE):
                self.phase=Phase(int(self.phase)+1)
            elif self.phase==Phase.SPAWN and self.vram_row==32:
                self.control=replace(self.control,x=5,y=0,orientation=self.next_orientation,fall=0,soft=0)
                self.next_orientation=self._randomizer.draw()
                self.phase=Phase.ACTIVE
                event['spawned']=True
        event['game_over']=self.over
        return event

    def get_state(self):
        return dict(format_version=1,seed=self.seed,rules=asdict(self.rules),board=self.board.tolist(),
            control=asdict(self.control),phase=int(self.phase),frames=self.frames,lines=self.lines,
            placements=self.placements,level=self.level,score_bytes=self.score_bytes.copy(),
            next_orientation=self.next_orientation,vram_row=self.vram_row,scan_index=self.scan_index,
            completed=self.completed,clear_column=self.clear_column,line_clear_counts=self.line_clear_counts.copy(),
            randomizer=dict(state=self._randomizer.state,count=self._randomizer.count,last=self._randomizer.last))

    @classmethod
    def from_state(cls,state):
        if state['format_version']!=1:
            raise ValueError('Unsupported NES state version')
        game=cls(state['seed'],Rules(**state['rules']))
        game.board=np.asarray(state['board'],dtype=np.uint8)
        if game.board.shape!=(20,10):
            raise ValueError('Invalid NES board shape')
        game.control=Control(**state['control'])
        game.phase=Phase(state['phase'])
        for name in ('frames','lines','placements','level','next_orientation','vram_row','scan_index',
                     'completed','clear_column','score_bytes','line_clear_counts'):
            setattr(game,name,state[name].copy() if isinstance(state[name],list) else state[name])
        for name,value in state['randomizer'].items():
            setattr(game._randomizer,name,value)
        return game

    def signature(self):
        data=json.dumps(self.get_state(),sort_keys=True,separators=(',',':')).encode()
        return hashlib.sha256(data).hexdigest()
