"""Visible-state candidate actions executed through the real NES frame engine."""
from dataclasses import dataclass

import numpy as np

from nes.engine import (Buttons,CELLS,Control,Observation,Phase,ROTATE_CCW,ROTATE_CW,
                        TYPE,collides,control_frame,gravity_frames,level_at_lines)


ENCODING_VERSION=1


@dataclass(frozen=True)
class ActionRules:
    reaction_frames: int=3
    navigation_frames: int=48
    rotation_period: int=4
    horizontal_period: int=0  # 0=held DAS; >=2=press/release pulse period.

    def __post_init__(self):
        if self.reaction_frames<0 or self.navigation_frames<1 or self.rotation_period<2:
            raise ValueError('Invalid timed action profile')
        if self.horizontal_period and self.horizontal_period<2:
            raise ValueError('Horizontal pulse period must be zero or at least two')


@dataclass
class Actions:
    features: np.ndarray
    boards: np.ndarray
    lines: np.ndarray
    terminal: np.ndarray
    paths: tuple
    controls: tuple
    heuristic: np.ndarray


def structural(boards):
    occupied=np.asarray(boards)!=0
    heights=np.where(occupied.any(axis=1),20-occupied.argmax(axis=1),0)
    holes=(np.maximum.accumulate(occupied,axis=1)&~occupied).sum(axis=1)
    summary=np.column_stack((heights.sum(axis=1)/200,holes.sum(axis=1)/200,
        np.abs(np.diff(heights,axis=1)).sum(axis=1)/200,heights.max(axis=1)/20))
    return np.concatenate((summary,heights/20,holes/20),axis=1).astype(np.float32)


def rotate_button(current,target):
    if not 0<=current<19 or not 0<=target<19 or TYPE[current]!=TYPE[target]:
        raise ValueError('Rotation target must belong to the current piece')
    clockwise=counterclockwise=0
    pose=current
    while pose!=target:
        pose=ROTATE_CW[pose]
        clockwise+=1
    pose=current
    while pose!=target:
        pose=ROTATE_CCW[pose]
        counterclockwise+=1
    return int(Buttons.A if clockwise<=counterclockwise else Buttons.B)


def preview_lock(board,state):
    result=board.copy()
    terminal=collides(board,state.x,state.y,state.orientation)
    if terminal:
        return result,0,True
    for dx,dy in CELLS[state.orientation]:
        if state.y+dy>=0:
            result[state.y+dy,state.x+dx]=TYPE[state.orientation]+1
    cleared=0
    for row in range(max(0,state.y-2),max(0,state.y-2)+4):
        if row<20 and result[row].all():
            result[1:row+1]=result[:row].copy()
            result[0]=0
            cleared+=1
    return result,cleared,False


def enumerate_actions(observation,rules=ActionRules(),*,board_input=False):
    if observation.phase!=Phase.ACTIVE:
        raise ValueError('Plan only for an active piece')
    board=observation.board
    piece=TYPE[observation.control.orientation]
    options=[]
    seen={}
    transitions={}
    for target in range(19):
        if TYPE[target]!=piece: continue
        min_x=-min(dx for dx,dy in CELLS[target])
        max_x=9-max(dx for dx,dy in CELLS[target])
        for target_x in range(min_x,max_x+1):
            state=observation.control
            path=[]
            for frame in range(rules.reaction_frames+rules.navigation_frames+160):
                elapsed=frame-rules.reaction_frames
                if elapsed<0:
                    buttons=0
                elif elapsed>=rules.navigation_frames:
                    buttons=int(Buttons.DOWN)
                else:
                    horizontal=int(Buttons.LEFT if state.x>target_x else Buttons.RIGHT if state.x<target_x else 0)
                    moving = bool(horizontal)
                    if rules.horizontal_period and elapsed%rules.horizontal_period:
                        horizontal=0
                    if state.orientation!=target:
                        rotation=rotate_button(state.orientation,target)
                        pulse=rotation if elapsed%rules.rotation_period==0 and not state.previous & rotation else 0
                        buttons=horizontal|pulse
                    else:
                        buttons=horizontal if moving else int(Buttons.DOWN)
                key=state,buttons
                if key not in transitions:
                    transitions[key]=control_frame(board,state,buttons,observation.level)
                state,failed=transitions[key]
                path.append(buttons)
                if failed: break
            else:
                raise RuntimeError('Timed candidate did not lock within its action bound')
            result,cleared,terminal=preview_lock(board,state)
            key=(result.astype(bool).tobytes(),state.das,terminal)
            candidate=(result,cleared,terminal,tuple(path),state)
            if key not in seen:
                seen[key]=len(options)
                options.append(candidate)
            elif len(path)<len(options[seen[key]][3]):
                options[seen[key]]=candidate
    boards=np.asarray([row[0] for row in options])
    lines=np.asarray([row[1] for row in options],dtype=np.int64)
    terminal=np.asarray([row[2] for row in options],dtype=bool)
    paths=tuple(row[3] for row in options)
    controls=tuple(row[4] for row in options)
    after=structural(boards)
    before=structural(board[None])[0]
    preview=np.zeros(7,dtype=np.float32)
    preview[observation.next_piece]=1
    # Context is exclusively visible. The newly drawn preview after this action
    # is unknown and never simulated or read from a randomizer.
    extras=np.asarray([[observation.level/29,gravity_frames(observation.level)/48,
        observation.lines%10/10,line/4,float(dead),len(path)/180,control.das/16,
        control.soft_points/20,float(control.y<2),
        level_at_lines(observation.start_level,observation.lines+int(line))/29,
        observation.start_level/19,min(observation.lines,500)/500]
        for line,dead,path,control in zip(lines,terminal,paths,controls)],dtype=np.float32)
    parts=[after,np.broadcast_to(before,after.shape),np.broadcast_to(preview,(len(boards),7)),extras]
    if board_input: parts.insert(0,(boards!=0).reshape(len(boards),-1))
    features=np.concatenate(parts,axis=1).astype(np.float32)
    heuristic=(0.76*lines-102*after[:,0]-900*after[:,1]-36*after[:,2]-1000*terminal).astype(np.float32)
    return Actions(features,boards,lines,terminal,paths,controls,heuristic)


def execute(game,actions,index,*,callback=None):
    """Return actual cleared lines/done after a legal path and entry processing."""
    before_lines,before_placements=game.lines,game.placements
    path=actions.paths[index]
    def tick(buttons):
        event=game.tick(buttons)
        if callback: callback(game,int(buttons),event)
        return event
    for buttons in path:
        if game.phase!=Phase.ACTIVE:
            raise AssertionError('Candidate path continued after active control ended')
        tick(buttons)
    if game.phase!=Phase.LOCK or game.control!=actions.controls[index]:
        raise AssertionError('Preview control does not match actual legal inputs')
    for _ in range(120):
        if game.over or game.phase==Phase.ACTIVE: break
        tick(0)
    else:
        raise AssertionError('NES entry processing exceeded its bound')
    actual=game.lines-before_lines
    assert actual==int(actions.lines[index]),'Preview/engine cleared lines disagree'
    assert game.over==bool(actions.terminal[index]),'Preview/engine terminal state disagrees'
    assert np.array_equal(game.board!=0,actions.boards[index]!=0),'Preview/engine locked boards disagree'
    assert game.placements-before_placements==int(not game.over)
    return actual,game.over
