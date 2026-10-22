"""Optional CPU-compiled provisional control/geometry for lookahead only.

This never advances the original emulator. Its frame step is differentially
tested against nes.engine.control_frame. No fast-math or hidden game state.
"""
import numpy as np
from numba import njit

from nes.engine import CELLS,GRAVITY,ROTATE_CW,ROTATE_CCW,TYPE

SHAPES = np.asarray(CELLS,dtype=np.int64)
CW = np.asarray(ROTATE_CW,dtype=np.int64)
CCW = np.asarray(ROTATE_CCW,dtype=np.int64)
KINDS = np.asarray(TYPE,dtype=np.int64)
SPEED = np.asarray(GRAVITY,dtype=np.int64)
ROTATE = np.zeros((19,19),dtype=np.int64)
for a in range(19):
    for b in range(19):
        if TYPE[a] != TYPE[b]:
            continue
        cw=ccw=0
        p=a
        while p!=b:
            p=ROTATE_CW[p];cw+=1
        p=a
        while p!=b:
            p=ROTATE_CCW[p];ccw+=1
        ROTATE[a,b]=128 if cw<=ccw else 64


@njit(cache=True)
def collision(board,x,y,orientation):
    for i in range(4):
        xx=x+SHAPES[orientation,i,0]
        yy=y+SHAPES[orientation,i,1]
        if xx<0 or xx>=10 or yy< -2 or yy>=20:
            return True
        if yy>=0 and board[yy,xx]:
            return True
    return False


@njit(cache=True)
def frame(board,state,buttons,level):
    x,y,orientation,fall,das,soft,points,previous=state
    pressed=buttons & ~previous
    fall=(fall+1)&255
    if not buttons&4:
        shift=False
        if pressed&3:
            das=0
            shift=True
        elif buttons&3:
            das=(das+1)&255
            if ((das-16)&255)<128:
                das=10
                shift=True
        if shift:
            trial=x+(1 if buttons&1 else -1)
            if collision(board,trial,y,orientation):
                das=16
            else:
                x=trial
    target=CW[orientation] if pressed&128 else CCW[orientation] if pressed&64 else orientation
    if not collision(board,x,y,target):
        orientation=target
    forced=False
    if soft>=128:
        if pressed&4:
            soft=0
        else:
            soft=(soft+1)&255
            return (x,y,orientation,fall,das,soft,points,buttons),False
    if soft:
        if buttons&15==4:
            soft+=1
            if soft>=3:
                soft=1
                points=(points+1)&255
                forced=True
        else:
            soft=points=0
    elif not buttons&3 and pressed&15==4:
        soft=1
    failed=False
    if forced or ((fall-SPEED[min(level,29)])&255)<128:
        fall=0
        if collision(board,x,y+1,orientation):
            failed=True
        else:
            y+=1
    return (x,y,orientation,fall,das,soft,points,buttons),failed


@njit(cache=True)
def lock_score(board,state,line_power=1,well_column=-1):
    x,y,orientation=state[:3]
    if collision(board,x,y,orientation):
        result=board
        cleared=0
        dead=True
    else:
        result=board.copy()
        for i in range(4):
            xx=x+SHAPES[orientation,i,0]
            yy=y+SHAPES[orientation,i,1]
            if yy>=0:
                result[yy,xx]=True
        cleared=0
        for row in range(max(0,y-2),max(0,y-2)+4):
            if row>=20:
                continue
            full=True
            for col in range(10):
                if not result[row,col]:
                    full=False
                    break
            if full:
                for r in range(row,0,-1):
                    for col in range(10):
                        result[r,col]=result[r-1,col]
                for col in range(10):
                    result[0,col]=False
                cleared+=1
        dead=False
    height_sum=holes=bump=last_height=max_height=well_height=0
    for col in range(10):
        seen=False
        height=0
        for row in range(20):
            if result[row,col] and not seen:
                height=20-row
                seen=True
            elif not result[row,col] and seen:
                holes+=1
        height_sum+=height
        max_height=max(max_height,height)
        if col==well_column:
            well_height=height
        if col:
            bump+=abs(height-last_height)
        last_height=height
    # Match the existing float32 geometric heuristic's arithmetic stages.
    if well_column<0:
        score=(np.float32(.76)*(cleared**line_power) - np.float32(102)*np.float32(height_sum/200.)
            - np.float32(900)*np.float32(holes/200.) - np.float32(36)*np.float32(bump/200.)
            - np.float32(1000)*dead)
    else:
        score=(.76*cleared**line_power-.25*height_sum-4.5*holes-.25*bump
            -1.5*well_height-.15*max_height**2-1000*dead)
    return float(score)


@njit(cache=True)
def best_future(board,initial,level,horizontal_period=0,reaction=3,navigation=48,rotation_period=4,line_power=1,well_column=-1):
    best=-1.e20
    kind=KINDS[initial[2]]
    for target in range(19):
        if KINDS[target]!=kind:
            continue
        left=10
        right=-10
        for i in range(4):
            left=min(left,SHAPES[target,i,0])
            right=max(right,SHAPES[target,i,0])
        for target_x in range(-left,10-right):
            state=initial
            failed=False
            for f in range(reaction+navigation+160):
                elapsed=f-reaction
                if elapsed<0:
                    buttons=0
                elif elapsed>=navigation:
                    buttons=4
                else:
                    horizontal=2 if state[0]>target_x else 1 if state[0]<target_x else 0
                    moving=horizontal!=0
                    if horizontal_period and elapsed%horizontal_period:
                        horizontal=0
                    if state[2]!=target:
                        rotation=ROTATE[state[2],target]
                        pulse=rotation if elapsed%rotation_period==0 and not state[7]&rotation else 0
                        buttons=horizontal|pulse
                    else:
                        buttons=horizontal if moving else 4
                state,failed=frame(board,state,buttons,level)
                if failed:
                    break
            if not failed:
                raise RuntimeError('Compiled provisional action exceeded its bound')
            best=max(best,lock_score(board,state,line_power,well_column))
    return best


def warmup(line_power=1,well_column=-1):
    board=np.zeros((20,10),dtype=np.bool_)
    state=(5,0,2,0,0,0,0,0)
    best_future(board,state,18,0,3,48,4,line_power,well_column)
    best_future(board,state,18,2,3,48,4,line_power,well_column)
