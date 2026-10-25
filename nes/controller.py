"""Untrained stacking probe. Reads visible observations and emits held buttons."""
import time

import numpy as np

from nes.engine import Buttons,CELLS,Phase,ROTATE_CCW,ROTATE_CW,TYPE,collides


def targets(observation):
    """Geometric proposals only; the live game decides whether each is reachable."""
    board=observation.board
    piece=TYPE[observation.control.orientation]
    proposals=[]
    for orientation in range(19):
        if TYPE[orientation]!=piece:
            continue
        for x in range(10):
            if collides(board,x,0,orientation):
                continue
            y=0
            while not collides(board,x,y+1,orientation):
                y+=1
            result=board.copy()
            for dx,dy in CELLS[orientation]:
                if y+dy>=0:
                    result[y+dy,x+dx]=piece+1
            full=result.all(axis=1)
            lines=int(full.sum())
            if lines:
                result=np.concatenate((np.zeros((lines,10),dtype=np.uint8),result[~full]))
            occupied=result!=0
            heights=np.where(occupied.any(axis=0),20-occupied.argmax(axis=0),0)
            holes=int((np.maximum.accumulate(occupied,axis=0)&~occupied).sum())
            quality=0.76*lines-0.51*heights.sum()-4.5*holes-0.18*np.abs(np.diff(heights)).sum()
            proposals.append((float(quality),orientation,x))
    return proposals


class StackingProbe:
    """Baseline for exercising mechanics, not a trained NES agent.

    Horizontal motion uses held DAS, never alternating-frame tapping. Rotations
    use ordinary A/B press/release edges. Planning latency is recorded but not
    yet charged to the game clock, so results are development probes only.
    """
    name='untrained-stacking-probe'

    def __init__(self):
        self.last_placement=-1
        self.target=None
        self.horizontal=0
        self.planning_seconds=0.0
        self.plans=0

    def command(self,observation):
        if observation.phase!=Phase.ACTIVE:
            return self.horizontal
        if observation.placements!=self.last_placement:
            started=time.perf_counter()
            proposals=targets(observation)
            self.target=max(proposals) if proposals else None
            self.last_placement=observation.placements
            self.planning_seconds+=time.perf_counter()-started
            self.plans+=1
        if self.target is None:
            return int(Buttons.DOWN)
        _,orientation,x=self.target
        c=observation.control
        self.horizontal=int(Buttons.LEFT if c.x>x else Buttons.RIGHT if c.x<x else Buttons.NONE)
        if c.orientation!=orientation:
            cw=ccw=0
            current=c.orientation
            while current!=orientation:
                current=ROTATE_CW[current]
                cw+=1
            current=c.orientation
            while current!=orientation:
                current=ROTATE_CCW[current]
                ccw+=1
            rotation=int(Buttons.A if cw<=ccw else Buttons.B)
            return self.horizontal | (rotation if not c.previous & rotation else 0)
        return self.horizontal or int(Buttons.DOWN)
