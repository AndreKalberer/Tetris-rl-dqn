"""Experimental geometric scores. No game transitions or hidden information."""
import numpy as np


def action_scores(actions,line_power=1,well_column=-1):
    if well_column==-1:
        return actions.heuristic + .76*(actions.lines**line_power-actions.lines)
    if not 0<=well_column<10:
        raise ValueError('Well column must be -1 or0..9')
    occupied=actions.boards!=0
    heights=np.where(occupied.any(axis=1),20-occupied.argmax(axis=1),0)
    holes=(np.maximum.accumulate(occupied,axis=1)&~occupied).sum(axis=(1,2))
    bump=np.abs(np.diff(heights,axis=1)).sum(axis=1)
    # Hypothesis: less aggregate-height pressure permits a four-row stack;
    # an explicit empty-well preference makes the long I useful for Tetrises.
    # Quadratic peak pressure still penalizes dangerous stacks. Not validated.
    return (.76*actions.lines**line_power - .25*heights.sum(axis=1)
        - 4.5*holes - .25*bump - 1.5*heights[:,well_column]
        - .15*heights.max(axis=1)**2 - 1000*actions.terminal).astype(np.float32)
