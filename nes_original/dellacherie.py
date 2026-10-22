"""Experimental six-feature placement selector, evaluated in original NES code.

Feature family and coefficients (-1,+1,-1,-1,-4,-1) are described in
Thiery & Scherrer, Building Controllers for Tetris (2009), section2.1:
https://citeseerx.ist.psu.edu/document?doi=1d724d4f65598a33449b710f07393f5f10157173&repid=rep1&type=pdf
This implementation explicitly uses solid side walls/floor and empty space
above the board. Their reported scores are NOT results for our NES controller.
Only visible board and approximate candidate placements are scored; no RNG or
unseen pieces are read. All real transitions still run through Mesen.
"""
import numpy as np

from nes.engine import CELLS


def board_features(board):
    """Row transitions, column transitions, holes, cumulative wells."""
    board = np.asarray(board, dtype=bool)
    if board.shape != (20, 10):
        raise ValueError('Expected a20x10 board')
    rows = np.pad(board, ((0, 0), (1, 1)), constant_values=True)
    row_transitions = np.count_nonzero(rows[:, 1:] != rows[:, :-1])
    # The floor is filled; the unbounded space above the ceiling is empty.
    columns = np.vstack((np.zeros((1, 10), bool), board, np.ones((1, 10), bool)))
    column_transitions = np.count_nonzero(columns[1:] != columns[:-1])
    holes = np.count_nonzero(np.maximum.accumulate(board, axis=0) & ~board)
    wells = (~board) & rows[:, :-2] & rows[:, 2:]
    depths = np.zeros(10, dtype=np.int64)
    cumulative_wells = 0
    for row in wells:
        depths = (depths+1)*row
        cumulative_wells += int(depths.sum())
    return np.array((row_transitions, column_transitions, holes, cumulative_wells), dtype=float)


def placement_features(before, state, after, cleared):
    """Landing-center height, eroded cells, then four after-board features.

    Eroded cells is cleared rows times cells of THIS piece in those rows;
    earlier occupied cells do not contribute to that second factor.
    """
    before = np.asarray(before, dtype=bool)
    occupied = before.copy()
    ys = []
    for dx, dy in CELLS[state.orientation]:
        x, y = state.x+dx, state.y+dy
        if 0 <= y < 20 and 0 <= x < 10:
            occupied[y, x] = True
        ys.append(y)
    full_rows = occupied.all(axis=1)
    eroded = int(cleared)*sum(0 <= y < 20 and full_rows[y] for y in ys)
    landing_height = 20-(min(ys)+max(ys))/2
    return np.concatenate(([landing_height, eroded], board_features(after)))


class Dellacherie:
    """One-piece experimental prior on the existing reachable target family."""
    weights = np.array((-1., 1., -1., -1., -4., -1.))

    def __init__(self):
        self.calls = 0

    def __call__(self, observation, plan):
        self.calls += 1
        actions = plan.actions
        scores = np.array([placement_features(observation.board, state, after, cleared) @ self.weights
            - 1000000*bool(dead) for state, after, cleared, dead in
            zip(actions.controls, actions.boards, actions.lines, actions.terminal)])
        return int(scores.argmax())
