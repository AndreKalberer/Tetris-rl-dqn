"""Pure, observable one-piece previews for a candidate-board Q network.

No environment steps, state clones, randomizer reads, or future-piece draws occur
here. Losing placements remain candidates. Equivalent rotations are deduplicated.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from tetris_macro_env import TetrisPlacementEnv, _locked_playfield_01, compute_features

ENCODING_VERSION = 1


@dataclass(frozen=True)
class VisibleState:
    board: np.ndarray
    current: int
    queue: tuple[int, ...]
    held: int
    can_hold: bool = True
    padding: int = 4
    include_hold: bool = True


@dataclass
class Candidates:
    actions: np.ndarray
    features: np.ndarray
    boards: np.ndarray
    lines: np.ndarray
    terminal: np.ndarray
    states: tuple[VisibleState, ...] = ()


@lru_cache(maxsize=7)
def rotations(piece: int):
    from tetris_gymnasium.envs.tetris import Tetris
    matrix = Tetris.TETROMINOES[piece].matrix
    seen = set()
    out = []
    for rotation in range(4):
        rotated = np.rot90(matrix, rotation)
        ys, xs = np.nonzero(rotated)
        # Translation-equivalent rotations have the same possible final boards.
        key = tuple(zip((ys - ys.min()).tolist(), (xs - xs.min()).tolist()))
        if key not in seen:
            out.append((rotation, ys, xs))
            seen.add(key)
    return out


def one_hot(piece):
    out = np.zeros(7, dtype=np.float32)
    if piece >= 0:
        out[piece] = 1
    return out


def board_features(board):
    return batch_board_features(board[None])[0]


def batch_board_features(boards):
    """Same encoding as board_features, with NumPy work shared across candidates."""
    _, h, w = boards.shape
    occupied = boards != 0
    heights = np.where(occupied.any(axis=1), h - occupied.argmax(axis=1), 0)
    covered = np.maximum.accumulate(occupied, axis=1) & ~occupied
    hole_columns = covered.sum(axis=1)
    summary = np.column_stack((heights.sum(axis=1)/(h*w),
        hole_columns.sum(axis=1)/(h*w), np.abs(np.diff(heights, axis=1)).sum(axis=1)/(h*w),
        heights.max(axis=1)/h))
    return np.concatenate((summary, heights/h, hole_columns/h), axis=1).astype(np.float32)


def preview_rotation(board, ys, xs, positions):
    """Preview every horizontal position of one rotation in one array operation."""
    h, w = board.shape
    columns = positions[:, None] + xs[None, :]
    padded = np.pad(board, ((0, int(ys.max())+1), (0, 0)), constant_values=1)
    y_grid = np.arange(h+1)[:, None, None] + ys[None, None, :]
    collisions = padded[y_grid, columns[None, :, :]].any(axis=2)
    terminal = collisions[0].copy()
    landing = np.maximum(0, collisions.argmax(axis=0)-1)
    results = np.broadcast_to(board, (len(positions), h, w)).copy()
    valid = np.flatnonzero(~terminal)
    results[valid[:, None], landing[valid, None]+ys[None, :], columns[valid]] = 1
    full = results.all(axis=2)
    full[terminal] = False
    lines = full.sum(axis=1)
    for i in np.flatnonzero(lines):
        results[i] = np.concatenate((np.zeros((int(lines[i]), w), dtype=np.uint8), results[i][~full[i]]))
    return results, lines, terminal


def preview(board, ys, xs, x):
    """Return locked board, lines and spawn collision; drop vertically from y=0."""
    h, _ = board.shape
    cols = xs + x
    if np.any(board[ys, cols]):
        return board.copy(), 0, True
    # For each occupied cell, find its first obstacle below its initial position.
    # Including every cell handles cavities and empty rows in rotation matrices.
    landing = h
    for dy, col in zip(ys, cols):
        obstacles = np.flatnonzero(board[int(dy)+1:, col])
        distance = int(obstacles[0]) if obstacles.size else h - int(dy) - 1
        landing = min(landing, distance)
    result = board.copy()
    result[ys + landing, cols] = 1
    full = result.all(axis=1)
    lines = int(full.sum())
    if lines:
        result = np.concatenate((np.zeros((lines, result.shape[1]), dtype=np.uint8), result[~full]))
    return result, lines, False


def enumerate_candidates(env: TetrisPlacementEnv, *, board_input=True) -> Candidates:
    u = env.unwrapped
    if u.game_over:
        raise RuntimeError("Cannot preview a terminal state")
    if env.next_n < (2 if env.include_hold else 1):
        raise ValueError("Candidate agent needs next_n >= 2 with hold, >= 1 without")
    state = VisibleState(_locked_playfield_01(env, {}), env._current_piece_index(),
        tuple(env._next_piece_indices()), env._held_piece_idx, not u.has_swapped,
        u.padding, env.include_hold)
    return enumerate_visible(state, board_input=board_input)


def enumerate_visible(state: VisibleState, *, board_input=True) -> Candidates:
    """Preview a visible state, with -1 for queue pieces not yet observed."""
    from tetris_gymnasium.envs.tetris import Tetris
    board = state.board
    width = board.shape[1]
    x_count = width + state.padding + 1
    previous_features = board_features(board)
    queue = list(state.queue)
    current = state.current
    held = state.held
    actions, boards, lines_list, terminal_list, contexts, child_contexts = [], [], [], [], [], []
    for hold in range(2 if state.include_hold and state.can_hold else 1):
        piece = current if not hold else (held if held >= 0 else queue[0])
        consume = 2 if hold and held < 0 else 1
        if len(queue) < consume or piece < 0 or queue[consume-1] < 0:
            raise ValueError('Lookahead exhausted the visible piece queue')
        next_piece = queue[consume-1]
        remaining_queue = queue[consume:] + [-1]*consume
        held_after = current if hold else held
        context = np.concatenate([one_hot(next_piece), *[one_hot(p) for p in remaining_queue], one_hot(held_after)])
        for rotation, ys, xs in rotations(piece):
            positions = np.arange(-int(xs.min()), width-int(xs.max()))
            result, lines, terminal = preview_rotation(board, ys, xs, positions)
            next_matrix = Tetris.TETROMINOES[next_piece].matrix
            ny, nx = np.nonzero(next_matrix)
            spawn_x = width // 2 - next_matrix.shape[0] // 2
            terminal |= result[:, ny, nx+spawn_x].any(axis=1)
            actions.append(hold*4*x_count + rotation*x_count + positions + state.padding)
            boards.append(result)
            lines_list.append(lines)
            terminal_list.append(terminal)
            contexts.extend([context]*len(positions))
            child_contexts.extend([(next_piece, tuple(remaining_queue), held_after)]*len(positions))
    boards = np.concatenate(boards)
    lines = np.concatenate(lines_list)
    terminal = np.concatenate(terminal_list)
    parts = [batch_board_features(boards), np.broadcast_to(previous_features, (len(boards),len(previous_features))),
             np.asarray(contexts), np.column_stack((lines/4, terminal))]
    if board_input:
        parts.insert(0, boards.reshape(len(boards), -1))
    features = np.concatenate(parts, axis=1).astype(np.float32)
    states = tuple(VisibleState(b, cur, queue, held, True, state.padding, state.include_hold)
                   for b, (cur, queue, held) in zip(boards, child_contexts))
    return Candidates(np.concatenate(actions).astype(np.int64), features, boards, lines, terminal, states)


def heuristic_scores(candidates):
    """Transparent hand-written baseline, not a trained RL policy."""
    stats = np.asarray([compute_features(b) for b in candidates.boards])
    return (2.0*candidates.lines - 0.5*stats[:, 0] - 0.8*stats[:, 1]
            - 0.2*stats[:, 2] - 1000*candidates.terminal)
