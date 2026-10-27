"""The NES bot's hand-weighted evaluator and beam search, ported to the offline game.

Board scoring reuses nes_original.sim.board_features (the NES bot's numba feature
code) on the 20 visible rows, plus landing height, eroded cells and the clear type.
The default weights are the NES bot's Dellacherie survive weights.

Search: the first placement uses the same timed, legal reachable paths as the
learned bot (offline.bot), including the hold option. Deeper plies use the known
preview pieces and hold, with a cheaper placement family (rotate and shift at
the top, then drop), and keep the best `beam` boards per ply. Only visible
pieces are read; nothing beyond the next five is assumed.

The controller timing is identical to offline.bot.Bot: a 30-frame reaction wait,
one input every 3 frames, at most 12 commands, then hard drop.
"""
from collections import deque
import time

import numpy as np
from numba import njit

from nes_original.sim import (BOARD_FEATURES, DELLACHERIE, FEATURES, N_BOARD, N_FEATURES,
                              board_features, pad_weights)
from offline.engine import advance, cells, collides, locked_board, spawn_pose

HIDDEN = 4
ROWS = 24
DEAD = -1.e12
# CELLS[piece, rotation, i] = (dx, dy) in the offline engine's convention.
CELLS = np.array([[cells(p, r) for r in range(4)] for p in range(7)], dtype=np.int64)
SPAWN_X = np.array([spawn_pose(p, type('R', (), dict(width=10, hidden_rows=HIDDEN))).x for p in range(7)],
                   dtype=np.int64)
SPAWN_Y = HIDDEN-2
# Distinct rotations per piece (O has one; I/S/Z still need all four poses for kicks,
# but their final shapes repeat, so 2 suffice for the drop family).
ROTATIONS = np.array((2, 1, 4, 2, 2, 4, 4), dtype=np.int64)


@njit(cache=True)
def _collide(board, piece, rot, x, y):
    for i in range(4):
        xx = x+CELLS[piece, rot, i, 0]
        yy = y+CELLS[piece, rot, i, 1]
        if xx < 0 or xx >= 10 or yy < 0 or yy >= ROWS or board[yy, xx]:
            return True
    return False


@njit(cache=True)
def _lock(board, piece, rot, x, y):
    """Return (board, cleared, dead) using the offline top-out rule."""
    result = board.copy()
    for i in range(4):
        result[y+CELLS[piece, rot, i, 1], x+CELLS[piece, rot, i, 0]] = True
    out = np.zeros_like(result)
    write = ROWS-1
    cleared = 0
    for row in range(ROWS-1, -1, -1):
        full = True
        for col in range(10):
            if not result[row, col]:
                full = False
                break
        if full:
            cleared += 1
        else:
            out[write] = result[row]
            write -= 1
    dead = False
    for row in range(HIDDEN):
        for col in range(10):
            if out[row, col]:
                dead = True
    return out, cleared, dead


@njit(cache=True)
def _score(before, after, piece, rot, x, y, cleared, w, well, scratch):
    """Return (total value, placement-only part) for one lock, NES feature order."""
    board_features(after[HIDDEN:], well, scratch)
    low = 99
    high = -99
    in_full = 0
    for i in range(4):
        yy = y+CELLS[piece, rot, i, 1]
        low = min(low, yy)
        high = max(high, yy)
        if cleared:
            xx = x+CELLS[piece, rot, i, 0]
            full = True
            for col in range(10):
                if col == xx or before[yy, col]:
                    continue
                other = False
                for j in range(4):
                    if y+CELLS[piece, rot, j, 1] == yy and x+CELLS[piece, rot, j, 0] == col:
                        other = True
                if not other:
                    full = False
                    break
            if full:
                in_full += 1
    scratch[N_BOARD] = 20-((low-HIDDEN)+(high-HIDDEN))/2
    scratch[N_BOARD+1] = cleared*in_full
    for k in range(4):
        scratch[N_BOARD+2+k] = 1. if cleared == k+1 else 0.
    total = 0.
    part = 0.
    for k in range(N_FEATURES):
        total += w[k]*scratch[k]
        if k >= N_BOARD:
            part += w[k]*scratch[k]
    return total, part


@njit(cache=True)
def _drops(board, piece, out):
    """Cheap placement family: rotate at spawn, shift along the spawn row, drop.

    Fills out[n] = (rot, x, y) and returns n; empty if the piece cannot spawn.
    """
    n = 0
    sx = SPAWN_X[piece]
    if _collide(board, piece, 0, sx, SPAWN_Y):
        return 0
    for rot in range(ROTATIONS[piece]):
        if _collide(board, piece, rot, sx, SPAWN_Y):
            continue
        for direction in (-1, 1):
            x = sx if direction == -1 else sx+1
            while not _collide(board, piece, rot, x, SPAWN_Y):
                y = SPAWN_Y
                while not _collide(board, piece, rot, x, y+1):
                    y += 1
                out[n, 0] = rot
                out[n, 1] = x
                out[n, 2] = y
                n += 1
                x += direction
    return n


def _children(board, piece_seq, qpos, held, w, well, scratch, drops):
    """Yield (value, part, board, qpos, held) for each next placement, hold included."""
    if qpos >= len(piece_seq):
        return
    options = [(piece_seq[qpos], qpos+1, held)]
    if held < 0:
        if qpos+1 < len(piece_seq):
            options.append((piece_seq[qpos+1], qpos+2, piece_seq[qpos]))
    elif held != piece_seq[qpos]:
        options.append((held, qpos+1, piece_seq[qpos]))
    for piece, next_q, next_held in options:
        n = _drops(board, piece, drops)
        for i in range(n):
            rot, x, y = drops[i]
            after, cleared, dead = _lock(board, piece, rot, x, y)
            if dead:
                continue
            total, part = _score(board, after, piece, rot, x, y, cleared, w, well, scratch)
            yield total, part, after, next_q, next_held


class Planner:
    """Beam search over the visible pieces with the NES evaluator."""

    def __init__(self, weights=DELLACHERIE, well=-1, beam=8, depth=3):
        self.w = pad_weights(weights)
        self.well, self.beam, self.depth = int(well), int(beam), int(depth)
        self.scratch = np.zeros(N_FEATURES)
        self.drops = np.zeros((96, 3), np.int64)

    def first_scores(self, before, locks):
        """Score the exact first locks. locks: list of (piece, pose, board, cleared, dead)."""
        rows = []
        for piece, pose, after, cleared, dead in locks:
            if dead:
                rows.append((DEAD, DEAD))
                continue
            rows.append(_score(before, after, piece, pose.rotation, pose.x, pose.y, cleared, self.w, self.well,
                               self.scratch))
        return rows

    def choose(self, before, locks, seq_after):
        """Pick a first lock index. seq_after[i] = (piece sequence, qpos, held) after lock i."""
        scored = self.first_scores(before, locks)
        values = np.array([t for t, _ in scored])
        if self.depth <= 1 or len(locks) == 1:
            return int(values.argmax())
        # Each beam entry: (accumulated placement value, last board value, board, qpos, held, root).
        order = np.argsort(-values, kind='stable')
        frontier, seen = [], set()
        for i in order:
            if values[i] <= DEAD or len(frontier) >= self.beam:
                break
            board = locks[i][2]
            seq, qpos, held = seq_after[i]
            key = (board.tobytes(), qpos, held)
            if key in seen:
                continue
            seen.add(key)
            frontier.append((scored[i][1], values[i]-scored[i][1], board, qpos, held, int(i)))
        best_root = {}
        for _ in range(self.depth-1):
            candidates = []
            for acc, board_value, board, qpos, held, root in frontier:
                seq = seq_after[root][0]
                any_child = False
                for total, part, after, next_q, next_held in _children(board, seq, qpos, held, self.w, self.well,
                                                                      self.scratch, self.drops):
                    any_child = True
                    candidates.append((acc+part, total-part, after, next_q, next_held, root))
                if not any_child:
                    # No visible piece left or every child tops out: keep the board as it is.
                    candidates.append((acc, board_value if qpos >= len(seq) else DEAD, board, qpos, held, root))
            if not candidates:
                break
            candidates.sort(key=lambda c: -(c[0]+c[1]))
            frontier, seen = [], set()
            for c in candidates:
                key = (c[2].tobytes(), c[3], c[4])
                if key in seen:
                    continue
                seen.add(key)
                frontier.append(c)
                if len(frontier) >= self.beam:
                    break
        for acc, board_value, _, _, _, root in frontier:
            value = acc+board_value
            if value > best_root.get(root, DEAD*10):
                best_root[root] = value
        if not best_root:
            return int(values.argmax())
        return max(best_root, key=lambda r: (best_root[r], -r))


def control_edge(board, piece, pose, action, rules):
    accepted = True
    for frame in range(rules.control_frames):
        pose, locked, ok = advance(board, piece, pose, action if frame == 0 else 'none', rules)
        accepted = accepted and ok
        if locked:
            return pose, True, accepted
    return pose, False, accepted


def enumerate_locks(game, start_pose, max_commands=12):
    """Same timed reachable-path family as offline.bot.enumerate_paths, without network features.

    Returns paths, locks (piece, landing pose, board, cleared, dead), the visible
    piece sequence after each lock as (sequence, next index, held), and the full boards.
    """
    rules = game.rules
    queue = list(game.queue)[:5]
    paths, locks, after, full = [], [], [], []
    seen_results = set()
    for hold in range(1 if game.hold_used else 2):
        piece = game.piece if not hold else (game.held if game.held >= 0 else queue[0])
        held = game.piece if hold else game.held
        consume = 2 if hold and game.held < 0 else 1
        # Pieces still to come, in order, after this lock.
        seq = tuple(queue[consume-1:])
        future = seq[0]
        root = start_pose
        prefix = ()
        if hold:
            root = spawn_pose(piece, rules)
            if collides(game.board, piece, root):
                continue
            locked = False
            for _ in range(rules.control_frames):
                root, locked, _ = advance(game.board, piece, root, 'none', rules)
                if locked:
                    break
            if locked:
                continue
            prefix = ('hold',)
        frontier = deque([(root, prefix)])
        seen_poses = {(root.x, root.y, root.rotation)}
        while frontier:
            pose, path = frontier.popleft()
            landing, _, _ = advance(game.board, piece, pose, 'hard', rules)
            result, cleared, terminal = locked_board(game.board, piece, landing, rules)
            terminal = terminal or collides(result, future, spawn_pose(future, rules))
            key = (result.tobytes(), held)
            if key not in seen_results:
                seen_results.add(key)
                paths.append(path+('hard',))
                locks.append((piece, landing, result != 0, cleared, terminal))
                after.append((seq, 0, held))
                full.append(result)
            if len(path)-len(prefix) >= max_commands:
                continue
            for action in ('left', 'right', 'cw', 'ccw'):
                moved, locked, ok = control_edge(game.board, piece, pose, action, rules)
                k = (moved.x, moved.y, moved.rotation)
                if ok and not locked and k not in seen_poses:
                    seen_poses.add(k)
                    frontier.append((moved, path+(action,)))
    if not locks:
        raise RuntimeError('No reachable lock found')
    return paths, locks, after, full


class ClassicBot:
    """Drop-in replacement for offline.bot.Bot with the same timing and counters."""

    def __init__(self, planner=None):
        self.planner = planner or Planner()
        self.beam = self.planner.beam
        self.pending = deque()
        self.delay = self.cooldown = 0
        self.last_placement = -1
        self.plans = self.overruns = self.no_plan = 0
        self.planning_seconds = 0.0
        self.max_plan_seconds = 0.0
        self.expected_board = None
        self.path_mismatches = 0

    def command(self, game):
        if game.placements != self.last_placement:
            if self.expected_board is not None and not np.array_equal(self.expected_board, game.board):
                self.path_mismatches += 1
            self.expected_board = None
            self.last_placement = game.placements
            self.pending.clear()
            self.delay = self.cooldown = 0
        if not self.pending and self.delay == 0:
            predicted = game.pose
            for _ in range(game.rules.reaction_frames):
                predicted, locked, _ = advance(game.board, game.piece, predicted, 'none', game.rules)
                if locked:
                    self.no_plan += 1
                    return 'none'
            started = time.perf_counter()
            paths, locks, after, full = enumerate_locks(game, predicted)
            selected = self.planner.choose(game.board != 0, locks, after)
            elapsed = time.perf_counter()-started
            self.plans += 1
            self.planning_seconds += elapsed
            self.max_plan_seconds = max(self.max_plan_seconds, elapsed)
            if elapsed > game.rules.reaction_frames/game.rules.fps:
                self.overruns += 1
                self.delay = max(1, int(np.ceil(elapsed*game.rules.fps)))
                return 'none'
            self.pending = deque(paths[selected])
            self.expected_board = full[selected]
            self.delay = game.rules.reaction_frames
        if self.delay:
            self.delay -= 1
            return 'none'
        if self.cooldown:
            self.cooldown -= 1
            return 'none'
        if not self.pending:
            return 'none'
        self.cooldown = game.rules.control_frames-1
        return self.pending.popleft()


def warmup(planner=None):
    """Compile the numba helpers before a timed game, so the first plan is not an overrun."""
    from offline.engine import Game, Rules
    game = Game(0, Rules())
    bot = ClassicBot(planner or Planner())
    paths, locks, after, full = enumerate_locks(game, game.pose)
    bot.planner.choose(game.board != 0, locks, after)
