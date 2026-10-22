"""Experimental bounded risk-aware search; provisional geometry, native actions.

Only board/current/one visible preview are read. The third piece is seven
hypotheses, never a queried RNG/future queue. No defaults elsewhere are changed.
"""
import time

import numpy as np
from numba import njit

from nes.actions import ActionRules
from nes.engine import SPAWN, TYPE
from nes_original import fast
from nes_original.progression import after_clear

SPAWNS = np.asarray(SPAWN, dtype=np.int64)


@njit(cache=True)
def _afterboard(board, state):
    x, y, orientation = state[:3]
    result = board.copy()
    if fast.collision(board, x, y, orientation):
        return result, 0, True
    for i in range(4):
        xx = x + fast.SHAPES[orientation, i, 0]
        yy = y + fast.SHAPES[orientation, i, 1]
        if yy >= 0:
            result[yy, xx] = True
    cleared = 0
    for row in range(max(0, y-2), max(0, y-2)+4):
        if row >= 20:
            continue
        full = True
        for col in range(10):
            if not result[row, col]:
                full = False
                break
        if full:
            for r in range(row, 0, -1):
                for col in range(10):
                    result[r, col] = result[r-1, col]
            for col in range(10):
                result[0, col] = False
            cleared += 1
    return result, cleared, False


@njit(cache=True)
def _height_cost(board, safe_height, height_weight):
    maximum = 0
    for col in range(10):
        for row in range(20):
            if board[row, col]:
                maximum = max(maximum, 20-row)
                break
    return height_weight * max(0, maximum-safe_height)**2


@njit(cache=True)
def _candidates(board, initial, level, period, reaction, navigation, rotation_period,
                line_power, safe_height, height_weight, terminal_cost):
    """Enumerate the same static-target control family as nes.actions."""
    boards = np.empty((40, 20, 10), dtype=np.bool_)
    controls = np.empty((40, 8), dtype=np.int64)
    lines = np.empty(40, dtype=np.int64)
    dead = np.empty(40, dtype=np.bool_)
    scores = np.empty(40, dtype=np.float64)
    count = 0
    kind = fast.KINDS[initial[2]]
    for target in range(19):
        if fast.KINDS[target] != kind:
            continue
        left, right = 10, -10
        for i in range(4):
            left = min(left, fast.SHAPES[target, i, 0])
            right = max(right, fast.SHAPES[target, i, 0])
        for target_x in range(-left, 10-right):
            state = initial
            locked = False
            for frame in range(reaction+navigation+160):
                elapsed = frame-reaction
                if elapsed < 0:
                    buttons = 0
                elif elapsed >= navigation:
                    buttons = 4
                else:
                    horizontal = 2 if state[0] > target_x else 1 if state[0] < target_x else 0
                    moving = horizontal != 0
                    if period and elapsed % period:
                        horizontal = 0
                    if state[2] != target:
                        rotation = fast.ROTATE[state[2], target]
                        pulse = rotation if elapsed % rotation_period == 0 and not state[7] & rotation else 0
                        buttons = horizontal | pulse
                    else:
                        buttons = horizontal if moving else 4
                state, locked = fast.frame(board, state, buttons, level)
                if locked:
                    break
            if not locked:
                raise RuntimeError('Risk-search provisional target exceeded its bound')
            result, cleared, terminal = _afterboard(board, state)
            boards[count] = result
            for i in range(8):
                controls[count, i] = state[i]
            lines[count], dead[count] = cleared, terminal
            scores[count] = (fast.lock_score(board, state, line_power, -1)
                - _height_cost(result, safe_height, height_weight) - (terminal_cost-1000.0)*terminal)
            count += 1
    return boards[:count], controls[:count], lines[:count], dead[:count], scores[:count]


@njit(cache=True)
def _spawn(board, pose, das, level):
    state = (5, 0, pose, 0, das, 0, 0, 0)
    if fast.collision(board, 5, 0, pose):
        return state, True
    for _ in range(2):
        state, failed = fast.frame(board, state, 0, level)
        if failed:
            return state, True
    return state, False


@njit(cache=True)
def _hypotheses(board, das, level, period, reaction, navigation, rotation_period,
                line_power, safe_height, height_weight, terminal_cost):
    values = np.empty(7, dtype=np.float64)
    for kind in range(7):
        state, failed = _spawn(board, SPAWNS[kind], das, level)
        if failed:
            values[kind] = -terminal_cost
        else:
            _, _, _, _, scores = _candidates(board, state, level, period, reaction,
                navigation, rotation_period, line_power, safe_height, height_weight, terminal_cost)
            values[kind] = np.max(scores)
    return values


class RiskAwareNextPiece:
    """A context_selector for FeedbackController; not a promoted policy.

    risk=0 uses a seven-type arithmetic mean; risk=1 uses the worst type.
    The average is a design objective, not an asserted NES randomizer model.
    Peak calibration is provisional: above height6, .15*excess**2; predicted
    terminal cost1000 matches the retained geometric prior. Actual emulator
    results, latency charges and audits remain authoritative.
    """
    def __init__(self, root_beam=4, future_beam=3, risk=.5, tap_period=2,
                 safe_height=6, height_weight=.15, terminal_cost=1000., line_power=1):
        if not 1 <= root_beam <= 40 or not 1 <= future_beam <= 40:
            raise ValueError('Use bounded root/future beams in1..40')
        if (not all(np.isfinite(value) for value in (risk, safe_height, height_weight, terminal_cost))
                or not 0 <= risk <= 1 or not 0 <= safe_height <= 20 or height_weight < 0 or terminal_cost <= 0):
            raise ValueError('Invalid risk or safety calibration')
        if line_power not in (1, 2):
            raise ValueError('Line power must be1 or2')
        self.rules = ActionRules(horizontal_period=tap_period)
        self.root_beam, self.future_beam, self.risk = root_beam, future_beam, float(risk)
        self.safe_height, self.height_weight = safe_height, float(height_weight)
        self.terminal_cost, self.line_power = float(terminal_cost), line_power
        self.calls = self.hypothesis_searches = 0
        self.times = []
        self.last_details = []
        # Warm full explicit signatures outside any charged decision interval.
        board = np.zeros((20, 10), dtype=np.bool_)
        state, _ = _spawn(board, 2, 0, 29)
        _candidates(board, state, 29, *self._arguments())
        _hypotheses(board, 0, 29, *self._arguments())

    def _arguments(self):
        r = self.rules
        return (r.horizontal_period, r.reaction_frames, r.navigation_frames, r.rotation_period,
                self.line_power, self.safe_height, self.height_weight, self.terminal_cost)

    def __call__(self, obs, plan):
        started = time.perf_counter()
        actions = plan.actions
        occupied = actions.boards != 0
        heights = np.where(occupied.any(axis=1), 20-occupied.argmax(axis=1), 0).max(axis=1)
        root_scores = (actions.heuristic + .76*(actions.lines**self.line_power-actions.lines)
            - self.height_weight*np.maximum(0, heights-self.safe_height)**2
            - (self.terminal_cost-1000.)*actions.terminal)
        shortlist, seen = [], set()
        for i in np.argsort(-root_scores, kind='stable'):
            if actions.terminal[i]:
                continue
            key = (occupied[i].tobytes(), actions.controls[i].das)
            if key not in seen:
                seen.add(key)
                shortlist.append(int(i))
            if len(shortlist) >= self.root_beam:
                break
        details = []
        for i in shortlist:
            level, lines = after_clear(obs.level, obs.lines, int(actions.lines[i]))
            state, failed = _spawn(occupied[i], SPAWN[TYPE[obs.next_orientation]], actions.controls[i].das, level)
            best = -np.inf
            if not failed:
                boards, controls, clears, terminal, scores = _candidates(occupied[i], state, level, *self._arguments())
                candidates, known_seen = [], set()
                for child in np.argsort(-scores, kind='stable'):
                    if terminal[child]:
                        continue
                    key = (boards[child].tobytes(), int(controls[child, 4]))
                    if key not in known_seen:
                        known_seen.add(key)
                        candidates.append(int(child))
                    if len(candidates) >= self.future_beam:
                        break
                for child in candidates:
                    next_level, _ = after_clear(level, lines, int(clears[child]))
                    values = _hypotheses(boards[child], int(controls[child, 4]), next_level, *self._arguments())
                    self.hypothesis_searches += 7
                    continuation = (1-self.risk)*float(values.mean()) + self.risk*float(values.min())
                    best = max(best, .76*int(clears[child])**self.line_power + continuation)
            if not np.isfinite(best):
                best = -self.terminal_cost
            score = .76*int(actions.lines[i])**self.line_power + best
            details.append(dict(index=i, score=score))
        self.calls += 1
        self.times.append(time.perf_counter()-started)
        self.last_details = details
        return (details[int(np.argmax([row['score'] for row in details]))]['index']
                if details else int(root_scores.argmax()))

    def stats(self):
        return dict(root_beam=self.root_beam, future_beam=self.future_beam, risk=self.risk,
            safe_height=self.safe_height, height_weight=self.height_weight, terminal_cost=self.terminal_cost,
            line_power=self.line_power, calls=self.calls, hypothetical_type_searches=self.hypothesis_searches,
            selector_p50_ms=float(np.median(self.times)*1000) if self.times else 0,
            selector_p95_ms=float(np.percentile(self.times, 95)*1000) if self.times else 0,
            selector_max_ms=max(self.times, default=0)*1000,
            information='Current and one visible preview; seven hypothetical unknown types, no RNG/queue reads',
            status='Experimental provisional search; no established native survival gain')
