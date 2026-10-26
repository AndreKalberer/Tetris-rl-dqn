"""Numba twin of the per-frame path loop in nes.actions.enumerate_actions.

Same button schedule and the same frame rules as nes.engine.control_frame, compiled
so a plan takes ~1 ms instead of ~15 ms. enumerate_actions keeps the pure-Python loop
as a fallback when numba is missing (or NES_FAST_PATHS=0) and the tests check that
both give identical paths, controls and boards.
"""
import numpy as np
import numba

from nes.engine import CELLS, GRAVITY, ROTATE_CCW, ROTATE_CW, TYPE

_CELLS = np.asarray(CELLS, dtype=np.int64)            # (19, 4, 2) as (dx, dy)
_CW = np.asarray(ROTATE_CW, dtype=np.int64)
_CCW = np.asarray(ROTATE_CCW, dtype=np.int64)
_GRAVITY = np.asarray(GRAVITY, dtype=np.int64)
_TYPE = np.asarray(TYPE, dtype=np.int64)


def _rotate_table():
    from nes.actions import rotate_button
    table = np.zeros((19, 19), dtype=np.int64)
    for a in range(19):
        for b in range(19):
            if TYPE[a] == TYPE[b]:
                table[a, b] = rotate_button(a, b)
    return table


_ROTATE = _rotate_table()


@numba.njit(cache=True)
def _collides(board, x, y, o, cells):
    for i in range(4):
        xx = x + cells[o, i, 0]
        yy = y + cells[o, i, 1]
        if xx < 0 or xx >= 10 or yy < -2 or yy >= 20:
            return True
        if yy >= 0 and board[yy, xx] != 0:
            return True
    return False


@numba.njit(cache=True)
def _frame(board, s, buttons, gravity, cells, cw, ccw):
    """control_frame on s = [x, y, orientation, fall, das, soft, points, previous] (in place)."""
    pressed = buttons & ~s[7]
    x = s[0]; y = s[1]; o = s[2]
    das = s[4]; soft = s[5]; points = s[6]
    fall = (s[3] + 1) & 255
    if not buttons & 4:
        shift = False
        if pressed & 3:
            das = 0
            shift = True
        elif buttons & 3:
            das = (das + 1) & 255
            if ((das - 16) & 255) < 128:
                das = 10
                shift = True
        if shift:
            trial = x + (1 if buttons & 1 else -1)
            if _collides(board, trial, y, o, cells):
                das = 16
            else:
                x = trial
    if pressed & 128:
        target = cw[o]
    elif pressed & 64:
        target = ccw[o]
    else:
        target = o
    if not _collides(board, x, y, target, cells):
        o = target
    forced = False
    if soft >= 128:
        if pressed & 4:
            soft = 0
        else:
            soft = (soft + 1) & 255
            s[0] = x; s[1] = y; s[2] = o; s[3] = fall; s[4] = das; s[5] = soft; s[6] = points; s[7] = buttons
            return False
    if soft:
        if buttons & 15 == 4:
            soft += 1
            if soft >= 3:
                soft = 1
                points = (points + 1) & 255
                forced = True
        else:
            soft = 0
            points = 0
    elif not buttons & 3 and pressed & 15 == 4:
        soft = 1
    failed = False
    if forced or ((fall - gravity) & 255) < 128:
        fall = 0
        if _collides(board, x, y + 1, o, cells):
            failed = True
        else:
            y += 1
    s[0] = x; s[1] = y; s[2] = o; s[3] = fall; s[4] = das; s[5] = soft; s[6] = points; s[7] = buttons
    return failed


@numba.njit(cache=True)
def _paths(board, start, gravity, targets, target_xs, reaction, navigation, rotation_period,
           horizontal_period, cells, cw, ccw, rotate):
    n = targets.shape[0]
    limit = reaction + navigation + 160
    paths = np.zeros((n, limit), dtype=np.int64)
    lengths = np.zeros(n, dtype=np.int64)
    finals = np.zeros((n, 8), dtype=np.int64)
    s = np.empty(8, dtype=np.int64)
    for c in range(n):
        target = targets[c]
        tx = target_xs[c]
        for i in range(8):
            s[i] = start[i]
        locked = False
        for frame in range(limit):
            elapsed = frame - reaction
            if elapsed < 0:
                buttons = 0
            elif elapsed >= navigation:
                buttons = 4
            else:
                horizontal = 2 if s[0] > tx else (1 if s[0] < tx else 0)
                moving = horizontal != 0
                if horizontal_period and elapsed % horizontal_period:
                    horizontal = 0
                if s[2] != target:
                    rot = rotate[s[2], target]
                    pulse = rot if (elapsed % rotation_period == 0 and not s[7] & rot) else 0
                    buttons = horizontal | pulse
                else:
                    buttons = horizontal if moving else 4
            paths[c, frame] = buttons
            if _frame(board, s, buttons, gravity, cells, cw, ccw):
                lengths[c] = frame + 1
                locked = True
                break
        if not locked:
            lengths[c] = -1
        for i in range(8):
            finals[c, i] = s[i]
    return paths, lengths, finals


def timed_paths(board, control, level, piece, rules):
    """Paths for every (orientation, x) target of `piece`, in enumerate_actions order.

    Returns a list of (path tuple, final control tuple) per target.
    """
    targets, xs = [], []
    for target in range(19):
        if TYPE[target] != piece: continue
        cells = CELLS[target]
        for tx in range(-min(dx for dx, dy in cells), 9-max(dx for dx, dy in cells)+1):
            targets.append(target); xs.append(tx)
    start = np.asarray((control.x, control.y, control.orientation, control.fall, control.das,
                        control.soft, control.soft_points, control.previous), dtype=np.int64)
    paths, lengths, finals = _paths(np.ascontiguousarray(board, dtype=np.uint8), start,
        int(_GRAVITY[min(level, 29)]), np.asarray(targets, dtype=np.int64), np.asarray(xs, dtype=np.int64),
        rules.reaction_frames, rules.navigation_frames, rules.rotation_period, rules.horizontal_period,
        _CELLS, _CW, _CCW, _ROTATE)
    return paths, lengths, finals
