"""The numba path loop (nes.fast_paths) must match the pure-Python loop exactly."""
import numpy as np
import pytest

pytest.importorskip('numba')

import nes.actions as A
from nes import fast_paths
from nes.engine import Control, Observation, Phase, SPAWN, collides


def _cases(count, seed=7):
    rng = np.random.default_rng(seed)
    made = 0
    while made < count:
        board = np.zeros((20, 10), np.uint8)
        heights = rng.integers(0, rng.choice([4, 8, 12, 17]), 10)
        for x in range(10):
            for y in range(20-heights[x], 20):
                if rng.random() < .85: board[y, x] = rng.integers(1, 8)
        for y in range(20):
            if board[y].all(): board[y, rng.integers(10)] = 0
        if rng.random() < .3:
            y = 19-rng.integers(0, 4); board[y, :] = 1; board[y, rng.integers(10)] = 0
        orientation = int(rng.choice(SPAWN)) if rng.random() < .6 else int(rng.integers(19))
        x = int(rng.integers(0, 10)) if rng.random() < .3 else 5
        y = int(rng.integers(-1, 8)) if rng.random() < .3 else 0
        if collides(board, x, y, orientation): continue
        control = Control(x, y, orientation, int(rng.integers(0, 60)), int(rng.integers(0, 17)),
            int(rng.choice([0, 1, 2, 0xA0, 200, 255])), int(rng.integers(0, 30)),
            int(rng.choice([0, 0, 1, 2, 4, 64, 128, 5])))
        level = int(rng.choice([0, 9, 16, 18, 19, 25, 29, 100, 154, 255]))
        rules = A.ActionRules(reaction_frames=int(rng.choice([0, 1, 3])),
            navigation_frames=int(rng.choice([48, 20])), rotation_period=int(rng.choice([4, 2, 5])),
            horizontal_period=int(rng.choice([0, 2, 3])))
        made += 1
        yield Observation(board, control, int(rng.integers(7)), Phase.ACTIVE, level,
                          int(rng.integers(0, 300)), 0, 0, 0, int(rng.integers(0, 20))), rules


def _run(observation, rules, module, monkeypatch):
    monkeypatch.setattr(A, '_FAST', module)
    try: return A.enumerate_actions(observation, rules, board_input=True)
    except RuntimeError as error: return str(error)


def test_numba_paths_match_python(monkeypatch):
    for observation, rules in _cases(400):
        slow = _run(observation, rules, False, monkeypatch)
        fast = _run(observation, rules, fast_paths, monkeypatch)
        if isinstance(slow, str) or isinstance(fast, str):
            assert slow == fast
            continue
        assert slow.paths == fast.paths and slow.controls == fast.controls
        for name in ('boards', 'lines', 'terminal', 'features', 'heuristic'):
            assert np.array_equal(getattr(slow, name), getattr(fast, name)), name
