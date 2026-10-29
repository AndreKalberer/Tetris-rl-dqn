"""Log trajectories of the score bot at 1G for value-net training (level-235 physics, 8-row tetris/burn switch,
top 12 / cap 4 playouts, empty start).

Per placement it stores the board after the lock (200 bits, packed) and the lines cleared; a game ends at LINES lines
or at a top-out.  Off-policy moves: with probability EPS (only while the stack is at most MAXH rows) the move is a
random pick among the 8 best one-ply placements.  NET/LAM play with a value net at the playout leaves (policy
improvement), otherwise the plain playouts are used.

Usage: python nes-bot/tools/value_net/gen.py OUT.npz SEED0 GAMES LINES   (env: EPS, MAXH, NET, LAM, WORKERS)
"""
import json, multiprocessing as mp, os, sys
from pathlib import Path

import numpy as np

NES = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NES))
from nes_original import sim  # noqa: E402

W = json.loads((NES/'nes_original'/'weights'/'tetris-playout12-b8.json').read_text(encoding='utf-8'))
ALT = sim.pad_weights(W['alt_weights'])
BURN = sim.pad_weights(W['burn_weights'])
EPS = float(os.environ.get('EPS', 0))
MAXH = int(os.environ.get('MAXH', 20))
LAM = float(os.environ.get('LAM', 12))
NET = sim.load_value_net(os.environ['NET']) if os.environ.get('NET') else None


def pieces(seed, n=7000):
    """The NES piece sequence of one game: LFSR seeded by `seed`, 30-49 frames between spawns."""
    rng = 0x8988
    for _ in range(seed % 32767):
        rng = sim._lfsr(rng)
    out, count, last = [], 0, 0
    gaps = np.random.default_rng(seed)
    for _ in range(n):
        for _ in range(30+int(gaps.integers(0, 20))):
            rng = sim._lfsr(rng)
        rng, count, p = sim._draw(rng, count, last)
        last = p
        out.append(int(p))
    return out


def choose(board, state, upcoming, well, w):
    if NET is None:
        return sim.playout_choose(board, state, upcoming, 29, 2, 0, well, w, 12, 4, 2, -1000.)
    states = np.zeros((sim.MAX_CANDIDATES, 9), np.int64)
    n = sim.candidates(board, state, 29, 2, states, 0)
    if n == 0:
        return 0, 0
    scratch = np.zeros(sim.N_FEATURES)
    roots = np.zeros((n, 20, 10), np.bool_)
    part = np.full(n, sim.DEAD)
    das = np.zeros(n, np.int64)
    values = np.full(n, sim.DEAD)
    for i in range(n):
        result, dead, value, p = sim._scored_lock(board, states[i, 0], states[i, 1], states[i, 2], well, w, scratch)
        if not dead:
            roots[i], part[i], das[i], values[i] = result, p, states[i, 4], value
    scores = sim.playout_scores_net(roots, part, das, upcoming, 29, 2, 0, well, w, 12, 4, 2, -1000., LAM, *NET)
    return (int(np.argmax(values)) if scores.max() <= sim.DEAD else int(np.argmax(scores))), n


def game(job):
    seed, max_lines = job
    seq = pieces(seed)
    rnd = np.random.default_rng(seed+12345)
    board = np.zeros((20, 10), np.bool_)
    st = np.zeros((sim.MAX_CANDIDATES, 9), np.int64)
    lines = das = k = 0
    boards, cleared, dead_end = [], [], False
    while lines < max_lines:
        cur, upcoming = seq[k], seq[k+1]
        k += 1
        h = sim.stack_height(board)
        w, well = (ALT, 9) if h <= 8 else (BURN, -1)
        state, failed = sim.spawn_state(cur, das, board, 29)
        if failed:
            dead_end = True
            break
        idx, n = choose(board, state, upcoming, well, w)
        if n == 0:
            dead_end = True
            break
        sim.candidates(board, state, 29, 2, st, 0)
        if rnd.random() < EPS and h <= MAXH:
            vals = [sim._scored_lock(board, st[i, 0], st[i, 1], st[i, 2], well, w, np.zeros(sim.N_FEATURES))
                    for i in range(n)]
            vals = [-1e18 if d else v for _, d, v, _ in vals]
            best = [i for i in np.argsort(vals)[::-1][:8] if vals[i] > -1e17]
            if best:
                idx = int(rnd.choice(best))
        board, c, dead = sim.lock(board, st[idx, 0], st[idx, 1], st[idx, 2])
        if dead:
            dead_end = True
            break
        das = st[idx, 4]
        lines += c
        boards.append(np.packbits(board.ravel()))
        cleared.append(c)
    return seed, np.array(boards, np.uint8).reshape(-1, 25), np.array(cleared, np.int8), dead_end


if __name__ == '__main__':
    out, s0, n, max_lines = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    res = []
    with mp.get_context('spawn').Pool(int(os.environ.get('WORKERS', 4))) as pool:
        for r in pool.imap_unordered(game, [(s, max_lines) for s in range(s0, s0+n)], chunksize=1):
            res.append(r)
            print(r[0], len(r[2]), r[3], flush=True)
    np.savez_compressed(out, seeds=np.array([r[0] for r in res]), boards=np.concatenate([r[1] for r in res]),
                        cleared=np.concatenate([r[2] for r in res]), lengths=np.array([len(r[2]) for r in res]),
                        dead=np.array([r[3] for r in res]))
