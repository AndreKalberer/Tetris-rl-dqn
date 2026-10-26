"""Recorded simulator outcomes for the regression tests in tests/test_sim_regression.py.

The fast simulator (nes_original.sim) and the crash-zone simulator (tools/zone_sim.py)
are deterministic per seed, so a fixed set of seeded games, crash-zone runs and
reachability limits pins their behaviour. An intended behaviour change (for example
a shorter spawn reaction wait) must re-record the values with

  python nes-bot/tools/sim_golden.py --write

after the survival checks in the same test file still pass. Values are stored per
reaction wait (frames from spawn before the first input), so goldens for several
reaction settings can live side by side.
"""
import argparse
import hashlib
import os
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # nes-bot/
REPO = ROOT.parent  # repository root: runs/ and progress.md live here
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))
if 'NUMBA_CACHE_DIR' not in os.environ:
    # Same per-source cache folder as tests/conftest.py, so recordings never use stale compiled code.
    _digest = hashlib.sha1()
    for _path in sorted((ROOT/'nes').glob('*.py'))+sorted((ROOT/'nes_original').glob('*.py')):
        _digest.update(_path.name.encode()+_path.read_bytes())
    os.environ['NUMBA_CACHE_DIR'] = str(REPO/'.numba-cache'/_digest.hexdigest()[:16])
from nes_original import sim  # noqa: E402

GOLDEN = ROOT/'tests'/'data'/'sim_golden.json'
WEIGHTS = ROOT/'nes_original'/'weights'
PIECES = dict(T=2, J=7, Z=8, O=10, S=11, L=14, I=18)


def reaction():
    """Frames the controller waits after spawn before its first input (3 until the wait is removed)."""
    return int(getattr(sim, 'REACTION', 3))


# v8c crash-zone configuration, as in tools/run255.sh.
def v8c_config():
    w = str(WEIGHTS)+'/'
    return dict(beam=5, zone_weights=w+'nosingles-156-158.json', zone_levels=(156, 158), zone_ban=2, zone_expect=4,
                extra_zones=[(w+'lowstack-246-249.json', 246, 249, 0, 4), (w+'dellacherie.json', 250, 254, 0, 4),
                             (w+'dellacherie.json', 168, 172, 28, 2), (w+'dellacherie.json', 173, 173, 4, 2)],
                preempt=True, risk_budget=True, trap_penalty=30., budget_table=None)


# Seeded fast-simulator games: name -> play_game keyword arguments.
D = sim.DELLACHERIE
GAMES = {
    'default-l18-beam0': dict(seed=3, start_level=18, max_lines=120),
    'dellacherie-l18-beam2-s0': dict(w_score=D, w_survive=D, well=-1, well_survive=-1, seed=0, start_level=18,
                                     max_lines=200, beam=2),
    'dellacherie-l18-beam2-s1': dict(w_score=D, w_survive=D, well=-1, well_survive=-1, seed=1, start_level=18,
                                     max_lines=200, beam=2),
    'dellacherie-l19-to-30': dict(w_score=D, w_survive=D, well=-1, well_survive=-1, seed=7, start_level=19,
                                  max_lines=260, beam=1),
    'dellacherie-1g-beam5-s1': dict(w_score=D, w_survive=D, well=-1, well_survive=-1, seed=1, start_level=29,
                                    max_lines=300, beam=5),
    'dellacherie-1g-beam5-s2': dict(w_score=D, w_survive=D, well=-1, well_survive=-1, seed=2, start_level=29,
                                    max_lines=300, beam=5),
    'dellacherie-1g-expect2': dict(w_score=D, w_survive=D, well=-1, well_survive=-1, seed=4, start_level=29,
                                   max_lines=120, beam=3, expect=2),
    'score-start-alt4-1g': dict(w_score=D, w_survive=D, well=-1, well_survive=-1, seed=5, start_level=29,
                                max_lines=300, beam=5, initial_score=sim.MAXED, alt_weights=sim.SCORE_START,
                                alt_below=4),
    'score-to-survive-switch': dict(w_score=sim.SCORE_START, w_survive=D, well=9, well_survive=-1, seed=6,
                                    start_level=18, max_lines=200, beam=2, switch_level=24),
}
# Crash-zone games with the v8c bot: (zone, seed). Early 1012 and 1055 are baseline losses.
ZONE_GAMES = [('early', s) for s in range(1000, 1008)]+[('early', 1012), ('early', 1055)]+[('late', s) for s in range(2000, 2004)]


def play_named(name):
    return sim.play_game(**GAMES[name])


def zone_game(zone, seed, config=None):
    import zone_sim
    config = dict(v8c_config() if config is None else config, record_moves=True)
    r = zone_sim.play((seed, zone, config))
    moves = json.dumps(r.pop('moves'), sort_keys=True).encode()
    r.pop('snapshots', None)
    r.pop('points', None); r.pop('bands', None)   # score tally added with the full-game mode; not part of the recording
    r['moves_sha1'] = hashlib.sha1(moves).hexdigest()
    return r


def reaction_argument():
    """True once sim.candidates takes the reaction wait as an argument (after tap_period)."""
    import inspect
    func = getattr(sim.candidates, 'py_func', sim.candidates)
    return 'reaction' in inspect.signature(func).parameters


def candidates(board, state, level, states, reaction_frames=None):
    if reaction_argument():
        return sim.candidates(board, state, level, 2, states, reaction() if reaction_frames is None else reaction_frames)
    if reaction_frames not in (None, reaction()):
        raise ValueError(f'this sim has a fixed {reaction()}-frame reaction wait')
    return sim.candidates(board, state, level, 2, states)


def placed_columns(board, orientation, level, das=0, reaction_frames=None):
    """Columns a piece can put a cell in, from the real controller's planning pose (one frame after spawn)."""
    state, failed = sim.frame(board, (5, 0, int(orientation), 0, das, 0, 0, 0), 0, level)
    if failed:
        return set()
    states = np.zeros((sim.MAX_CANDIDATES, 9), np.int64)
    n = candidates(board, state, level, states, reaction_frames)
    cols = set()
    for i in range(n):
        x, y, o = (int(v) for v in states[i, :3])
        if sim.lock(board, x, y, o)[2]:
            continue
        cols.update(x+int(sim.SHAPES[o, k, 0]) for k in range(4))
    return cols


def well_limit(piece, column, level, reaction_frames=None):
    """Tallest flat stack (every column but `column`) over which `piece` still puts a cell in `column`."""
    best = -1
    for height in range(0, 19):
        board = np.zeros((20, 10), np.bool_)
        board[20-height:, :] = True
        board[:, column] = False
        if column in placed_columns(board, PIECES[piece], level, reaction_frames=reaction_frames):
            best = height
        else:
            break
    return best


def reach_table(level, reaction_frames=None):
    return {p: [well_limit(p, c, level, reaction_frames) for c in range(10)] for p in PIECES}


REACH_LEVELS = (18, 19, 29)
CRASH_LEVELS = tuple(range(150, 176))+tuple(range(246, 256))


def crash_fingerprint():
    """Crash model on a grid of clears per crash-zone level: max probability and a digest of the whole grid,
    plus zone_sim's zero-line table. v8c games rarely take a modelled risk, so seeded games alone miss this."""
    import zone_sim
    from nes_original.crash_model import model_crash_probability
    levels = {}
    for level in CRASH_LEVELS:
        first = zone_sim.LEVEL_AT.index(level)
        values = []
        for lines in (first, first+5):
            for cleared in range(5):
                for rows_high in range(cleared+1):
                    for carry in (False, True):
                        for piece in (('TJ', 'hZ', 'other') if cleared == 0 else (None,)):
                            for pushdown in (0, 2, 4):
                                values.append(round(float(model_crash_probability(level, lines, cleared,
                                    rows_high=rows_high, pushdown_raw=pushdown, clear_count_carry=carry,
                                    piece_class=piece)), 6))
        levels[str(level)] = dict(max=max(values), sha1=hashlib.sha1(json.dumps(values).encode()).hexdigest())
    zero = {str(k): v for k, v in zone_sim.ZERO_LINE_CRASH.items()}
    return dict(levels=levels, zero_line=zero)


def record():
    return dict(
        games={name: play_named(name) for name in GAMES},
        zone_games={f'{z}-{s}': zone_game(z, s) for z, s in ZONE_GAMES},
        reach={str(level): reach_table(level) for level in REACH_LEVELS},
        crash=crash_fingerprint(),
    )


def load():
    return json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--write', action='store_true', help=f'store the values for this reaction in {GOLDEN}')
    args = p.parse_args()
    values = record()
    key = f'reaction_{reaction()}'
    if args.write:
        data = load()
        data[key] = values
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(data, indent=1, sort_keys=True)+'\n')
        print(f'wrote {key} to {GOLDEN}')
    else:
        print(json.dumps({key: values}, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
