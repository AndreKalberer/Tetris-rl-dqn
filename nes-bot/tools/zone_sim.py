"""Crash-zone simulator: play the risky late-game stretches with the real bot's decision code.

Uses the same selector objects as the original-game runs (sim.tuned_selector wrapped
by crash_model.CrashMapPrioritizer, with the same zones), fed by the fast simulator's
candidate moves (parity-tested against nes.actions). The exact original level/line
counter (progression.after_clear) drives levels. Crashes are sampled from the
HydrantDude model: each chosen line clear crashes with its modelled probability
(true clear-count carry and row height), and at 249/253/255 a zero-line non-T/J
lock crashes with its modelled probability (ZERO_LINE_CRASH) when its actual pushdown
is below 2. Actual pushdown = predicted minus 0 or 1 (measured on 199 real locks).

Simulation only: piece sequences and timing between pieces are approximate.

  python nes-bot/tools/zone_sim.py --zone early --games 400 --output runs/zone-early-v6.json
  python nes-bot/tools/zone_sim.py --zone late  --games 400 --output runs/zone-late-v6.json
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # nes-bot/
REPO = ROOT.parent  # repository root: runs/ and progress.md live here
sys.path.insert(0, str(ROOT))
from nes.engine import SPAWN, TYPE  # noqa: E402
from nes_original import sim  # noqa: E402
from nes_original.crash_model import NEEDS_PUSHDOWN, CrashMapPrioritizer, piece_class  # noqa: E402
from nes_original.progression import after_clear  # noqa: E402

# start lines (warm-up included), stop level. Lap 2 (after the 255 -> 0 rebirth at 3,300 lines):
# level L starts at 3,300 + 10L, except level 189 which lasts 810 lines (5,190-5,999).
ZONES = {'early': (1340, 176), 'late': (3140, 255), 'mid': (1440, 160),
         'lap2early': (4700, 176), 'lap2late': (6500, 255)}
# Modelled fatal probability of a zero-line lock without pushdown (outputs/NES-CRASH-MAP-RESEARCH.json).
# Level bands for the score breakdown (raw level byte; repeats every lap).
BANDS = ((18, 28), (29, 130), (131, 153), (154, 175), (176, 235), (236, 245), (246, 255), (0, 17))
ZERO_LINE_CRASH = {249: {'hZ': .75, 'other': .583}, 253: {'other': .104}, 255: {'hZ': .104, 'other': .75}}
LEVEL_AT = [18]
for _ in range(20000):
    LEVEL_AT.append(after_clear(LEVEL_AT[-1], len(LEVEL_AT)-1, 1)[0])


class Control:
    __slots__ = ('x', 'y', 'orientation', 'das', 'soft_points')

    def __init__(self, s):
        self.x, self.y, self.orientation, self.das, self.soft_points = int(s[0]), int(s[1]), int(s[2]), int(s[4]), int(s[6])


_SELECTORS = {}


def _selector(config):
    key = json.dumps({k: config.get(k) for k in ('beam', 'zone_weights', 'zone_levels', 'zone_ban', 'zone_expect',
                                                 'extra_zones', 'reaction', 'alt_weights', 'alt_levels', 'playout')},
                     sort_keys=True, default=str)
    if key not in _SELECTORS:
        _SELECTORS[key] = sim.tuned_selector('dellacherie', beam=config['beam'], zone_weights=config['zone_weights'],
            zone_levels=tuple(config['zone_levels']), zone_ban=config['zone_ban'], zone_expect=config['zone_expect'],
            extra_zones=[tuple(z) for z in config['extra_zones']], reaction=config.get('reaction', 3),
            alt_weights=config.get('alt_weights'), alt_levels=config.get('alt_levels'), **(config.get('playout') or {}))
    return _SELECTORS[key]


def play(job):
    seed, zone, config = job
    rng = np.random.default_rng(seed)
    start_lines, stop_level = config['zone_def'] if config.get('zone_def') else ZONES[zone]
    base = _selector(config)
    prio = CrashMapPrioritizer(base, preempt=config['preempt'], risk_budget=config['risk_budget'],
                               trap_penalty=config.get('trap_penalty', 30.))
    if config.get('budget_table'):
        prio.budget_table = tuple(tuple(x) for x in config['budget_table'])
    full = config.get('stop_lines') is not None
    if full:
        # Whole game from the level-18 start: true counters, stop at an absolute line count.
        lines, placements = 0, 0
        prio.counts = [0, 0, 0, 0, 0]
    else:
        lines = start_lines + int(rng.integers(0, 10)) if config.get('start_board') is None else start_lines
        placements = 3000
        prio.counts = [0, int(rng.integers(900, 1300)), int(rng.integers(150, 260)), int(rng.integers(5, 20)), 0]
    level = LEVEL_AT[lines]
    points = 0
    bands = [[0, 0, 0, 0, 0, 0, 0, 0] for _ in BANDS]   # points, lines, tetris lines, placements, clears 1-4
    reaction = int(config.get('reaction', 3))
    prio.counts_known, prio.previous = True, (placements, lines)
    board = np.zeros((20, 10), np.bool_)
    if config.get('start_board') is not None:
        board = np.asarray(config['start_board'], dtype=np.bool_)
        if config.get('start_counts'):
            prio.counts = list(config['start_counts'])
    lfsr = 0x8988
    for _ in range(int(rng.integers(1, 30000))):
        lfsr = sim._lfsr(lfsr)
    count = 0
    lfsr, count, current = sim._draw(lfsr, count, 0)
    lfsr, count, upcoming = sim._draw(lfsr, count, current)
    forced = list(config.get('piece_sequence') or [])
    if forced:
        current = forced.pop(0)
        upcoming = forced.pop(0) if forced else upcoming
    das = 0
    states = np.zeros((sim.MAX_CANDIDATES, 9), np.int64)
    snapshots = {}
    moves = []
    def end(result, **extra):
        return dict(seed=seed, result=result, level=level, lines=lines, points=points, snapshots=snapshots,
                    moves=moves, bands={f'{a}-{b}': v for (a, b), v in zip(BANDS, bands)}, **extra)

    while (lines < config['stop_lines']) if full else (level < stop_level):
        for t in (1480, 1500, 1510):
            if t not in snapshots and lines >= t and lines < t+20:
                filled = board.any(axis=1)
                height = 0 if not filled.any() else 20-int(np.argmax(filled))
                holes = int((np.maximum.accumulate(board, axis=0) & ~board).sum())
                snapshots[t] = (height, holes)
        gravity_level = min(level, 29)
        # The real controller plans one frame after spawn (piece row 1 in every traced
        # decision), not after the two settle frames sim.spawn_state uses.
        state, failed = sim.frame(board, (5, 0, int(current), 0, das, 0, 0, 0), 0, gravity_level)
        if failed:
            return end('top_out')
        # Optional: a late plan (over one frame) loses the first input frame for this piece.
        late = config.get('late_prob') and rng.random() < config['late_prob']
        n = sim.candidates(board, state, gravity_level, 2, states, reaction+1 if late else reaction)
        boards, cleared, dead, controls, frames = [], [], [], [], []
        seen = {}
        for i in range(n):
            result, c, d = sim.lock(board, int(states[i, 0]), int(states[i, 1]), int(states[i, 2]))
            # Same candidate list as nes.actions.enumerate_actions: one entry per (board, das,
            # terminal), first position kept, the shortest button path wins.
            key = (result.tobytes(), int(states[i, 4]), bool(d))
            if key in seen:
                j = seen[key]
                if int(states[i, 8]) < frames[j]:
                    controls[j], frames[j] = Control(states[i]), int(states[i, 8])
                continue
            seen[key] = len(boards)
            boards.append(result.astype(np.uint8)); cleared.append(c); dead.append(d); controls.append(Control(states[i]))
            frames.append(int(states[i, 8]))
        if all(dead):
            return end('top_out')
        actions = SimpleNamespace(boards=boards, lines=np.asarray(cleared), terminal=np.asarray(dead), controls=controls)
        plan = SimpleNamespace(actions=actions)
        obs = SimpleNamespace(board=board.astype(np.uint8), level=level, lines=lines, score=999999,
                              placements=placements, next_orientation=int(upcoming))
        index = prio(obs, plan)
        row = prio.last_rows[index]
        if config.get('record_moves'):
            moves.append(dict(lines=lines, piece=int(current), x=controls[index].x, y=controls[index].y,
                              orientation=controls[index].orientation, cleared=int(cleared[index]), das=int(das),
                              next=int(upcoming), counts=list(prio.counts)))
        if dead[index]:
            return end('top_out')
        # Crash sampling with the true conditions for the chosen move.
        p = row['modelled_crash_probability'] if cleared[index] else 0.
        if not cleared[index] and level in NEEDS_PUSHDOWN and piece_class(controls[index].orientation) in NEEDS_PUSHDOWN[level]:
            # Measured: actual pushdown is 0 (65%) or 1 (35%) below the prediction.
            actual = max(0, controls[index].soft_points - int(rng.random() < .35))
            if actual < 2:
                p = ZERO_LINE_CRASH[level].get(piece_class(controls[index].orientation), 0.)
        if config.get('record_moves'):
            moves[-1]['crash_p'] = round(float(p), 9)
        if p and rng.random() < p:
            return end('crash', clear=int(cleared[index]),
                       alert=row.get('alert') or (row.get('accepted_risk') or {}).get('alert'))
        board = boards[index].astype(np.bool_)
        das = controls[index].das
        level, lines = LEVEL_AT[lines+int(cleared[index])], lines+int(cleared[index])
        gained = int(sim.BASE_POINTS[int(cleared[index])])*(level+1)
        points += gained
        if cleared[index]:
            band = next(k for k, (a, b) in enumerate(BANDS) if a <= level <= b)
            bands[band][0] += gained
            bands[band][1] += int(cleared[index])
            bands[band][2] += 4 if cleared[index] == 4 else 0
            bands[band][3+int(cleared[index])] += 1
        placements += 1
        for _ in range(frames[index] + 14 + (20 if cleared[index] else 0)):
            lfsr = sim._lfsr(lfsr)
        current = upcoming
        lfsr, count, upcoming = sim._draw(lfsr, count, current)
        if forced:
            upcoming = forced.pop(0)
    return end('pass')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--zone', choices=sorted(ZONES)+['custom'], default='custom')
    p.add_argument('--games', type=int, default=200)
    p.add_argument('--seed-base', type=int, default=0)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--zone-weights', default=str(REPO/'runs/tune-nosingles-v2/weights.json'))
    p.add_argument('--zone-levels', type=int, nargs=2, default=(156, 158))
    p.add_argument('--zone-ban', type=int, default=2)
    p.add_argument('--zone-expect', type=int, default=2)
    p.add_argument('--extra-zone', nargs='+', action='append', default=None,
                   help='WEIGHTS FIRST LAST [CHILD_BAN_MASK [EXPECT]]; repeat for several zones')
    p.add_argument('--beam', type=int, default=3)
    p.add_argument('--no-preempt', action='store_true')
    p.add_argument('--no-risk-budget', action='store_true')
    p.add_argument('--trap-penalty', type=float, default=30.)
    p.add_argument('--budget', type=float, nargs='+', help='risk budget pairs: rows risk rows risk ...')
    p.add_argument('--start-board', type=Path, help='JSON board (20 rows of 0/1 or "#." strings) to start from')
    p.add_argument('--start-lines', type=int, help='line counter to start from (with --start-board)')
    p.add_argument('--start-counts', type=int, nargs=5, help='clears so far by type (index 0..4)')
    p.add_argument('--stop-level', type=int)
    p.add_argument('--piece-sequence', type=Path, help='JSON list of spawn orientations to play first')
    p.add_argument('--reaction', type=int, default=3, help='frames before the first input after spawn')
    p.add_argument('--alt-weights', help='low-stack tetris weights.json used outside the zones')
    p.add_argument('--alt-levels', type=int, nargs='+', help='level ranges for the tetris weights: lo hi lo hi ...')
    p.add_argument('--stop-lines', type=int, help='play a whole game from level 18 to this absolute line count')
    p.add_argument('--playout-top', type=int, default=0,
                   help='StackRabbit-style playouts in the tetris/burn modes: pairs played out (0 = off, beam/expect)')
    p.add_argument('--playout-cap', type=int, default=4, help='max played-out pairs per current-piece placement')
    p.add_argument('--playout-length', type=int, default=2, help='unknown pieces per playout (7**N sequences)')
    p.add_argument('--playout-death', type=float, default=-1000., help='value of a playout that tops out')
    p.add_argument('--late-prob', type=float, default=0.,
                   help='chance per piece that the plan is one frame late (its first input frame is lost)')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    extra = args.extra_zone if args.extra_zone is not None else [[str(REPO/'runs/lowstack-246-249.json'), '246', '249']]
    config = dict(beam=args.beam, zone_weights=args.zone_weights or None, zone_levels=args.zone_levels,
                  zone_ban=args.zone_ban, zone_expect=args.zone_expect,
                  extra_zones=[tuple(z) for z in extra],
                  preempt=not args.no_preempt, risk_budget=not args.no_risk_budget, trap_penalty=args.trap_penalty,
                  budget_table=[(int(args.budget[i]), args.budget[i+1]) for i in range(0, len(args.budget), 2)]
                  if args.budget else None,
                  reaction=args.reaction, alt_weights=args.alt_weights,
                  alt_levels=[(args.alt_levels[i], args.alt_levels[i+1]) for i in range(0, len(args.alt_levels), 2)]
                  if args.alt_levels else None, stop_lines=args.stop_lines,
                  playout=dict(playout_top=args.playout_top, playout_cap=args.playout_cap,
                               playout_length=args.playout_length, playout_death=args.playout_death)
                  if args.playout_top else None, late_prob=args.late_prob)
    if args.start_board:
        data = json.loads(args.start_board.read_text())
        rows = data['rows'] if isinstance(data, dict) else data
        config['start_board'] = [[1 if c in '#1' else 0 for c in r] if isinstance(r, str) else r for r in rows]
        config['start_counts'] = args.start_counts
        if args.piece_sequence:
            config['piece_sequence'] = json.loads(args.piece_sequence.read_text())
        config['zone_def'] = (args.start_lines, args.stop_level or 160)
        args.zone = 'custom'
    jobs = [(args.seed_base+i, args.zone, config) for i in range(args.games)]
    with mp.Pool(args.workers) as pool:
        rows = pool.map(play, jobs, chunksize=1)
    passes = sum(r['result'] == 'pass' for r in rows)
    fails = [r for r in rows if r['result'] != 'pass']
    by_level = {}
    for r in fails:
        key = f"{r['result']}@{r['level']}"
        by_level[key] = by_level.get(key, 0)+1
    summary = dict(zone=args.zone, games=len(rows), passes=passes, pass_rate=passes/len(rows),
                   failures=dict(sorted(by_level.items(), key=lambda kv: -kv[1])),
                   mean_points=float(np.mean([r['points'] for r in rows])),
                   mean_lines=float(np.mean([r['lines'] for r in rows])),
                   bands={k: dict(points_m=round(float(np.mean([r['bands'][k][0] for r in rows]))/1e6, 2),
                                  lines=round(float(np.mean([r['bands'][k][1] for r in rows])), 1),
                                  clears=[round(float(np.mean([r['bands'][k][j] for r in rows])), 1) for j in (4, 5, 6, 7)],
                                  tetris_rate=round(sum(r['bands'][k][2] for r in rows)/max(1, sum(r['bands'][k][1] for r in rows)), 3))
                          for k in rows[0]['bands']}, config=config)
    print(json.dumps(summary, indent=1))
    if args.output:
        args.output.write_text(json.dumps(dict(summary=summary, rows=rows), indent=1))


if __name__ == '__main__':
    main()
