"""HydrantDude / TetrisGYM crash-timing cycle model for the original NTSC ROM.

Pure-Python port of the research re-implementation (outputs/NES-CRASH-MAP-RESEARCH.md,
appendix and "model" block of the .json; scratch model.js). Usable at any 8-bit level
and any raw line counter, so it also covers the second lap after the 255 -> 0 wrap.

Sources
  [GYM]   TetrisGYM src/modes/crash.asm (kirjavascript/TetrisGYM, commit 40fc3ddf,
          2026-06-02), the in-game implementation of Hydrant's "Crash Theory" sheet.
  [SHEET] HydrantDude "Crash Theory" Google Sheet, tabs "Crash Dodge" (gid 1013692687)
          and "List of crashes by setup" (gid 2041037812), retrieved 2026-10-23.
  [ROM]   pinned disassembly work/nes-reference/main.asm,
          playState_updateLinesAndStatistics / incrementLines / addHoldDownPoints.

What the model computes: CPU cycles from a fixed point in the line-clear frame to each
of the 8 jump-table "switch" calls that follow the score loop. If the NMI lands inside a
switch's critical window, NMI and the switch share $00/$01 and the jump target is
corrupted (crash). Later totals reach the original sprite-staging loops ("confetti") or
just lag. A few terms (PRNG, frame parity, NMI return instruction, OAM-DMA alignment) are
not observable by the bot; like the research JS they are assumed uniform and independent
(48 equally likely combinations). In a deterministic emulator each event either crashes
or not; probabilities express uncertainty about those hidden terms.

Assumptions (all hold for this bot): NTSC, next box shown, no allegro (row 15 empty,
music not switching), Select not newly pressed, no expansion-port input, score capped at
999,999 (always true at the crash levels, both laps), A-type game.
"""
from collections import Counter

from nes_original.progression import after_clear

# --- deterministic terms -----------------------------------------------------------------
# [GYM] cycleCount initialised to $1C6F: cycles that always happen.
CONSTANT = 0x1C6F                      # 7279
# [GYM] @clearedLine: per completed row, 97 if the row number is >= 11 (low on the board,
# row 0 = top), 95 if <= 10 ("high").
ROW_COST_LOW = 97
ROW_COST_HIGH = 95
HIGH_ROW_MAX = 10
# [GYM] @linesNotCleared: next box rendering, 394 = $18A.
NEXT_BOX = 394
# [GYM] @nextOff: music/allegro routine with allegroIndex = 0 and wasAllegro = 0.
NO_ALLEGRO = 149
# [GYM] crashState bit 0/1/3, set by [ROM] incrementLines: low BCD digit of `lines` wraps
# (+79, includes the level-up threshold check that only runs then), tens digit wraps into
# the binary hundreds byte (+12), levelNumber incremented (+18).
TENS_CARRY = 79
HUNDREDS_CARRY = 12
LEVEL_UP = 18
# [GYM] @pushDown / [ROM] addHoldDownPoints: raw holdDownPoints >= 2 adds raw-1 points to a
# capped score; 99 cycles if the low BCD digit needs the +6 fix-up ('p'), 90 if not ('P').
PUSHDOWN_SMALL = 99                    # 'p'  (sheet "Pd <7", GYM raw 2..7)
PUSHDOWN_LARGE = 90                    # 'P'  (sheet "Pd 7+", GYM raw >= 8)
# [GYM] @single/@notsingle: the no-pushdown branch, singles vs multi-clears.
NO_PUSHDOWN_SINGLE = 53
NO_PUSHDOWN_MULTI = 42
# [ROM] clear-type stat BCD fix-up (lineClearStatsByType $D8-$DB): this clear makes that
# type's count end in 0. The ROM runs it before the pushdown branch (applies always);
# GYM only adds it on the no-pushdown branch (gym_clear_count=True reproduces GYM).
CLEAR_COUNT_CARRY = 11
# [GYM] sumTable ($E1 + 2*256 for 0 lines) and factorTable, times the post-clear
# levelNumber: the (level+1)-iteration score loop. An 8-bit level 255 runs 256 iterations
# ([ROM] inc generalCounter wraps to 0), so the linear term holds at every level.
SUM_BY_CLEARED = (0xE1+512,0x1C,0x38,0x54,0x80)      # 737, 28, 56, 84, 128
LEVEL_FACTOR = (0x53,0x88,0x7D,0x7D,0x7D)            # 83, 136, 125, 125, 125
# [GYM] Tetris with frameCounter & 7 == 0: +5. Unobserved -> probability 1/8.
TETRIS_FRAME_MOD8_ZERO = 5
# [GYM] @currentPieceCheck: currentPiece < 8 (T/J) +0, == 8 (horizontal Z) +8, > 8 +12.
# During a line clear the current piece is the hidden orientation 19 ([SHEET] note), so
# line clears always take +12; the class only matters for zero-line locks.
PIECE_TERM = {'TJ':0,'hZ':8,'other':12}
LINE_CLEAR_PIECE_TERM = 12

# --- hidden jitter ([GYM] @randomFactors) ------------------------------------------------
# oneThirdPRNG 0..2, frame parity 0/1, rng_seed bit 0/1, NMI return instruction 0/3,
# OAM-DMA sync 0/1. Assumed uniform and independent (research assumption).
JITTER = Counter(a+b+d+e+g for a in (0,1,2) for b in (0,1) for d in (0,1) for e in (0,3) for g in (0,1))
JITTER_COMBINATIONS = sum(JITTER.values())          # 48

# --- outcome windows ([GYM] @loop and after) ---------------------------------------------
# Offsets of switch calls 1..8 ([GYM] switchTable read in reverse; x=8 is switch 1).
SWITCH_OFFSETS = (0,60,162,222,323,383,502,562)
CRASH_WINDOWS = ((29739,29744),(29750,29762))
RED_WINDOW = (29763,29768)             # crash, except switch 2 (none) and switch 6 (graphics)
RED_NO_CRASH_SWITCH = 2                # NTSC
RED_GRAPHICS_SWITCH = 6                # "Summoning Satan" graphics corruption
# Totals after all 562 offset cycles:
CONFETTI_LIMITED = (30407,30746)       # exits on A/Select, or neutral at frameCounter 255
CONFETTI_INFINITE = (31329,31757)      # any held button exits
LEVEL_LAG_FROM = 30879
LINE_LAG_FROM = 31074

OUTCOMES = ('crash','graphics','confetti_limited','confetti_infinite','line_lag','level_lag','ok')
PIECE_CLASSES = ('TJ','hZ','other')


def pushdown_category(raw):
    """'none', 'p' or 'P' for a raw holdDownPoints value, as the ROM branches on it.

    [ROM] addHoldDownPoints: `cmp #$02 / bmi` skips pushdown for raw 0..1 (and raw >= $80,
    signed). Otherwise the capped score's low byte $99 is decremented and raw is added:
    $98+raw. The +6 BCD fix-up runs iff its low nibble is >= $A, which is 'p'. For raw
    2..17 this equals GYM's rule (2..7 'p', >= 8 'P'); raw 18..23 is 'p' again in the ROM
    (GYM would say 'P'). Assumes a capped score and raw < $68.
    """
    raw = int(raw)
    if raw<0:
        raise ValueError('raw pushdown must be >= 0')
    if raw<2 or raw>=0x80:
        return 'none'
    return 'p' if ((0x98+raw)&0x0F)>=0x0A else 'P'


def cycle_base(post_level,cleared,*,rows_high=0,tens_carry=False,hundreds_carry=False,level_up=False,
               pushdown='none',clear_count_carry=False,piece_class='other',next_box=True,
               tetris_frame_mod8_zero=False,gym_clear_count=False):
    """Deterministic cycle total before jitter and switch offsets.

    pushdown: 'none' | 'p' | 'P' (see pushdown_category) or a raw holdDownPoints int.
    rows_high: how many of the cleared rows are board rows 0..10.
    """
    if not 0<=post_level<=255 or not 0<=cleared<=4 or not 0<=rows_high<=cleared:
        raise ValueError('Expected 8-bit level, 0..4 cleared and 0..cleared high rows')
    if not isinstance(pushdown,str):
        pushdown = pushdown_category(pushdown)
    c = CONSTANT+ROW_COST_LOW*(cleared-rows_high)+ROW_COST_HIGH*rows_high
    if next_box:
        c += NEXT_BOX
    c += NO_ALLEGRO
    c += TENS_CARRY*bool(tens_carry)+HUNDREDS_CARRY*bool(hundreds_carry)+LEVEL_UP*bool(level_up)
    if pushdown=='p':
        c += PUSHDOWN_SMALL
    elif pushdown=='P':
        c += PUSHDOWN_LARGE
    elif pushdown=='none':
        c += NO_PUSHDOWN_SINGLE if cleared==1 else NO_PUSHDOWN_MULTI if cleared>1 else 0
    else:
        raise ValueError(f'Unknown pushdown category {pushdown!r}')
    if clear_count_carry and cleared and (not gym_clear_count or pushdown=='none'):
        c += CLEAR_COUNT_CARRY
    c += SUM_BY_CLEARED[cleared]+LEVEL_FACTOR[cleared]*post_level
    if cleared==4 and tetris_frame_mod8_zero:
        c += TETRIS_FRAME_MOD8_ZERO
    if cleared:
        c += LINE_CLEAR_PIECE_TERM
    else:
        if piece_class not in PIECE_TERM:
            raise ValueError(f'piece_class must be one of {PIECE_CLASSES}')
        c += PIECE_TERM[piece_class]
    return c


def classify(total,next_box=True):
    """Outcome for a cycle total that already includes the jitter ([GYM] @loop..@allegroClear)."""
    for i,offset in enumerate(SWITCH_OFFSETS):
        switch = i+1
        v = total+offset
        if any(lo<=v<=hi for lo,hi in CRASH_WINDOWS):
            return 'crash'
        if RED_WINDOW[0]<=v<=RED_WINDOW[1]:
            if switch==RED_NO_CRASH_SWITCH:
                continue
            return 'graphics' if switch==RED_GRAPHICS_SWITCH else 'crash'
    e = total+SWITCH_OFFSETS[-1]
    if not next_box:
        # GYM adds the next-box cycles only now and skips the limited-confetti check.
        e += NEXT_BOX
    elif CONFETTI_LIMITED[0]<=e<=CONFETTI_LIMITED[1]:
        return 'confetti_limited'
    if e>=LINE_LAG_FROM:
        return 'confetti_infinite' if CONFETTI_INFINITE[0]<=e<=CONFETTI_INFINITE[1] else 'line_lag'
    if e>=LEVEL_LAG_FROM:
        return 'level_lag'
    return 'ok'


def distribution(post_level,cleared,*,tetris_frame_mod8_zero=None,next_box=True,**terms):
    """Outcome probabilities over the hidden jitter for cycle_base(post_level, cleared, **terms).

    For a Tetris, tetris_frame_mod8_zero=None mixes the +5 term with weight 1/8.
    Returns {outcome: probability} (only non-zero outcomes) plus 'cycle_base'.
    """
    if cleared==4 and tetris_frame_mod8_zero is None:
        frames = ((False,7/8),(True,1/8))
    else:
        frames = ((bool(tetris_frame_mod8_zero),1.),)
    result = Counter()
    for f0,weight in frames:
        base = cycle_base(post_level,cleared,next_box=next_box,tetris_frame_mod8_zero=f0,**terms)
        for jitter,count in JITTER.items():
            result[classify(base+jitter,next_box)] += weight*count/JITTER_COMBINATIONS
    out = {k:result[k] for k in OUTCOMES if result[k]}
    out['cycle_base'] = cycle_base(post_level,cleared,next_box=next_box,tetris_frame_mod8_zero=False,**terms)
    return out


def counter_carries(pre_level,pre_lines,cleared):
    """Post level/lines and the tens/hundreds/level-up flags of a clear ([ROM] incrementLines)."""
    post_level,post_lines = after_clear(pre_level,pre_lines,cleared)
    tens = hundreds = False
    lines = pre_lines
    for _ in range(cleared):
        lines = (lines+1)%25600
        tens |= lines%10==0
        hundreds |= lines%100==0
    return dict(post_level=post_level,post_lines=post_lines,tens_carry=tens,hundreds_carry=hundreds,
                level_up=post_level!=pre_level)


def crash_outcomes(pre_level,pre_lines,cleared,*,rows_high,pushdown_raw,clear_count_carry,piece_class=None,
                   next_box=True,tetris_frame_mod8_zero=None):
    """Outcome probabilities for one lock, from things the bot observes or derives.

    pre_level, pre_lines: RAM levelNumber and raw line counter (binary hundreds*100 + BCD
        low byte, as in progression.after_clear) before the lock.
    cleared: lines this lock clears (0..4).
    rows_high: how many cleared rows are board rows 0..10 (row 0 = top).
    pushdown_raw: raw holdDownPoints at the lock.
    clear_count_carry: this clear makes its type's count since game start end in 0
        (10th, 20th ... single/double/triple/tetris). Must be False for cleared == 0.
    piece_class: 'TJ' | 'hZ' | 'other', required for zero-line locks, ignored for clears.
    tetris_frame_mod8_zero: frameCounter % 8 == 0 for a Tetris, or None if unknown.

    Returns probabilities 'fatal' (crash), 'graphics' (switch-6 red window corruption,
    counted separately as in CRASH_MAP), 'confetti_limited', 'confetti_infinite', 'lag'
    (= 'level_lag' + 'line_lag'), 'ok', plus the derived inputs and 'cycle_base'.
    """
    if cleared==0:
        if piece_class not in PIECE_TERM:
            raise ValueError(f'zero-line locks need piece_class in {PIECE_CLASSES}')
        if clear_count_carry:
            raise ValueError('a zero-line lock cannot complete a clear-type count')
    else:
        piece_class = 'other'
    flags = counter_carries(pre_level,pre_lines,cleared)
    pushdown = pushdown_category(pushdown_raw)
    dist = distribution(flags['post_level'],cleared,rows_high=rows_high,tens_carry=flags['tens_carry'],
        hundreds_carry=flags['hundreds_carry'],level_up=flags['level_up'],pushdown=pushdown,
        clear_count_carry=bool(clear_count_carry),piece_class=piece_class,next_box=next_box,
        tetris_frame_mod8_zero=tetris_frame_mod8_zero)
    get = lambda k: dist.get(k,0.)
    return dict(fatal=get('crash'),graphics=get('graphics'),confetti_limited=get('confetti_limited'),
        confetti_infinite=get('confetti_infinite'),lag=get('level_lag')+get('line_lag'),
        level_lag=get('level_lag'),line_lag=get('line_lag'),ok=get('ok'),
        cycle_base=dist['cycle_base'],pushdown_category=pushdown,pre_level=pre_level,pre_lines=pre_lines,
        cleared=cleared,**flags)


def worst_fatal(pre_level,pre_lines,cleared,*,rows_high=None,pushdown_raw=0,clear_count_carry=None,
                piece_class=None,**kwargs):
    """Max fatal probability over the inputs left as None (unknown)."""
    highs = range(cleared+1) if rows_high is None else (rows_high,)
    carries = ((False,True) if cleared else (False,)) if clear_count_carry is None else (clear_count_carry,)
    pieces = (PIECE_CLASSES if piece_class is None else (piece_class,)) if cleared==0 else ('other',)
    return max(crash_outcomes(pre_level,pre_lines,cleared,rows_high=h,pushdown_raw=pushdown_raw,
               clear_count_carry=c,piece_class=p,**kwargs)['fatal']
               for h in highs for c in carries for p in pieces)


# --- whole-game tables -------------------------------------------------------------------

def walk_progress(start_level=18,start_lines=0,laps=2):
    """[(lap, level, raw_lines)] for every line count, one line at a time, via after_clear.

    lap 0 is the first pass; lap increments when the 8-bit level wraps 255 -> 0. Stops
    before the state that would start lap `laps`.
    """
    states = []
    level,lines,lap = start_level,start_lines,0
    while True:
        states.append((lap,level,lines))
        new_level,new_lines = after_clear(level,lines,1)
        if new_lines<lines:
            raise ValueError('raw line counter wrapped')
        if new_level<level:
            lap += 1
            if lap>=laps:
                return states
        level,lines = new_level,new_lines


_VARIANTS = (('base',False,'low'),('clear_count_carry',True,'low'),('all_rows_high',False,'high'),
             ('carry_and_high',True,'high'))


def fatal_hazard_table(lap,start_level=18,start_lines=0,pushdown_raw=0,levels=range(0,256)):
    """CRASH_MAP-like table for one lap: {(pre_level, post_level, cleared): entry}.

    Each entry has the max fatal probability (over all pre-clear line counts of that lap
    giving this key) for the four CRASH_MAP variants (all rows low/high, with/without
    clear-count carry), 'any_rows_mix' (max over every high/low row count and carry),
    'by_rows_high' (max fatal per number of high rows, without/with carry), the counter
    carries that occur and the pre-clear line range. Only keys with a non-zero fatal probability are kept.
    """
    cache = {}
    table = {}
    for state_lap,level,lines in walk_progress(start_level,start_lines,lap+1):
        if state_lap!=lap or level not in levels:
            continue
        for cleared in range(1,5):
            flags = counter_carries(level,lines,cleared)
            key = (level,flags['post_level'],cleared)
            sig = (flags['post_level'],cleared,flags['tens_carry'],flags['hundreds_carry'],flags['level_up'])
            if sig not in cache:
                fatal = {}
                for carry in (False,True):
                    for high in range(cleared+1):
                        fatal[carry,high] = distribution(sig[0],cleared,rows_high=high,tens_carry=sig[2],
                            hundreds_carry=sig[3],level_up=sig[4],pushdown=pushdown_category(pushdown_raw),
                            clear_count_carry=carry).get('crash',0.)
                cache[sig] = fatal
            fatal = cache[sig]
            entry = table.setdefault(key,dict(pre_level=key[0],post_level=key[1],cleared=cleared,
                **{name:0. for name,_,_ in _VARIANTS},any_rows_mix=0.,
                by_rows_high={'no_carry':[0.]*(cleared+1),'carry':[0.]*(cleared+1)},carries=set(),pre_lines=[]))
            for (carry,high),value in fatal.items():
                row = entry['by_rows_high']['carry' if carry else 'no_carry']
                row[high] = max(row[high],value)
            for name,carry,rows in _VARIANTS:
                entry[name] = max(entry[name],fatal[carry,0 if rows=='low' else cleared])
            entry['any_rows_mix'] = max(entry['any_rows_mix'],max(fatal.values()))
            entry['carries'].add(('tens' if sig[2] else '')+('+hundreds' if sig[3] else '')+('+level_up' if sig[4] else ''))
            entry['pre_lines'].append(lines)
    out = {}
    for key,entry in table.items():
        if entry['any_rows_mix']>0:
            entry['carries'] = sorted(entry['carries'])
            entry['pre_lines'] = [min(entry['pre_lines']),max(entry['pre_lines'])]
            out[key] = entry
    return out
