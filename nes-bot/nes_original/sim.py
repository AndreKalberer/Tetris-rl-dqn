"""Fast whole-game NES simulator and a tunable two-mode placement evaluator.

Simulation only. Active-piece control reuses nes_original.fast.frame, which is
differentially tested against nes.engine.control_frame, and the candidate
family matches nes.actions.enumerate_actions. Entry delay and randomizer timing
between pieces are approximate, so piece sequences differ from the original
game. Use this to compare and tune evaluators cheaply; confirm in the emulator.

Two weight vectors: "score" mode is used until the score reaches the original
cap (999999) or the optional switch level, then "survive" mode for good. Each vector weights FEATURES below.
"""
import numpy as np
from numba import njit

from nes.engine import SPAWN
from nes_original.fast import KINDS,ROTATE,SHAPES,collision,frame

MAXED = 999999
SPAWNS = np.asarray(SPAWN,dtype=np.int64)
BASE_POINTS = np.asarray((0,40,100,300,1200),dtype=np.int64)
BOARD_FEATURES = ('holes','hole_rows','covered','height_sum','max_height','bump','row_transitions',
    'column_transitions','well_height','tetris_ready','i_dependencies','danger','cumulative_wells')
PLACEMENT_FEATURES = ('landing_height','eroded','single','double','triple','tetris')
# Level-29 terms in the style of StackRabbit's evaluator (appended last so older 19-weight
# files keep their indices; they are padded with 0). Right well = column 9.
REACH_FEATURES = ('inacc_left','inacc_right','covered_well','well_burns','avg_excess','col8_excess')
FEATURES = BOARD_FEATURES+PLACEMENT_FEATURES+REACH_FEATURES
N_REACH0 = len(BOARD_FEATURES)+len(PLACEMENT_FEATURES)
# Highest column a piece can pass over at 30 Hz, level 29+ (fast simulator, measured):
# a flat T/J/L/O tapped to the left wall, and a vertical I/O tapped to column 9.
LEFT_LIMIT = np.asarray((7,7,7,8,10,12,14,20,20,20),dtype=np.int64)
RIGHT_LIMIT = np.asarray((20,20,20,20,13,11,9,7,7,7),dtype=np.int64)
SCARE_HEIGHT = 4.
# Default frames the controller waits after spawn before its first input (v8c: 3). Every
# enumeration takes the wait as an explicit `reaction` argument; this is only the default.
REACTION = 3
SAFE_COL8 = 5
N_BOARD = len(BOARD_FEATURES)
N_FEATURES = len(FEATURES)
DEAD = -1.e12
BANNED = -1.e9    # crash-alerted clear (bitmask `ban` by cleared lines), simulation only
SOFT_DEAD = -1.e6  # a top-out for one unknown piece, inside an expectation
MAX_CANDIDATES = 48


def pad_weights(w):
    """Accept weight vectors written before the reach terms existed (zero weight for them)."""
    w = np.asarray(w,dtype=np.float64)
    if w.shape==(N_REACH0,):
        w = np.concatenate([w,np.zeros(N_FEATURES-N_REACH0)])
    return w


def weights(**named):
    """Build a weight vector from feature names; unnamed features are zero."""
    unknown = set(named)-set(FEATURES)
    if unknown:
        raise ValueError(f'Unknown features: {sorted(unknown)}')
    return np.asarray([float(named.get(name,0.)) for name in FEATURES])


# Retained original-game selectors expressed in this feature set (well=-1).
RETAINED_PRIOR = weights(height_sum=-.51,holes=-4.5,bump=-.18,single=.76,double=1.52,triple=2.28,tetris=3.04)
DELLACHERIE = weights(landing_height=-1,eroded=1,row_transitions=-1,column_transitions=-1,holes=-4,
    cumulative_wells=-1)

# Hand-set starting points for tuning; not tuned results.
DEFAULT_SCORE = weights(holes=-6,hole_rows=-3,covered=-.3,height_sum=-.08,max_height=-.1,bump=-.35,
    row_transitions=-.4,column_transitions=-.6,well_height=-3,tetris_ready=1.2,i_dependencies=-2.5,
    danger=-.6,landing_height=-.15,eroded=.1,single=-3,double=-2.5,triple=-1.5,tetris=8)
# Score-mode tuning start: Dellacherie survival terms plus a right-well tetris setup.
SCORE_START = weights(landing_height=-1,eroded=.5,row_transitions=-.6,column_transitions=-1,holes=-4,
    cumulative_wells=-1,well_height=-2,tetris_ready=2,i_dependencies=-2,danger=-.3,
    single=-2,double=-1.5,triple=-1,tetris=6)
DEFAULT_SURVIVE = weights(holes=-6,hole_rows=-3,covered=-.3,height_sum=-.15,max_height=-.3,bump=-.35,
    row_transitions=-.5,column_transitions=-.7,well_height=-.5,tetris_ready=.3,i_dependencies=-2.5,
    danger=-1.,landing_height=-.3,eroded=.3,single=0,double=.3,triple=.6,tetris=2)


@njit(cache=True)
def lock(board,x,y,orientation):
    """Lock a final pose. Returns (board, cleared, dead), matching fast.lock_score."""
    if collision(board,x,y,orientation):
        return board.copy(),0,True
    result = board.copy()
    for i in range(4):
        yy = y+SHAPES[orientation,i,1]
        if yy>=0:
            result[yy,x+SHAPES[orientation,i,0]] = True
    cleared = 0
    start = max(0,y-2)
    for row in range(start,start+4):
        if row>=20:
            continue
        full = True
        for col in range(10):
            if not result[row,col]:
                full = False
                break
        if full:
            for r in range(row,0,-1):
                for col in range(10):
                    result[r,col] = result[r-1,col]
            for col in range(10):
                result[0,col] = False
            cleared += 1
    return result,cleared,False


@njit(cache=True)
def board_features(board,well,out):
    """Fill out[0:N_BOARD]. well<0 disables the well-specific terms."""
    heights = np.zeros(10,np.int64)
    holes = covered = 0
    hole_row = np.zeros(20,np.bool_)
    for col in range(10):
        filled_above = 0
        for row in range(20):
            if board[row,col]:
                if heights[col]==0:
                    heights[col] = 20-row
                filled_above += 1
            elif heights[col]:
                holes += 1
                covered += filled_above
                hole_row[row] = True
    height_sum = max_height = bump = 0
    for col in range(10):
        max_height = max(max_height,heights[col])
        if col!=well:
            height_sum += heights[col]
        if col<9 and col!=well and col+1!=well:
            bump += abs(heights[col]-heights[col+1])
    row_transitions = 0
    for row in range(20):
        previous = True
        for col in range(10):
            if board[row,col]!=previous:
                row_transitions += 1
            previous = board[row,col]
        if not previous:
            row_transitions += 1
    column_transitions = 0
    for col in range(10):
        previous = False
        for row in range(20):
            if board[row,col]!=previous:
                column_transitions += 1
            previous = board[row,col]
        if not previous:
            column_transitions += 1
    well_height = ready = 0
    if well>=0:
        well_height = heights[well]
        row = 19-heights[well]
        while row>=0 and ready<4:
            full = True
            for col in range(10):
                if col!=well and not board[row,col]:
                    full = False
                    break
            if not full or board[row,well]:
                break
            ready += 1
            row -= 1
    dependencies = 0
    for col in range(10):
        if col==well:
            continue
        left = 20 if col==0 or col-1==well else heights[col-1]
        right = 20 if col==9 or col+1==well else heights[col+1]
        if min(left,right)-heights[col]>=3:
            dependencies += 1
    danger = max(0,max_height-8)
    cumulative = 0
    for col in range(10):
        if col==well:
            continue
        depth = 0
        for row in range(20):
            left = col==0 or board[row,col-1]
            right = col==9 or board[row,col+1]
            if not board[row,col] and left and right:
                depth += 1
                cumulative += depth
            else:
                depth = 0
    out[0] = holes
    out[1] = hole_row.sum()
    out[2] = covered
    out[3] = height_sum
    out[4] = max_height
    out[5] = bump
    out[6] = row_transitions
    out[7] = column_transitions
    out[8] = well_height
    out[9] = ready
    out[10] = dependencies
    out[11] = danger*danger
    out[12] = cumulative
    # Left access matters when column 0 is not built out above the others.
    inacc_left = 0.
    if heights[0]<=LEFT_LIMIT[0] or heights[0]<heights[8]:
        worst = 0
        for col in range(7):
            worst = max(worst,heights[col]-LEFT_LIMIT[col])
        if worst>0:
            inacc_left = 1.+.2*worst*worst
    inacc_right = 0.
    if heights[9]<=RIGHT_LIMIT[9] or heights[9]<heights[8]:
        worst = 0
        for col in range(4,10):
            worst = max(worst,heights[col]-RIGHT_LIMIT[col])
        if worst>0:
            inacc_right = 1.+.2*worst*worst
    covered_well = 0.
    well_burns = 0
    for row in range(20):
        if board[row,9]:
            if covered_well==0.:
                ratio = (20.-row)/max(3.,SCARE_HEIGHT)
                covered_well = ratio*ratio*ratio
            well_burns += 1
    average = 0.
    for col in range(9):
        average += heights[col]/9.
    excess = max(0.,average-SCARE_HEIGHT)
    col8 = max(0,heights[8]-SAFE_COL8)
    out[N_REACH0] = inacc_left
    out[N_REACH0+1] = inacc_right
    out[N_REACH0+2] = covered_well
    out[N_REACH0+3] = well_burns
    out[N_REACH0+4] = excess*excess
    out[N_REACH0+5] = col8*col8


@njit(cache=True)
def placement_features(before,x,y,orientation,cleared,out):
    """Fill out[N_BOARD:]: landing height, eroded cells and the clear type."""
    low = 99
    high = -99
    for i in range(4):
        yy = y+SHAPES[orientation,i,1]
        low = min(low,yy)
        high = max(high,yy)
    eroded = 0
    if cleared:
        pieces_in_full = 0
        for i in range(4):
            yy = y+SHAPES[orientation,i,1]
            if yy<0:
                continue
            xx = x+SHAPES[orientation,i,0]
            full = True
            for col in range(10):
                if col==xx:
                    continue
                if not before[yy,col]:
                    other = False
                    for j in range(4):
                        if y+SHAPES[orientation,j,1]==yy and x+SHAPES[orientation,j,0]==col:
                            other = True
                    if not other:
                        full = False
                        break
            if full:
                pieces_in_full += 1
        eroded = cleared*pieces_in_full
    out[N_BOARD] = 20-(low+high)/2
    out[N_BOARD+1] = eroded
    for k in range(4):
        out[N_BOARD+2+k] = 1. if cleared==k+1 else 0.


@njit(cache=True)
def candidates(board,initial,level,tap_period,states,reaction):
    """Fill states with the final pose of every timed candidate; return count.

    Same button schedule as nes.actions.enumerate_actions with `reaction` idle frames first,
    navigation48 and rotation period4. Duplicated final poses are kept.
    """
    n = 0
    kind = KINDS[initial[2]]
    for target in range(19):
        if KINDS[target]!=kind:
            continue
        left = 10
        right = -10
        for i in range(4):
            left = min(left,SHAPES[target,i,0])
            right = max(right,SHAPES[target,i,0])
        for target_x in range(-left,10-right):
            state = initial
            failed = False
            frames = 0
            for f in range(reaction+48+160):
                elapsed = f-reaction
                if elapsed<0:
                    buttons = 0
                elif elapsed>=48:
                    buttons = 4
                else:
                    horizontal = 2 if state[0]>target_x else 1 if state[0]<target_x else 0
                    moving = horizontal!=0
                    if tap_period and elapsed%tap_period:
                        horizontal = 0
                    if state[2]!=target:
                        rotation = ROTATE[state[2],target]
                        pulse = rotation if elapsed%4==0 and not state[7]&rotation else 0
                        buttons = horizontal|pulse
                    else:
                        buttons = horizontal if moving else 4
                state,failed = frame(board,state,buttons,level)
                frames += 1
                if failed:
                    break
            if not failed:
                raise RuntimeError('Candidate exceeded its frame bound')
            for k in range(8):
                states[n,k] = state[k]
            states[n,8] = frames
            n += 1
    return n


@njit(cache=True)
def level_after(start_level,lines):
    first = min(start_level*10+10,max(100,start_level*10-50))
    return start_level if lines<first else start_level+1+(lines-first)//10


@njit(cache=True)
def spawn_state(orientation,das,board,level):
    """Spawn pose plus two neutral settling frames, as nes_original.lookahead does."""
    state = (5,0,orientation,0,das,0,0,0)
    for _ in range(2):
        state,failed = frame(board,state,0,level)
        if failed:
            return state,True
    return state,False


@njit(cache=True)
def piece_bit(orientation):
    """1 = T/J, 2 = horizontal Z, 4 = other (crash_model.piece_class)."""
    return 1 if orientation<8 else 2 if orientation==8 else 4


def push_rule(level):
    """(min predicted pushdown, piece-class mask) for zero-line locks at this level, or (0, 0)."""
    from nes_original.crash_model import MIN_PREDICTED_PUSHDOWN,NEEDS_PUSHDOWN
    classes = NEEDS_PUSHDOWN.get(int(level))
    if not classes:
        return 0,0
    return MIN_PREDICTED_PUSHDOWN,sum({'TJ':1,'hZ':2,'other':4}[c] for c in classes)


@njit(cache=True)
def best_child(board,orientation,das,level,tap_period,reaction,well,w,scratch,states,ban=0,push_min=0,push_mask=0):
    """Best placement-plus-board value of one known piece; DEAD if none survive."""
    state,failed = spawn_state(orientation,das,board,level)
    if failed:
        return DEAD
    n = candidates(board,state,level,tap_period,states,reaction)
    best = DEAD
    for i in range(n):
        x,y,o = states[i,0],states[i,1],states[i,2]
        result,cleared,dead = lock(board,x,y,o)
        if dead:
            continue
        board_features(result,well,scratch)
        placement_features(board,x,y,o,cleared,scratch)
        value = 0.
        for k in range(N_FEATURES):
            value += w[k]*scratch[k]
        if ban>>cleared&1:
            value += BANNED
        if push_min and not cleared and states[i,6]<push_min and push_mask&piece_bit(o):
            value += BANNED   # zero-line lock that would need pushdown it cannot get
        best = max(best,value)
    return best


@njit(cache=True)
def expected_child(board,orientation,das,level,start_level,lines,tap_period,reaction,well,w,keep,states,grand_states,ban=0,
                   push_min=0,push_mask=0):
    """Known piece, then the average over all 7 unknown following pieces.

    Only the best `keep` distinct placements of the known piece are expanded.
    Uniform piece weights approximate the original randomizer.
    """
    state,failed = spawn_state(orientation,das,board,level)
    if failed:
        return DEAD
    n = candidates(board,state,level,tap_period,states,reaction)
    scratch = np.zeros(N_FEATURES)
    values = np.full(n,DEAD)
    placement = np.zeros(n)
    for i in range(n):
        x,y,o = states[i,0],states[i,1],states[i,2]
        result,cleared,dead = lock(board,x,y,o)
        if dead:
            continue
        board_features(result,well,scratch)
        placement_features(board,x,y,o,cleared,scratch)
        value = 0.
        part = 0.
        for k in range(N_FEATURES):
            value += w[k]*scratch[k]
            if k>=N_BOARD:
                part += w[k]*scratch[k]
        if ban>>cleared&1:
            value += BANNED
            part += BANNED
        if push_min and not cleared and states[i,6]<push_min and push_mask&piece_bit(o):
            value += BANNED
            part += BANNED
        values[i] = value
        placement[i] = part
    order = np.argsort(-values)
    best = DEAD
    expanded = 0
    seen = np.zeros((max(1,keep),20,10),np.bool_)
    for rank in range(n):
        i = order[rank]
        if values[i]<=DEAD or expanded>=keep:
            break
        x,y,o = states[i,0],states[i,1],states[i,2]
        result,cleared,dead = lock(board,x,y,o)
        duplicate = False
        for j in range(expanded):
            if np.array_equal(seen[j],result):
                duplicate = True
                break
        if duplicate:
            continue
        seen[expanded] = result
        expanded += 1
        child_level = level_after(start_level,lines+cleared)
        total = 0.
        for piece in range(7):
            v = best_child(result,SPAWNS[piece],states[i,4],child_level,tap_period,reaction,well,w,scratch,grand_states,ban,
                           push_min,push_mask)
            total += v if v>DEAD else SOFT_DEAD
        best = max(best,placement[i]+total/7)
    return best


@njit(cache=True)
def choose(board,state,next_orientation,level,start_level,lines,tap_period,reaction,well,w,beam,states,child_states,
           expect=0,ban=0,child_ban=False):
    """Return (candidate index, n candidates). beam=0 is one-ply; expect>0 adds the
    unknown piece after the visible next one, expanding `expect` next placements."""
    n = candidates(board,state,level,tap_period,states,reaction)
    scratch = np.zeros(N_FEATURES)
    values = np.full(n,DEAD)
    placement = np.zeros(n)
    for i in range(n):
        x,y,o = states[i,0],states[i,1],states[i,2]
        result,cleared,dead = lock(board,x,y,o)
        if dead:
            continue
        board_features(result,well,scratch)
        placement_features(board,x,y,o,cleared,scratch)
        value = 0.
        part = 0.
        for k in range(N_FEATURES):
            value += w[k]*scratch[k]
            if k>=N_BOARD:
                part += w[k]*scratch[k]
        if ban>>cleared&1:
            value += BANNED   # ranked below every allowed survivor, like crash_alerts tiers
        values[i] = value
        placement[i] = part
    if beam<=0 or n==0:
        return int(np.argmax(values)),n
    order = np.argsort(-values)
    best = DEAD
    chosen = int(order[0])
    expanded = 0
    seen = np.zeros((beam,20,10),np.bool_)
    for rank in range(n):
        i = order[rank]
        if values[i]<=DEAD or expanded>=beam:
            break
        x,y,o = states[i,0],states[i,1],states[i,2]
        result,cleared,dead = lock(board,x,y,o)
        duplicate = False
        for j in range(expanded):
            same = True
            for r in range(20):
                for c in range(10):
                    if seen[j,r,c]!=result[r,c]:
                        same = False
                        break
                if not same:
                    break
            if same:
                duplicate = True
                break
        if duplicate:
            continue
        seen[expanded] = result
        expanded += 1
        child_level = level_after(start_level,lines+cleared)
        if expect>0:
            child = expected_child(result,next_orientation,states[i,4],child_level,start_level,lines+cleared,
                tap_period,reaction,well,w,expect,child_states,np.zeros((MAX_CANDIDATES,9),np.int64),ban if child_ban else 0)
        else:
            child = best_child(result,next_orientation,states[i,4],child_level,tap_period,reaction,well,w,scratch,child_states,
                ban if child_ban else 0)
        total = placement[i]+child if child>DEAD else DEAD+values[i]
        if values[i]<BANNED/2:
            total += BANNED
        if total>best:
            best = total
            chosen = int(i)
    return chosen,n


# NES randomizer odds P(next | previous) out of 64, rows/cols in T,J,Z,O,S,L,I order (StackRabbit's
# piece_rng.cpp table; this simulator's _draw matches it within sampling noise).
TRANSITION = np.asarray(((2,10,12,10,10,10,10),(12,2,10,10,10,10,10),(10,12,2,10,10,10,10),(10,10,10,4,10,10,10),
    (10,10,10,10,4,10,10),(12,10,10,10,10,2,10),(10,10,10,10,12,10,2)),dtype=np.float64)/64.
PIECE_TYPE = np.asarray((0,0,0,0,1,1,1,1,2,2,3,4,4,5,5,5,5,6,6),dtype=np.int64)


@njit(cache=True)
def _scored_lock(board,x,y,o,well,w,scratch):
    """(result, dead, full value, placement part) of one final pose."""
    result,cleared,dead = lock(board,x,y,o)
    if dead:
        return result,True,DEAD,0.
    board_features(result,well,scratch)
    placement_features(board,x,y,o,cleared,scratch)
    value = 0.
    part = 0.
    for k in range(N_FEATURES):
        value += w[k]*scratch[k]
        if k>=N_BOARD:
            part += w[k]*scratch[k]
    return result,False,value,part


@njit(cache=True)
def _greedy(board,orientation,das,level,tap_period,reaction,well,w,states,scratch):
    """Best one-ply placement: (board, das, dead, full value, placement part)."""
    state,failed = spawn_state(orientation,das,board,level)
    if failed:
        return board,das,True,DEAD,0.
    n = candidates(board,state,level,tap_period,states,reaction)
    best = DEAD
    chosen = -1
    best_part = 0.
    for i in range(n):
        result,dead,value,part = _scored_lock(board,states[i,0],states[i,1],states[i,2],well,w,scratch)
        if not dead and value>best:
            best = value
            chosen = i
            best_part = part
    if chosen<0:
        return board,das,True,DEAD,0.
    result,cleared,dead = lock(board,states[chosen,0],states[chosen,1],states[chosen,2])
    return result,states[chosen,4],False,best,best_part


@njit(cache=True)
def playout_sequences(last_type,length):
    """Every piece sequence of `length` (as spawn orientations) and its NES probability."""
    m = 7**length
    seqs = np.zeros((m,length),np.int64)
    prob = np.zeros(m)
    for s in range(m):
        code = s
        p = 1.
        previous = last_type
        for j in range(length):
            t = code%7
            code //= 7
            seqs[s,j] = SPAWNS[t]
            p *= TRANSITION[previous,t]
            previous = t
        prob[s] = p
    return seqs,prob


@njit(cache=True)
def playout_scores(roots,root_part,root_das,next_orientation,level,tap_period,reaction,well,w,keep,cap,length,death):
    """StackRabbit-style playouts over current-piece result boards.

    roots[i] is the board after the current piece's placement i (root_part[i] its placement
    value, root_das[i] the DAS after it; root_part[i]<=DEAD/2 skips it). Every (i, next placement)
    pair is ranked by root_part+full value; the best `keep` pairs (at most `cap` per root) are
    played out over all 7**length following-piece sequences, weighted by NES odds from the next
    piece and placed greedily with w. A playout scores its placement rewards plus the final
    full value, or `death` plus the rewards so far if it tops out. Returns, per root, the best
    pair score (two placement rewards + expected playout score), or DEAD if not expanded.
    """
    k_roots = roots.shape[0]
    out = np.full(k_roots,DEAD)
    states = np.zeros((MAX_CANDIDATES,9),np.int64)
    deep = np.zeros((MAX_CANDIDATES,9),np.int64)
    scratch = np.zeros(N_FEATURES)
    size = k_roots*MAX_CANDIDATES
    pair_root = np.zeros(size,np.int64)
    pair_value = np.full(size,DEAD)
    pair_reward = np.zeros(size)
    pair_pose = np.zeros((size,4),np.int64)
    count = 0
    for i in range(k_roots):
        if root_part[i]<=DEAD/2:
            continue
        state,failed = spawn_state(next_orientation,root_das[i],roots[i],level)
        if failed:
            continue
        m = candidates(roots[i],state,level,tap_period,states,reaction)
        for j in range(m):
            result,dead,value,part = _scored_lock(roots[i],states[j,0],states[j,1],states[j,2],well,w,scratch)
            if dead:
                continue
            pair_root[count] = i
            pair_value[count] = root_part[i]+value
            pair_reward[count] = root_part[i]+part
            pair_pose[count,0] = states[j,0]
            pair_pose[count,1] = states[j,1]
            pair_pose[count,2] = states[j,2]
            pair_pose[count,3] = states[j,4]
            count += 1
    if count==0:
        return out
    order = np.argsort(-pair_value[:count])
    seqs,prob = playout_sequences(PIECE_TYPE[next_orientation],length)
    used = np.zeros(k_roots,np.int64)
    expanded = 0
    for r in range(count):
        if expanded>=keep:
            break
        k = order[r]
        i = pair_root[k]
        if used[i]>=cap:
            continue
        used[i] += 1
        expanded += 1
        start,cleared,dead = lock(roots[i],pair_pose[k,0],pair_pose[k,1],pair_pose[k,2])
        total = 0.
        for s in range(seqs.shape[0]):
            board = start
            das = pair_pose[k,3]
            reward = 0.
            value = death
            for j in range(length):
                board,das,dead,v,part = _greedy(board,seqs[s,j],das,level,tap_period,reaction,well,w,deep,scratch)
                if dead:
                    value = death+reward
                    break
                if j==length-1:
                    value = reward+v
                else:
                    reward += part
            total += prob[s]*value
        out[i] = max(out[i],pair_reward[k]+total)
    return out


@njit(cache=True)
def playout_choose(board,state,next_orientation,level,tap_period,reaction,well,w,keep,cap,length,death,ban=0):
    """Return (candidate index, n candidates) using playout_scores over every placement."""
    states = np.zeros((MAX_CANDIDATES,9),np.int64)
    n = candidates(board,state,level,tap_period,states,reaction)
    if n==0:
        return 0,0
    scratch = np.zeros(N_FEATURES)
    roots = np.zeros((n,20,10),np.bool_)
    part = np.full(n,DEAD)
    das = np.zeros(n,np.int64)
    values = np.full(n,DEAD)
    for i in range(n):
        result,cleared,dead = lock(board,states[i,0],states[i,1],states[i,2])
        if dead:
            continue
        result,dead,value,p = _scored_lock(board,states[i,0],states[i,1],states[i,2],well,w,scratch)
        if ban>>cleared&1:
            value += BANNED
            p += BANNED
        roots[i] = result
        part[i] = p
        das[i] = states[i,4]
        values[i] = value
    scores = playout_scores(roots,part,das,next_orientation,level,tap_period,reaction,well,w,keep,cap,length,death)
    if scores.max()<=DEAD:
        return int(np.argmax(values)),n
    return int(np.argmax(scores)),n


@njit(cache=True)
def _lfsr(state):
    return (state>>1)|((((state>>1)^(state>>9))&1)<<15)


@njit(cache=True)
def _draw(state,count,last):
    count = (count+1)&255
    index = ((state>>8)+count)&7
    if index==7 or SPAWNS[index]==last:
        state = _lfsr(state)
        index = (((state>>8)&7)+last)%7
    return state,count,SPAWNS[index]


@njit(cache=True)
def play(w_score,w_survive,seed,start_level,max_lines,tap_period,well,beam,initial_score,max_placements,
         stop_at_maxout=False,well_survive=-1,switch_level=999,expect=0,ban=0,ban_from=0,child_ban=False,
         ban_w=np.zeros(0),alt_w=np.zeros(0),alt_below=0,alt_well=9,reaction=REACTION,
         playout_top=0,playout_cap=4,playout_length=2,playout_death=-1000.):
    """One simulated game. Returns an int64 stats vector (see GAME_STATS).

    playout_top>0 replaces the beam/expect search with playout_choose while the alt
    (low-stack tetris) or burn weights are in use, i.e. when alt_w is given and survive mode is on.
    """
    rng = 0x8988
    for _ in range(seed%32767):
        rng = _lfsr(rng)
    count = 0
    rng,count,current = _draw(rng,count,0)
    rng = _lfsr(rng)
    rng,count,upcoming = _draw(rng,count,current)
    board = np.zeros((20,10),np.bool_)
    states = np.zeros((MAX_CANDIDATES,9),np.int64)
    child_states = np.zeros((MAX_CANDIDATES,9),np.int64)
    lines = placements = 0
    score = initial_score
    level = start_level
    das = 0
    clears = np.zeros(5,np.int64)
    uncapped = initial_score
    dead = False
    maxout_lines = -1 if score<MAXED else 0
    while lines<max_lines and placements<max_placements:
        survive = score>=MAXED or level>=switch_level
        use_playout = playout_top>0 and survive and alt_w.shape[0]==N_FEATURES and alt_below>0
        w = w_survive if survive else w_score
        mode_well = well_survive if survive else well
        if ban and lines>=ban_from and ban_w.shape[0]==N_FEATURES:
            w = ban_w
        if survive and alt_w.shape[0]==N_FEATURES and alt_below>0:
            # Low stack: use the alternative (e.g. tetris-building) weights.
            top = 20
            for r in range(20):
                for c in range(10):
                    if board[r,c]:
                        top = r
                        break
                if top<20:
                    break
            if 20-top<=alt_below:
                w = alt_w
                mode_well = alt_well
        state,failed = spawn_state(current,das,board,level)
        if failed:
            dead = True
            break
        if use_playout:
            index,n = playout_choose(board,state,upcoming,level,tap_period,reaction,mode_well,w,playout_top,playout_cap,
                playout_length,playout_death)
            candidates(board,state,level,tap_period,states,reaction)
        else:
            index,n = choose(board,state,upcoming,level,start_level,lines,tap_period,reaction,mode_well,w,beam,states,
                child_states,expect,ban if lines>=ban_from else 0,child_ban)
        if n==0:
            dead = True
            break
        x,y,o = states[index,0],states[index,1],states[index,2]
        board,cleared,dead = lock(board,x,y,o)
        if dead:
            break
        das = states[index,4]
        placements += 1
        lines += cleared
        clears[cleared] += 1
        level = level_after(start_level,lines)
        uncapped += BASE_POINTS[cleared]*(level+1)
        if score<MAXED:
            score = min(MAXED,score+BASE_POINTS[cleared]*(level+1))
            if score>=MAXED:
                maxout_lines = lines
                if stop_at_maxout:
                    break
        # Approximate entry delay: lock/render, plus the line-clear animation.
        for _ in range(states[index,8]+2+12+(20 if cleared else 0)):
            rng = _lfsr(rng)
        current = upcoming
        rng,count,upcoming = _draw(rng,count,current)
    out = np.zeros(11,np.int64)
    out[0] = lines
    out[1] = score
    out[2] = placements
    out[3] = int(dead)
    out[4] = level
    out[5] = maxout_lines
    for k in range(4):
        out[6+k] = clears[k+1]
    out[10] = uncapped
    return out


GAME_STATS = ('lines','score','placements','dead','level','maxout_lines','singles','doubles','triples','tetrises',
    'uncapped_score')


def play_game(w_score=DEFAULT_SCORE,w_survive=DEFAULT_SURVIVE,*,seed=0,start_level=18,max_lines=400,
              tap_period=2,well=9,beam=0,initial_score=0,max_placements=100000,stop_at_maxout=False,
              well_survive=-1,switch_level=999,expect=0,ban=0,ban_from=0,child_ban=False,ban_weights=None,
              alt_weights=None,alt_below=0,alt_well=9,reaction=REACTION,playout_top=0,playout_cap=4,playout_length=2,
              playout_death=-1000.):
    out = play(np.asarray(w_score,dtype=np.float64),np.asarray(w_survive,dtype=np.float64),int(seed),
        int(start_level),int(max_lines),int(tap_period),int(well),int(beam),int(initial_score),int(max_placements),
        bool(stop_at_maxout),int(well_survive),int(switch_level),int(expect),int(ban),int(ban_from),bool(child_ban),
        np.zeros(0) if ban_weights is None else np.asarray(ban_weights,dtype=np.float64),
        np.zeros(0) if alt_weights is None else np.asarray(alt_weights,dtype=np.float64),int(alt_below),int(alt_well),int(reaction),
        int(playout_top),int(playout_cap),int(playout_length),float(playout_death))
    return dict(zip(GAME_STATS,(int(v) for v in out)))


def stack_height(board):
    """Rows from the floor to the highest filled cell."""
    rows = np.flatnonzero(np.asarray(board).any(axis=1))
    return 0 if rows.size==0 else 20-int(rows[0])


def action_features(observation,plan,well=9):
    """Feature rows for the original-game proposal family (same evaluator as play)."""
    actions = plan.actions
    before = np.asarray(observation.board)!=0
    rows = np.zeros((len(actions.boards),N_FEATURES))
    for i,(control,after,cleared) in enumerate(zip(actions.controls,actions.boards,actions.lines)):
        board_features(np.asarray(after)!=0,well,rows[i])
        placement_features(before,control.x,control.y,control.orientation,int(cleared),rows[i])
    return rows


class TunedSelector:
    """Original-game selector: two-mode evaluator, optional visible-next beam.

    Reads only the visible board, current proposals, score/level counters and
    the single visible next piece. Exposes score_actions for crash_alerts.
    """
    def __init__(self,w_score=DEFAULT_SCORE,w_survive=DEFAULT_SURVIVE,*,well=9,well_survive=-1,beam=0,tap_period=2,
                 switch_level=999,expect=0,zones=(),reaction=REACTION,alt=None):
        self.w_score = pad_weights(w_score)
        self.w_survive = pad_weights(w_survive)
        if self.w_score.shape!=(N_FEATURES,) or self.w_survive.shape!=(N_FEATURES,):
            raise ValueError(f'Expected {N_FEATURES} weights per mode')
        self.well,self.well_survive,self.beam,self.tap_period = well,well_survive,beam,tap_period
        self.switch_level = switch_level
        self.expect = expect
        # (first level, last level, weights, well[, child ban mask, expect]): replaces the
        # mode weights there; the ban (e.g. 2 = singles) also applies to lookahead children.
        self.zones = tuple(self._zone(*z) for z in zones)
        # Frames the controller waits after spawn before its first input (lookahead children
        # are enumerated with the same schedule).
        self.reaction = int(reaction)
        # Low-stack tetris mode outside the zones: (weights, max stack rows, well column,
        # ((first level, last level), ...)[, burn weights used above that stack height]) on the
        # raw level byte, so it repeats every lap.
        self.alt = None
        if alt is not None:
            alt_w,below,alt_well,ranges = alt[:4]
            burn = None if len(alt)<5 or alt[4] is None else pad_weights(alt[4])
            self.alt = (pad_weights(alt_w),int(below),int(alt_well),tuple((int(a),int(b)) for a,b in ranges),burn)
        # Clear sizes (bit mask by lines) banned while the low-stack tetris mode is active.
        self.alt_ban = 0
        self.alt_expect = None
        self.burn_expect = None
        # StackRabbit-style playouts in the alt/burn modes (0 = off): pairs played out, max pairs
        # per current placement, following pieces per playout, value of a playout top-out.
        self.playout_top = 0
        self.playout_cap = 4
        self.playout_length = 2
        self.playout_death = -1000.
        self.grand_states = np.zeros((MAX_CANDIDATES,9),np.int64)
        self.switched = False
        self.states = np.zeros((MAX_CANDIDATES,9),np.int64)
        self.scratch = np.zeros(N_FEATURES)
        self.calls = 0
        self.modes = dict(score=0,survive=0,zone=0,alt=0)

    def _zone(self,lo,hi,w,well,ban=0,expect=None):
        return (int(lo),int(hi),pad_weights(w),int(well),int(ban),
            self.expect if expect is None else int(expect))

    def score_actions(self,observation,plan):
        self.calls += 1
        # Latch: once capped or at the switch level, stay in survive mode
        # (the raw level byte wraps after255).
        self.switched = self.switched or observation.score>=MAXED or observation.level>=self.switch_level
        survive = self.switched
        self.modes['survive' if survive else 'score'] += 1
        w = self.w_survive if survive else self.w_score
        well = self.well_survive if survive else self.well
        ban,expect = 0,self.expect
        push_min,push_mask = push_rule(observation.level)
        in_zone = False
        root_ban = 0
        use_playout = False
        for lo,hi,zone_w,zone_well,zone_ban,zone_expect in self.zones:
            if lo<=observation.level<=hi:
                w,well,ban,expect = zone_w,zone_well,zone_ban,zone_expect
                self.modes['zone'] += 1
                in_zone = True
                break
        if self.alt is not None and not in_zone:
            alt_w,below,alt_well,ranges,burn = self.alt
            if any(lo<=observation.level<=hi for lo,hi in ranges):
                if stack_height(observation.board)<=below:
                    w,well = alt_w,alt_well
                    ban = root_ban = self.alt_ban
                    if self.alt_expect is not None:
                        expect = self.alt_expect
                    use_playout = self.playout_top>0
                    self.modes['alt'] += 1
                elif burn is not None:
                    w,well = burn,-1
                    if self.burn_expect is not None:
                        expect = self.burn_expect
                    use_playout = self.playout_top>0
                    self.modes['burn'] = self.modes.get('burn',0)+1
        actions = plan.actions
        rows = action_features(observation,plan,well)
        values = rows@w
        if root_ban:
            banned = (root_ban>>np.asarray(actions.lines,dtype=np.int64))&1==1
            values[banned] += BANNED
        else:
            banned = None
        values[np.asarray(actions.terminal,dtype=bool)] = DEAD
        if self.beam<=0 and not use_playout:
            return values
        from nes.engine import SPAWN as spawn, TYPE
        from nes_original.progression import after_clear
        raw = getattr(observation,'next_orientation',None)
        if raw is None:
            next_orientation = spawn[observation.next_piece]
        elif 0<=raw<19:
            next_orientation = spawn[TYPE[raw]]
        else:
            return values  # Preview not readable this frame: one-ply values only.
        placement = rows[:,N_BOARD:]@w[N_BOARD:]
        if use_playout:
            return self._playout_values(observation,actions,values,placement,banned,next_orientation,well,w)
        result = values.copy()
        seen = set()
        for i in np.argsort(-values,kind='stable'):
            if len(seen)>=self.beam or values[i]<=DEAD:
                break
            board = np.asarray(actions.boards[i])!=0
            key = board.tobytes()
            if key in seen:
                continue
            seen.add(key)
            level,_ = after_clear(observation.level,observation.lines,int(actions.lines[i]))
            das = actions.controls[i].das
            if expect>0:
                # Gravity only: start18 progression from the visible line count.
                child = expected_child(board,next_orientation,das,min(level,29),18,
                    observation.lines+int(actions.lines[i]),self.tap_period,self.reaction,well,w,expect,
                    self.states,self.grand_states,ban,push_min,push_mask)
            else:
                child = best_child(board,next_orientation,das,min(level,29),self.tap_period,self.reaction,
                    well,w,self.scratch,self.states,ban,push_min,push_mask)
            # Expanded proposals rank by two-ply value, kept above unexpanded ones.
            result[i] = 1.e6+placement[i]+child if child>DEAD else DEAD/2
            if banned is not None and banned[i]:
                result[i] += BANNED
        return result

    def _playout_values(self,observation,actions,values,placement,banned,next_orientation,well,w):
        """Rank proposals by playout_scores; expanded proposals stay above unexpanded ones."""
        k = len(values)
        roots = np.zeros((k,20,10),np.bool_)
        part = np.full(k,DEAD)
        das = np.zeros(k,np.int64)
        for i in range(k):
            if values[i]<=DEAD:
                continue
            roots[i] = np.asarray(actions.boards[i])!=0
            part[i] = placement[i]+(BANNED if banned is not None and banned[i] else 0.)
            das[i] = actions.controls[i].das
        scores = playout_scores(roots,part,das,next_orientation,min(int(observation.level),29),self.tap_period,
            self.reaction,well,w,self.playout_top,self.playout_cap,self.playout_length,self.playout_death)
        result = values.copy()
        expanded = scores>DEAD
        result[expanded] = 1.e6+scores[expanded]
        return result

    def __call__(self,observation,plan):
        return int(np.argmax(self.score_actions(observation,plan)))

    def stats(self):
        return dict(calls=self.calls,modes=dict(self.modes),reaction=self.reaction,
            alt=None if self.alt is None else dict(below=self.alt[1],well=self.alt[2],levels=self.alt[3]),well=self.well,well_survive=self.well_survive,beam=self.beam,switch_level=self.switch_level,expect=self.expect,
            playout=dict(top=self.playout_top,cap=self.playout_cap,length=self.playout_length,death=self.playout_death),
            zones=[(lo,hi,zw,zb,ze) for lo,hi,_,zw,zb,ze in self.zones],
            information='Visible board, counters and single visible next piece only')


def _zone_from_file(selector,path,lo,hi,ban,expect):
    import json
    from pathlib import Path
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('features') not in (list(FEATURES),list(FEATURES[:N_REACH0])):
        raise ValueError(f'{path} was written for a different feature list')
    mode = data.get('mode','survive')
    return selector._zone(lo,hi,data[mode],data.get('well_survive' if mode=='survive' else 'well',-1),ban,expect)


def tuned_selector(weights='score_start',*,beam=3,expect=0,zone_weights=None,zone_levels=(156,158),
                   zone_ban=2,zone_expect=2,extra_zones=(),reaction=REACTION,alt_weights=None,alt_levels=None,
                   playout_top=0,playout_cap=4,playout_length=2,playout_death=-1000.):
    """Build a TunedSelector from a tune.py weights.json or a built-in name.

    score_start: untuned tetris mode at levels18-28, Dellacherie after.
    dellacherie: Dellacherie in both modes (adds visible-next lookahead).
    reaction: frames the controller waits after spawn before its first input.
    alt_weights: low-stack tetris weights.json (alt_weights, alt_below, alt_well[, alt_levels]),
    used outside the zones; alt_levels ((first, last), ...) overrides the file's levels.
    playout_top>0: StackRabbit-style playouts (playout_scores) instead of beam/expect in the alt
    and burn modes; playout_cap, playout_length and playout_death as in playout_scores.
    """
    selector = _build_selector(weights,beam,expect)
    selector.reaction = int(reaction)
    selector.playout_top,selector.playout_cap = int(playout_top),int(playout_cap)
    selector.playout_length,selector.playout_death = int(playout_length),float(playout_death)
    if alt_weights:
        import json
        from pathlib import Path
        data = json.loads(Path(alt_weights).read_text(encoding='utf-8'))
        if data.get('features')!=list(FEATURES):
            raise ValueError(f'{alt_weights} was written for a different feature list')
        levels = alt_levels or data.get('alt_levels') or ((29,153),(176,245))
        burn = data.get('burn_weights')
        selector.alt = (pad_weights(data['alt_weights']),int(data['alt_below']),int(data.get('alt_well',9)),
                        tuple((int(a),int(b)) for a,b in levels),None if burn is None else pad_weights(burn))
        selector.alt_ban = int(data.get('alt_ban',0))
        selector.alt_expect = data.get('alt_expect')
        selector.burn_expect = data.get('burn_expect')
        selector.alt_source = str(alt_weights)
    if zone_weights:
        import json
        from pathlib import Path
        data = json.loads(Path(zone_weights).read_text(encoding='utf-8'))
        if data.get('features') not in (list(FEATURES),list(FEATURES[:N_REACH0])):
            raise ValueError(f'{zone_weights} was written for a different feature list')
        mode = data.get('mode','survive')
        selector.zones = (selector._zone(zone_levels[0],zone_levels[1],data[mode],
            data.get('well_survive' if mode=='survive' else 'well',-1),zone_ban,zone_expect),)
        selector.zone_source = str(zone_weights)
    # extra_zones: (weights.json, first level, last level[, child ban mask[, expect]]).
    for zone in extra_zones:
        path,lo,hi = zone[:3]
        ban = int(zone[3]) if len(zone)>3 else 0
        zone_expect = int(zone[4]) if len(zone)>4 else None
        selector.zones = selector.zones+(_zone_from_file(selector,path,int(lo),int(hi),ban,zone_expect),)
    return warm_paths(warmup(selector))


def _build_selector(weights,beam,expect):
    if weights=='score_start':
        return TunedSelector(SCORE_START,DELLACHERIE,well=9,well_survive=-1,beam=beam,switch_level=29,expect=expect)
    if weights=='dellacherie':
        return TunedSelector(DELLACHERIE,DELLACHERIE,well=-1,well_survive=-1,beam=beam,expect=expect)
    import json
    from pathlib import Path
    data = json.loads(Path(weights).read_text(encoding='utf-8'))
    if data.get('features') not in (list(FEATURES),list(FEATURES[:N_REACH0])):
        raise ValueError(f'{weights} was written for a different feature list')
    return TunedSelector(data['score'],data['survive'],well=data['well'],well_survive=data.get('well_survive',-1),
        beam=beam,switch_level=data.get('switch_level',999),expect=expect)


def warmup(selector):
    """Compile every numba path the selector uses before gameplay starts."""
    board = np.zeros((20,10),np.bool_)
    out = np.zeros(N_FEATURES)
    pairs = [(w,well) for w in (selector.w_score,selector.w_survive)
             for well in (selector.well,selector.well_survive)]+[(w,well) for _,_,w,well,_,_ in selector.zones]
    if selector.alt is not None:
        pairs.append((selector.alt[0],selector.alt[2]))
        if selector.alt[4] is not None:
            pairs.append((selector.alt[4],-1))
    for w,well in pairs:
        board_features(board,well,out)
        placement_features(board,5,18,2,0,out)
        best_child(board,2,0,18,selector.tap_period,selector.reaction,well,w,out,selector.states)
        for expect in {selector.expect,*(z[5] for z in selector.zones)}:
            for ban in (0,2):
                if expect>0:
                    expected_child(board,2,0,18,18,0,selector.tap_period,selector.reaction,well,w,expect,
                        selector.states,selector.grand_states,ban)
                best_child(board,2,0,18,selector.tap_period,selector.reaction,well,w,out,selector.states,ban,3,6)
        if selector.playout_top>0:
            playout_scores(np.zeros((1,20,10),np.bool_),np.zeros(1),np.zeros(1,np.int64),2,29,selector.tap_period,
                selector.reaction,well,w,selector.playout_top,selector.playout_cap,selector.playout_length,
                selector.playout_death)
    return selector


class _WarmControl:
    __slots__ = ('x','y','orientation','das','soft_points')

    def __init__(self,s):
        self.x,self.y,self.orientation,self.das,self.soft_points = int(s[0]),int(s[1]),int(s[2]),int(s[4]),int(s[6])


def warm_paths(selector,levels=(100,154,156,157,158,168,170,173,247,249,250,253)):
    """Run the selector's real decision code once per zone level before play.

    numba compiles a new specialization when optional arguments are passed explicitly
    instead of omitted; a first compile mid-game cost two real runs a ~2.3 s planning
    stall at level 156 (an uncontrolled piece at level-29 speed).
    """
    from types import SimpleNamespace
    board = np.zeros((20,10),np.bool_)
    board[16:,:9] = True
    states = np.zeros((MAX_CANDIDATES,9),np.int64)
    for level in levels:
        state,_ = spawn_state(SPAWNS[6],0,board,min(level,29))
        n = candidates(board,state,min(level,29),selector.tap_period,states,selector.reaction)
        boards,cleared,dead,controls = [],[],[],[]
        for i in range(n):
            result,c,d = lock(board,int(states[i,0]),int(states[i,1]),int(states[i,2]))
            boards.append(result.astype(np.uint8)); cleared.append(c); dead.append(d); controls.append(_WarmControl(states[i]))
        plan = SimpleNamespace(actions=SimpleNamespace(boards=boards,lines=np.asarray(cleared),
            terminal=np.asarray(dead),controls=controls))
        lines = 18*10+10*(level-19)+5 if level>=19 else 5
        obs = SimpleNamespace(board=board.astype(np.uint8),level=level,lines=lines,score=MAXED,placements=0,
                              next_orientation=int(SPAWNS[2]))
        selector.score_actions(obs,plan)
    selector.switched = False
    selector.calls = 0
    for key in selector.modes:
        selector.modes[key] = 0
    return selector
