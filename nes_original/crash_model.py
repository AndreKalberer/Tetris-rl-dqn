"""Crash-map alerts for original NES levels 155-255, assuming no soft-drop pushdown.

Source: HydrantDude's crash model (TetrisGYM crash.asm and the community
"Crash Dodge" sheet), re-implemented and checked in
outputs/NES-CRASH-MAP-RESEARCH.md/.json (986/988 sheet cells agree).

Line clears, levels 155-173: each (pre level, post level, cleared) entry has
fatal probabilities for four variants of two conditions the bot can derive:
  rows high  - a cleared row is in board rows 0-10 (conservative for mixed rows)
  count carry - this clear makes that clear type's count since game start end
                in 0 (10th, 20th ... single). Counted from the bot's own
                observations; if a placement is missed, carry is assumed.
Zero-line locks, levels 247-255: risk depends on pushdown and piece class.
Down is suppressed at most of these levels (NO_SOFTDROP_LEVELS); at 249/253 a
non-T/J lock that clears no lines needs raw pushdown >= 2 (predicted >= 4 for margin).

An unalerted candidate is modelled safe, not proven safe on hardware.
"""
from nes.engine import CELLS
from nes_original.crash_alerts import DocumentedHazardPrioritizer,candidate_alerts

# (pre_level, post_level, cleared): fatal probability (base, carry, high, carry+high).
CRASH_MAP = {
    (154,155,1): (1,.896,.979,1), (155,156,1): (0,.021,0,0), (156,156,1): (0,.417,0,.75),
    (157,157,1): (.75,.583,.417,.896), (157,158,1): (1,0,1,.021), (158,158,1): (0,.25,0,.021),
    (158,159,1): (.25,.271,.021,.271), (165,166,4): (1,.51,.49,.997), (166,167,3): (0,.75,0,0),
    (166,167,4): (.781,.51,.69,.997), (167,167,4): (0,0,.888,0), (167,168,2): (0,.75,0,.104),
    (167,168,3): (.021,.271,0,.75), (168,168,3): (1,.583,.75,1), (168,168,4): (0,.003,0,0),
    (168,169,2): (.021,.271,0,.875), (168,169,4): (0,0,.018,0), (169,169,2): (1,.583,.979,1),
    (169,169,3): (.417,0,1,0), (169,169,4): (.69,1,.112,.331), (169,170,3): (1,0,1,.417),
    (169,170,4): (.69,1,.112,.331), (170,170,2): (.417,0,.979,0), (170,170,3): (0,.417,0,0),
    (170,171,2): (1,0,1,.104), (170,171,3): (0,.417,0,0), (171,171,2): (0,.417,0,0),
    (171,171,4): (.091,0,.948,0), (171,172,2): (0,.417,0,0), (172,172,3): (1,.896,.417,1),
    (173,173,2): (1,.896,.75,1),
}

# Line clears from this level up use the exact cycle model (crash_cycle_model),
# which also covers the second lap after the 255 -> 0 rebirth; CRASH_MAP stays
# as the first-lap reference table it was validated against.
USE_CYCLE_MODEL = True
MODEL_FROM_LEVEL = 150
# At levels where Down is normally suppressed, a clear may hold Down when the model
# rates its pushdown safer than none (the controller follows CrashMapPrioritizer.wants_down).
DOWN_OPTION = True

# Zero-line locks without pushdown that the model rates fatal (piece classes).
NEEDS_PUSHDOWN = {249: ('hZ','other'), 253: ('other',), 255: ('hZ','other')}
# The native controller can start Down a little later than the proposal path
# (charged planning latency). Over 199 real 249/253/255 locks the actual raw
# pushdown was 0 (65%) or 1 (35%) below the prediction, never more, so a
# prediction of 3 guarantees the model's raw >= 2.
MIN_PREDICTED_PUSHDOWN = 3

# Down at 1G only adds pushdown points, which shift crash timing. Keep it where
# zero-line locks need pushdown.
NO_SOFTDROP_LEVELS = tuple(level for level in range(155,256) if level not in NEEDS_PUSHDOWN)

SCOPE = ('Crash-map alerts from the HydrantDude cycle model (155-173 line clears with row/count-carry '
         'conditions; 247-255 zero-line pushdown); modelled, not hardware-certified')


def piece_class(orientation):
    return 'TJ' if orientation<8 else 'hZ' if orientation==8 else 'other'


def name(pre_level,post_level,cleared):
    kind = 'entering' if post_level!=pre_level else 'within'
    return f'crash_map_{("single","double","triple","tetris")[cleared-1]}_{kind}_{post_level}'


def cleared_rows(board,control):
    """Board rows (0 = top) that the locked piece completes."""
    filled = board!=0
    cells = [(control.y+dy,control.x+dx) for dx,dy in CELLS[control.orientation]]
    rows = []
    for row in sorted({y for y,_ in cells if 0<=y<20}):
        if all(filled[row,col] or (row,col) in cells for col in range(10)):
            rows.append(row)
    return rows


def crash_map_alerts(observation,plan,counts=None):
    """Rows like crash_alerts.candidate_alerts, with alerts from this model.

    counts: clears of each type since game start (index 1..4), or None if unknown.
    """
    rows = candidate_alerts(observation,plan)
    actions = plan.actions
    controls = getattr(actions,'controls',None)
    board = getattr(observation,'board',None)
    for i,row in enumerate(rows):
        cleared = row['predicted_cleared_lines']
        key = (row['pre_level'],row['predicted_level'],cleared)
        probability = 0.
        detail = None
        if USE_CYCLE_MODEL and cleared and row['pre_level']>=MODEL_FROM_LEVEL:
            # Exact cycle model: any lap, exact number of high cleared rows, true
            # clear-count carry when counts are known (else worst case).
            carry = None if counts is None else (counts[cleared]+1)%10==0
            high_rows = None
            if controls is not None and board is not None:
                high_rows = sum(r<=10 for r in cleared_rows(board,controls[i]))
            level = row['pre_level']

            def risk(pushdowns):
                return max(model_crash_probability(level,int(observation.lines),cleared,rows_high=high_rows,
                    pushdown_raw=pd,clear_count_carry=carry) for pd in pushdowns)
            # With Down held the actual raw pushdown is the prediction or one less (measured).
            predicted = int(controls[i].soft_points) if controls is not None else 0
            with_down = tuple({predicted,max(0,predicted-1)})
            if level in NO_SOFTDROP_LEVELS:
                # Down is normally suppressed here; hold it for this clear only if the
                # model rates that strictly safer (e.g. single entering 158: 100% -> 0%).
                probability = no_down = risk((0,))
                if DOWN_OPTION and controls is not None and predicted>=2:
                    down = risk(with_down)
                    if down<no_down-1e-9:
                        probability = down
                        row['use_down'] = True
            else:  # Down kept at 249/253/255
                probability = risk(with_down) if controls is not None else risk((0,))
            detail = dict(rows_high=high_rows,count_carry=carry,counts_known=counts is not None,
                use_down=bool(row.get('use_down')),predicted_pushdown=predicted,source='cycle_model')
            alert = name(*key)
        elif key in CRASH_MAP:
            # Unknown condition: take the worst variant still possible.
            carries = (False,True) if counts is None else ((counts[cleared]+1)%10==0,)
            if controls is not None and board is not None:
                highs = (any(r<=10 for r in cleared_rows(board,controls[i])),)
            else:
                highs = (False,True)
            table = dict(zip(((False,False),(True,False),(False,True),(True,True)),CRASH_MAP[key]))
            probability = max(table[carry,high] for carry in carries for high in highs)
            detail = dict(rows_high=list(highs),count_carry=list(carries),counts_known=counts is not None)
            alert = name(*key)
        elif cleared==0 and row['pre_level'] in NEEDS_PUSHDOWN and controls is not None:
            control = controls[i]
            kind = piece_class(control.orientation)
            if kind in NEEDS_PUSHDOWN[row['pre_level']] and control.soft_points<MIN_PREDICTED_PUSHDOWN:
                probability = 1.
                detail = dict(piece_class=kind,predicted_pushdown=int(control.soft_points))
                alert = f'crash_map_zero_line_{kind}_no_pushdown_{row["pre_level"]}'
        if controls is not None:
            row['predicted_pushdown'] = int(controls[i].soft_points)
        row['alert'] = alert if probability>0 else None
        row['modelled_crash_probability'] = float(probability)
        row['alert_detail'] = detail
        row['alert_origin'] = 'hydrant_cycle_model' if probability>0 else None
    return rows


# A top-out ends the attempt for certain, so with a high stack a clear the model
# rates only partly risky is better than stacking higher: (min stack rows, max risk).
RISK_BUDGET = ((17,.9),(14,.5),(11,.3))


def stack_height(board):
    import numpy as np
    filled = np.asarray(board)!=0
    rows = np.flatnonzero(filled.any(axis=1))
    return 0 if not len(rows) else 20-int(rows[0])


def allowed_risk(board,table=None):
    if board is None:
        return 0.
    height = stack_height(board)
    return next((risk for rows,risk in (table or RISK_BUDGET) if height>=rows),0.)


# From this level on, a clear whose count would end in 0 is taken as soon as it is
# modelled safe, so it is not left pending for a level where it would crash.
PREEMPT_FROM = 150
PREEMPT_BONUS = 1.e7


PUSHDOWN_FALLBACK_BONUS = 1.e8
# Counter-trap lookahead: penalize ending on the line count just before a level change
# when the single that would enter the next level is risky (e.g. 1,489 at level 154:
# every single enters 155, ~100% fatal even with ordinary pushdown). Penalty = TRAP_PENALTY x that single's modelled fatal
# probability (best of no Down / ordinary Down), in base-score units.
TRAP_PENALTY = 30.
TRAP_FROM_LEVEL = 150


def trap_risk(level,lines,singles_so_far):
    """Fatal probability of the next single from this counter state (best Down option)."""
    if level<TRAP_FROM_LEVEL:
        return 0.
    from nes_original.progression import after_clear
    if after_clear(level,lines,1)[0]==level:
        return 0.  # within-level hazards cannot be dodged by the line count; only level-entry ones
    carry = None if singles_so_far is None else (singles_so_far+1)%10==0
    return min(model_crash_probability(level,lines,1,rows_high=0,pushdown_raw=pd,clear_count_carry=carry)
               for pd in (0,3))


class _PreemptScores:
    """Base scores plus a bonus for safe clears that resolve a pending count carry.

    Also, at 249/253/255, flagged zero-line locks are ranked by predicted
    pushdown first: if every survivor is flagged, the fallback takes the one with
    the most pushdown (least modelled risk), not the best board.
    """
    def __init__(self,owner,base):
        self.owner,self.base = owner,base

    def score_actions(self,observation,plan):
        import numpy as np
        scores = np.array(self.base.score_actions(observation,plan),dtype=float,copy=True)
        owner = self.owner
        if owner.trap_penalty and observation.level>=TRAP_FROM_LEVEL-2 and owner.last_rows:
            from nes_original.progression import after_clear
            for row in owner.last_rows:
                if row['predicted_terminal']:
                    continue
                k = row['predicted_cleared_lines']
                singles = None
                if owner.counts_known:
                    singles = owner.counts[1]+(1 if k==1 else 0)
                level,lines = row['predicted_level'],row['predicted_lines']
                risk = trap_risk(level,lines,singles)
                if risk:
                    scores[row['index']] -= owner.trap_penalty*risk
                    row['trap_risk'] = risk
        if observation.level in NEEDS_PUSHDOWN and owner.last_rows:
            for row in owner.last_rows:
                if row['alert'] is not None and row.get('predicted_pushdown') is not None:
                    scores[row['index']] += PUSHDOWN_FALLBACK_BONUS*row['predicted_pushdown']
        if owner.counts_known and observation.level>=PREEMPT_FROM and owner.last_rows:
            for row in owner.last_rows:
                k = row['predicted_cleared_lines']
                if (k and row['alert'] is None and not row['predicted_terminal']
                        and (owner.counts[k]+1)%10==0):
                    scores[row['index']] += PREEMPT_BONUS
                    owner.preemptions += 1
        return scores


class CrashMapPrioritizer(DocumentedHazardPrioritizer):
    """Prefer surviving candidates the crash model rates safe; track clear counts."""
    def __init__(self,base_selector=None,preempt=True,risk_budget=True,trap_penalty=TRAP_PENALTY):
        super().__init__(base_selector)
        self.trap_penalty = trap_penalty
        # Load and exercise the cycle model now: a first use mid-game cost a real run
        # a 134-frame planning stall (an uncontrolled piece at level-29 speed).
        model_crash_probability(155,1495,1,rows_high=0,pushdown_raw=0,clear_count_carry=False)
        model_crash_probability(249,3235,0,pushdown_raw=0,piece_class='other')
        trap_risk(154,1489,5)
        self.risk_budget = risk_budget
        self.budget_table = None  # None = RISK_BUDGET
        self.accepted_risks = 0
        self.wants_down = False
        self.down_choices = 0
        if preempt:
            self.base_selector = _PreemptScores(self,self.base_selector)
        self.last_rows = None
        self.preemptions = 0
        self.alert_candidates = {}
        self.counts = [0]*5
        self.counts_known = True
        self.previous = None

    def _track(self,observation):
        previous = self.previous
        self.previous = (observation.placements,observation.lines)
        if previous is None or observation.placements<previous[0]:
            # New game: counts start at zero only if we see it from the start.
            self.counts = [0]*5
            self.counts_known = observation.placements==0 and observation.lines==0
            return
        placed = observation.placements-previous[0]
        lines = observation.lines-previous[1]
        if placed==1 and 0<=lines<=4:
            self.counts[lines] += 1
        elif placed!=0 or lines!=0:
            self.counts_known = False

    def _candidate_alerts(self,observation,plan):
        self._track(observation)
        rows = crash_map_alerts(observation,plan,self.counts if self.counts_known else None)
        budget = allowed_risk(getattr(observation,'board',None),self.budget_table) if self.risk_budget else 0.
        for row in rows:
            if row['alert'] is not None and 0<row['modelled_crash_probability']<=budget:
                row['accepted_risk'] = dict(alert=row['alert'],stack_rows=stack_height(observation.board),
                    budget=budget)
                row['alert'] = None
                self.accepted_risks += 1
        self.last_rows = rows
        for row in rows:
            if row['alert'] is not None:
                self.alert_candidates.setdefault(row['alert'],0)
        return rows

    def __call__(self,observation,plan):
        selected = super().__call__(observation,plan)
        # The controller holds Down for this piece only if the chosen clear is safer with it.
        self.wants_down = bool(self.last_rows and self.last_rows[selected].get('use_down'))
        self.down_choices += int(self.wants_down)
        self.last_decision.update(scope=SCOPE,alert_profile='hydrant_crash_map_no_pushdown',wants_down=self.wants_down,
            clear_counts=list(self.counts),clear_counts_known=self.counts_known)
        return selected

    def stats(self):
        result = super().stats()
        result.update(scope=SCOPE,alert_profile='hydrant_crash_map_no_pushdown',entries=len(CRASH_MAP),
            clear_counts=list(self.counts),clear_counts_known=self.counts_known,preemptions=self.preemptions,
            accepted_risk_candidates=self.accepted_risks,down_choices=self.down_choices,risk_budget=RISK_BUDGET if self.risk_budget else None,
            base=self.base_selector.base.stats() if hasattr(self.base_selector,'base') and
                hasattr(self.base_selector.base,'stats') else None,
            no_softdrop_levels='155-255 except '+'/'.join(map(str,NEEDS_PUSHDOWN)))
        return result


def model_crash_probability(observation_level,observation_lines,cleared,rows_high=None,pushdown_raw=0,
                            clear_count_carry=None,piece_class=None):
    """Fatal-crash probability from the full cycle model (nes_original.crash_cycle_model).

    Unlike CRASH_MAP (first lap, levels 155-173 only) this works at any 8-bit level and
    raw line counter, including the second lap after the 255 -> 0 wrap. Inputs are the
    RAM level / raw line counter before the lock, lines cleared, how many cleared rows are
    board rows 0..10, raw holdDownPoints, whether this clear makes its type's count end in
    0, and the piece class ('TJ'/'hZ'/'other', zero-line locks only). Any of rows_high,
    clear_count_carry or piece_class left as None is treated as unknown: the worst case.
    Graphics corruption is not counted as fatal, matching CRASH_MAP. Not used by the bot yet.
    """
    from nes_original.crash_cycle_model import worst_fatal
    if cleared==0:
        clear_count_carry = False
    return worst_fatal(observation_level,observation_lines,cleared,rows_high=rows_high,
        pushdown_raw=pushdown_raw,clear_count_carry=clear_count_carry,piece_class=piece_class)
