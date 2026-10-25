"""Original Nintendo USA A-Type counter transitions, including late glitches.

Derived from the pinned incrementLines routine's CMP/BPL, not a level/10
formula. This projects at most four cleared lines for approximate search; the
actual emulator remains authoritative. Raw line high byte is binary hundreds.
"""


def after_clear(level,raw_lines,cleared):
    if not 0<=level<=255 or not 0<=raw_lines<25600 or not 0<=cleared<=4:
        raise ValueError('Expected an8-bit level, raw line counter and0..4 cleared lines')
    for _ in range(cleared):
        raw_lines=(raw_lines+1)%25600
        if raw_lines%10 == 0:
            hundreds,low=divmod(raw_lines,100)
            threshold=((hundreds&15)<<4)|(low//10)
            # The ROM tests the sign bit of8-bit CMP, not an unsigned less-than.
            if ((level-threshold)&255)&128:
                level=(level+1)&255
    return level,raw_lines


class MarathonProgress:
    """Track cycles/absolute lines at validated stable piece boundaries only.

    Reinitialize explicitly on a new game. Mid-copy RAM observations can be
    torn; callers must supply settled boundaries rather than every CPU sample.
    """
    def __init__(self,level,raw_lines,placements=0):
        self.level,self.raw_lines,self.placements=level,raw_lines,placements
        self.level_cycles=self.line_cycles=0
        self.absolute_lines=raw_lines

    def update(self,level,raw_lines,placements):
        if not 0<=level<=255 or not 0<=raw_lines<25600 or placements<self.placements:
            raise ValueError('Invalid or reset progress observation')
        placed=placements-self.placements
        delta=(raw_lines-self.raw_lines)%25600
        if delta>4*placed:
            raise ValueError('Unsettled/inconsistent line and placement counters')
        expected,expected_lines=after_clear(self.level,self.raw_lines,delta) if delta<=4 else (None,None)
        if delta<=4 and (level!=expected or raw_lines!=expected_lines):
            raise ValueError('Level/line transition differs from original counter routine')
        if level<self.level:
            if delta<=4 and (self.level,level)!=(255,0):
                raise ValueError('Unexpected level decrease')
            self.level_cycles+=1
        if raw_lines<self.raw_lines:
            self.line_cycles+=1
        self.absolute_lines+=delta
        self.level,self.raw_lines,self.placements=level,raw_lines,placements
        return dict(raw_level=level,level_cycles=self.level_cycles,
            absolute_level=256*self.level_cycles+level,raw_lines=raw_lines,
            line_counter_cycles=self.line_cycles,absolute_lines=self.absolute_lines)
