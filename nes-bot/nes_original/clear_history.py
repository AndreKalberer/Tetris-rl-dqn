"""Actual clear-event history from complete, publicly observed piece boundaries.

One single/double/triple/Tetris event increments its own count once, not by
the number of lines. No count is inferred from total lines, RAM, or RNG.
Callers must explicitly certify settled observations; a lost boundary makes
history unknown until an explicit genuine new-game reset.
"""
from nes_original.progression import after_clear


def encode_clear_events(count):
    """Original clear-stat byte: decimal units plus a four-bit tens counter.

    This is not ordinary unlimited BCD:159 ->0xF9,160 ->0x00. Its period is160.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError('Clear-event count must be a nonnegative integer')
    return 16*((count//10) % 16) + count % 10


def _valid(level, raw_lines, placements):
    return (all(isinstance(value, int) and not isinstance(value, bool)
                for value in (level, raw_lines, placements))
            and 0 <= level <= 255 and 0 <= raw_lines < 25600 and placements >= 0)


class ClearHistory:
    """Four event counts, known only after an unbroken new-game history.

    reset/update consume public raw level, decoded raw lines, and placement
    count at a certified settled boundary. update returning known=False is a
    data-quality outcome, not permission to guess counts from the line total.
    The first three values deliberately match MarathonProgress.update.
    """
    def __init__(self):
        self.known = False
        self.reason = 'explicit_new_game_reset_required'
        self._counts = [0, 0, 0, 0]
        self._boundary = None
        self.last_update = 'unknown'
        self.last_clear = self.last_candidate_c = None

    @property
    def counts(self):
        return tuple(self._counts) if self.known else None

    @property
    def raw_counters(self):
        return tuple(encode_clear_events(count) for count in self._counts) if self.known else None

    def candidate_c(self, cleared):
        """Candidate clear-stat carry for the NEXT event, using its PRE count.

        False for no clear; None for an unknown nonzero-clear history. This
        identifies a carry candidate, never a guarantee of a native crash.
        """
        if isinstance(cleared, bool) or not isinstance(cleared, int) or not 0 <= cleared <= 4:
            raise ValueError('Expected cleared lines0..4')
        if cleared == 0:
            return False
        return self._counts[cleared-1] % 10 == 9 if self.known else None

    def snapshot(self):
        return dict(known=self.known, reason=self.reason,
            clear_events=list(self.counts) if self.known else None,
            encoded_counters=list(self.raw_counters) if self.known else None,
            candidate_c_by_clear=[self.candidate_c(k) for k in range(1, 5)],
            last_update=self.last_update, last_clear=self.last_clear,
            last_candidate_c=self.last_candidate_c)

    def _unknown(self, reason):
        self.known = False
        self.reason = reason
        self.last_update = 'unknown'
        self.last_clear = self.last_candidate_c = None
        return self.snapshot()

    def reset(self, level, raw_lines, placements, *, settled=True):
        """Explicit new game only; observed18/0/0 is the sole known origin."""
        self._counts = [0, 0, 0, 0]
        self._boundary = None
        if settled is not True:
            return self._unknown('unsettled_reset')
        if not _valid(level, raw_lines, placements):
            return self._unknown('invalid_reset_observation')
        if (level, raw_lines, placements) != (18, 0, 0):
            return self._unknown('reset_is_not_observed_level18_new_game')
        self.known, self.reason = True, None
        self._boundary = (level, raw_lines, placements)
        self.last_update = 'reset'
        self.last_clear = self.last_candidate_c = None
        return self.snapshot()

    def update(self, level, raw_lines, placements, *, settled=True):
        if settled is not True:
            return self._unknown('unsettled_observation')
        if not _valid(level, raw_lines, placements):
            return self._unknown('invalid_observation')
        if not self.known:
            # Observing another apparent start cannot silently repair a gap.
            return self.snapshot()
        previous_level, previous_lines, previous_placements = self._boundary
        if (level, raw_lines, placements) == self._boundary:
            self.last_update = 'duplicate'
            self.last_clear = self.last_candidate_c = None
            return self.snapshot()
        if placements-previous_placements != 1:
            return self._unknown('missing_or_inconsistent_placement_boundary')
        cleared = (raw_lines-previous_lines) % 25600
        if cleared > 4:
            return self._unknown('inconsistent_line_delta')
        expected_level, expected_lines = after_clear(previous_level, previous_lines, cleared)
        if (level, raw_lines) != (expected_level, expected_lines):
            return self._unknown('inconsistent_original_level_transition')
        self.last_candidate_c = self.candidate_c(cleared)
        self.last_clear = cleared
        if cleared:
            self._counts[cleared-1] += 1
        self._boundary = (level, raw_lines, placements)
        self.last_update = 'clear' if cleared else 'no_clear'
        return self.snapshot()
