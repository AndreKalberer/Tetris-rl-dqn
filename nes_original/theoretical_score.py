"""Uncapped ("theoretical") score of a played original-game run.

The original cartridge stops its score at 999,999 (a "max out"); modified
cartridges that remove the cap show millions with A-F in the top digit. The
theoretical score is the uncapped total, which the original game does not
display. This rebuilds it from a run's gameplay events: every lock has an
active_exit row (player RAM $40-$5F and raw pushdown when the piece locks).
Lines cleared by a lock = line counter at the next lock minus this one; the
ROM's own score routine (post-clear level, soft-drop decimal quirk) is applied
with the 999,999 clamp replaced by a carry into millions.

Check: until the real counter reaches 999,999 the rebuilt total must match it.
The last lock of a run is not counted (no following observation).
"""
import argparse
import json
from pathlib import Path

BASE = (0,40,100,300,1200)
LEVEL,LINES_LOW,LINES_HIGH,SCORE = 4,16,17,19   # offsets from $40; score $53-$55 low byte first


def bcd(byte):
    return (byte>>4)*10+(byte&15)


def counters(ram_hex):
    ram = bytes.fromhex(ram_hex)
    score = bcd(ram[SCORE])+100*bcd(ram[SCORE+1])+10000*bcd(ram[SCORE+2])
    return dict(level=ram[LEVEL],lines=bcd(ram[LINES_LOW])+100*ram[LINES_HIGH],score=score)


def award(state,soft_points,completed,level):
    """The ROM's score routine (see nes.engine.Game._award_score), uncapped.

    state = [low, mid, high BCD bytes, millions]. Where the ROM clamps to
    999,999, carry into the millions count instead (what A-F digits display).
    """
    s = state
    if soft_points>=2:
        s[0] = ((s[0]-1)&255)+soft_points
        s[0] &= 255
        if s[0]&15>=10:
            s[0] = (s[0]+6)&255
        if s[0]&0xF0>=0xA0:
            s[0] = ((s[0]&0xF0)+0x60)&255
            s[1] = (s[1]+1)&255
    base = (0,0x0040,0x0100,0x0300,0x1200)[completed]
    for _ in range(level+1):
        s[0] = (s[0]+(base&255))&255
        if s[0]>=0xA0:
            s[0] = (s[0]+0x60)&255
            s[1] = (s[1]+1)&255
        s[1] = (s[1]+(base>>8))&255
        if s[1]&15>=10:
            s[1] = (s[1]+6)&255
        if s[1]&0xF0>=0xA0:
            s[1] = (s[1]+0x60)&255
            s[2] = (s[2]+1)&255
        if s[2]&15>=10:
            s[2] = (s[2]+6)&255
        if s[2]&0xF0>=0xA0:
            s[2] = (s[2]-0xA0)&255   # the ROM sets 99 99 99 here
            s[3] += 1


def value(state):
    return state[3]*1000000+bcd(state[2])*10000+bcd(state[1])*100+bcd(state[0])


def timeline(events_path):
    """[(controller-read index, uncapped score)] once each lock's score is known.

    A lock's points are known at the following lock (its line count and level).
    """
    locks = []
    for line in Path(events_path).read_text(encoding='utf-8').splitlines():
        event = json.loads(line)
        if event.get('event')=='active_exit':
            locks.append((int(event['input_count']),counters(event['before_player_ram_hex']),
                int(event['raw_pushdown'])))
    state = [0,0,0,0]
    rows = []
    for (_,now,pushdown),(index,after,_) in zip(locks,locks[1:]):
        award(state,pushdown,(after['lines']-now['lines'])%25600,after['level'])
        rows.append((index,value(state)))
    return rows


def theoretical(events_path):
    locks = []
    for line in Path(events_path).read_text(encoding='utf-8').splitlines():
        event = json.loads(line)
        if event.get('event')=='active_exit':
            locks.append((counters(event['before_player_ram_hex']),int(event['raw_pushdown'])))
    state = [0,0,0,0]
    clears = [0]*5
    checked = mismatches = 0
    maxout = None
    for (now,pushdown),(after,_) in zip(locks,locks[1:]):
        cleared = (after['lines']-now['lines'])%25600
        if not 0<=cleared<=4:
            raise ValueError(f'Unexpected line delta {cleared} between locks')
        award(state,pushdown,cleared,after['level'])
        total = value(state)
        clears[cleared] += 1
        if after['score']<999999:
            checked += 1
            mismatches += int(after['score']!=total)
        elif maxout is None:
            maxout = dict(lines=after['lines'],level=after['level'])
    return dict(theoretical_score=value(state),displayed_score_cap=999999,locks=len(locks),
        clears=dict(zip(('none','single','double','triple','tetris'),clears)),maxout=maxout,
        checked_against_real_counter=checked,counter_mismatches=mismatches)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('events',type=Path,nargs='+',help='gameplay-seed-*-events.jsonl files')
    args = p.parse_args()
    for path in args.events:
        print(path,json.dumps(theoretical(path)),flush=True)


if __name__=='__main__':
    main()
