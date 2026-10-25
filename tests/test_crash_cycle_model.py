"""Validation of nes_original.crash_cycle_model against the research data.

Data in evidence/nes-crash-model/: HydrantDude "Crash Dodge" and "List of crashes by
setup" sheet tabs (CSV exports, sha256 as in NES-CRASH-MAP-RESEARCH.json) and the table
built by the research JS model (build.js, research-model-table.json).
"""
import csv
import json
import re
import unittest
from pathlib import Path

from nes_original.crash_cycle_model import (JITTER,SWITCH_OFFSETS,classify,crash_outcomes,cycle_base,
    distribution,fatal_hazard_table,pushdown_category,walk_progress)
from nes_original.crash_model import CRASH_MAP,NEEDS_PUSHDOWN,model_crash_probability

DATA = Path(__file__).resolve().parent.parent/'evidence'/'nes-crash-model'
PD = {'std':'none','p':'p','P':'P','NNB':'none','C':'none'}


def sheet_cells():
    """(label, sheet probability, cycle_base kwargs) for every Crash Dodge cell (as val2.js)."""
    rows = list(csv.reader(open(DATA/'hydrant-crash-dodge.csv',encoding='utf-8')))
    level = None
    cols = dict(std=2,p=3,P=4,NNB=5,C=6,tstd=7,tp=8,tP=9,tNNB=10,tC=11)
    value = lambda s: float(re.sub(r'[^0-9.]','',s))/100 if s.strip() else 0.
    for r in rows[2:]:
        if r[0]:
            level = int(r[0])
        if not level or not r[1]:
            continue
        n = int(r[1])
        if level<=176:
            for k,idx in cols.items():
                trans = k.startswith('t')
                kk = k[1:] if trans else k
                yield (f'{level} n{n} {k}',value(r[idx]),dict(post_level=level,cleared=n,tens_carry=trans,
                    level_up=trans,hundreds_carry=trans and level%10==6,pushdown=PD[kk],
                    clear_count_carry=kk=='C',next_box=kk!='NNB',gym_clear_count=True))
        if level>=247:
            mod = ('std','p','P','NNB')[n-1]
            for piece,idx in (('TJ',12),('hZ',13),('other',14)):
                yield (f'{level} zero {mod} {piece}',value(r[idx]),dict(post_level=level,cleared=0,
                    pushdown=PD[mod],next_box=mod!='NNB',piece_class=piece))


def fatal(**kw):
    return distribution(kw.pop('post_level'),kw.pop('cleared'),**kw).get('crash',0.)


class SheetValidation(unittest.TestCase):
    def test_crash_dodge_agreement(self):
        cells = list(sheet_cells())
        agree = [((fatal(**kw)==0)==(v==0)) for _,v,kw in cells]
        mismatches = sorted(label for (label,_,_),ok in zip(cells,agree) if not ok)
        error = sum(abs(fatal(**kw)-v) for _,v,kw in cells)/len(cells)
        self.assertEqual(len(cells),988)
        self.assertEqual(sum(agree),986)
        # 156 transition single p is a graphics ("26.67%G") cell; 160 transition double NNB.
        self.assertEqual(mismatches,['156 n1 tp','160 n2 tNNB'])
        self.assertLess(error,0.0065)

    def test_setup_list_timings(self):
        rows = list(csv.reader(open(DATA/'hydrant-crashes-by-setup.csv',encoding='utf-8')))
        total = matched = 0
        unmatched = []
        for r in rows[1:40]:
            if not r[0] or not r[1].strip().isdigit():
                continue
            timing = int(r[1])
            for token in filter(None,re.split(r'[ ,]+',r[0])):
                m = re.fullmatch(r'([0-4])([hv]?)([-+])(T?)(\d+)([pPNC]*)',token)
                self.assertIsNotNone(m,token)
                n,piece,sign,trans,level,mods = m.groups()
                n,level,trans = int(n),int(level),trans=='T'
                base = cycle_base(level,n,tens_carry=trans,level_up=trans,hundreds_carry=trans and level%10==6,
                    pushdown='p' if 'p' in mods else 'P' if 'P' in mods else 'none',clear_count_carry='C' in mods,
                    next_box='N' not in mods,gym_clear_count=True,tetris_frame_mod8_zero=n==4 and sign=='+',
                    piece_class='other' if n or piece=='v' else 'hZ' if piece=='h' else 'TJ')
                total += 1
                if any(-10<=timing-base-o<=12 for o in SWITCH_OFFSETS):
                    matched += 1
                else:
                    unmatched.append(token)
        self.assertEqual(total,155)
        self.assertEqual(matched,151)
        # 3-157p is a label typo for 3-167p; 4+160p likely a typo; 3-T173N/2-T174N duplicates.
        self.assertEqual(sorted(unmatched),['2-T174N','3-157p','3-T173N','4+160p'])
        self.assertEqual(cycle_base(167,3,pushdown='p')+SWITCH_OFFSETS[-1],29745)


class ResearchTableRegression(unittest.TestCase):
    """Every entry the research JS (build.js) produced, recomputed in Python."""
    NAMES = dict(crash='crash',graphics='graphics',confettiA='confetti_limited',confettiB='confetti_infinite',
                 line_lag='line_lag',level_lag='level_lag',ok='ok')

    def check(self,expected,got):
        self.assertEqual(expected['model_cycle_base'],got['cycle_base'])
        exp = {self.NAMES[k]:v for k,v in expected.items() if k!='model_cycle_base'}
        self.assertEqual(set(exp),{k for k,v in got.items() if k!='cycle_base' and round(v,3)>0})
        for k,v in exp.items():
            self.assertAlmostEqual(v,got[k],places=3)

    def test_full_table(self):
        table = json.load(open(DATA/'research-model-table.json'))
        count = 0
        for rec in table['hazards']+table['nonfatal_stalls']:
            entering = rec['kind']=='entering'
            tens = rec['kind']!='within'
            for name,carry,high in (('base',0,0),('clear_count_carry',1,0),('all_rows_high',0,1),('carry_and_high',1,1)):
                got = distribution(rec['post_level'],rec['cleared'],rows_high=rec['cleared']*high,tens_carry=tens,
                    hundreds_carry=rec['hundreds_carry'],level_up=entering,pushdown=rec['pushdown'],
                    clear_count_carry=bool(carry))
                self.check(rec['variants'][name],got)
                count += 1
        for rec in table['zero_clear_hazards']:
            got = distribution(rec['level'],0,pushdown=rec['pushdown'],piece_class=rec['piece_class'])
            self.check({k:v for k,v in rec.items() if k not in ('level','cleared','pushdown','piece_class')},got)
            count += 1
        self.assertGreater(count,500)


class ObservedNativeCases(unittest.TestCase):
    """Retained native terminations (NES-CRASH-MAP-RESEARCH.json observed_cases)."""
    # (pre_level, pre_lines, cleared, raw_pushdown, post_level, base as played, crash as played,
    #  crash with no pushdown, crash with P pushdown)
    CASES = [
        (159,1535,1,3,159,29682,.271,0,.25),     # dellacherie-alerts-v1 33: single within 159
        (167,1610,3,6,167,29183,.417,0,.75),     # conditional-right-rescue-v1 33: triple within 167
        (168,1623,2,5,168,29183,.417,0,.75),     # conditional-right-rescue-v1 49
        (169,1638,2,4,170,29530,1.,0,.208),      # right-a-rescue-v2 33: double entering 170
        (168,1621,2,2,168,29183,.417,0,.75),     # right-a-rescue-v2 49: double within 168
    ]

    def test_fatal_cases(self):
        for pre,lines,n,raw,post,base,played,none,big in self.CASES:
            o = crash_outcomes(pre,lines,n,rows_high=0,pushdown_raw=raw,clear_count_carry=False)
            self.assertEqual((o['post_level'],o['cycle_base'],o['pushdown_category']),(post,base,'p'))
            self.assertAlmostEqual(o['fatal'],played,places=3)
            self.assertAlmostEqual(crash_outcomes(pre,lines,n,rows_high=0,pushdown_raw=0,
                clear_count_carry=False)['fatal'],none,places=3)
            self.assertAlmostEqual(crash_outcomes(pre,lines,n,rows_high=0,pushdown_raw=8,
                clear_count_carry=False)['fatal'],big,places=3)

    def test_stalls_and_unlogged_pushdown(self):
        for raw in (2,4):   # entry-down-v1: single entering 161 -> limited confetti
            o = crash_outcomes(160,1549,1,rows_high=0,pushdown_raw=raw,clear_count_carry=False)
            self.assertEqual((o['confetti_limited'],o['cycle_base']),(1.,30051))
        # alerts-neutral159-v1 33: actual event single 1609 -> 1610 entering 167: infinite confetti.
        self.assertEqual(crash_outcomes(166,1609,1,rows_high=0,pushdown_raw=6,clear_count_carry=False)
                         ['confetti_infinite'],1.)
        # 165/1599: single entering 166 -> line lag only (non-fatal).
        o = crash_outcomes(165,1599,1,rows_high=0,pushdown_raw=3,clear_count_carry=False)
        self.assertEqual((o['fatal'],o['lag']),(0.,1.))
        # Recorded as zero-line locks (cached lines): base 22365 / 22448, ok.
        self.assertEqual(crash_outcomes(165,1599,0,rows_high=0,pushdown_raw=3,clear_count_carry=False,
                                        piece_class='other')['cycle_base'],22365)
        # Single entering 155 (1489 -> 1490): 100% none, 87.5% p, 0% P.
        f = [crash_outcomes(154,1489,1,rows_high=0,pushdown_raw=r,clear_count_carry=False)['fatal'] for r in (0,3,8)]
        self.assertEqual([round(x,3) for x in f],[1.,.875,0.])
        # Single within 157 (1511 -> 1512): 75% none, 25% p, 0% P.
        f = [crash_outcomes(157,1511,1,rows_high=0,pushdown_raw=r,clear_count_carry=False)['fatal'] for r in (0,3,8)]
        self.assertEqual([round(x,3) for x in f],[.75,.25,0.])


class CrashMapConsistency(unittest.TestCase):
    def test_crash_map_equals_model_first_lap(self):
        starts = {}
        for lap,level,lines in walk_progress(18,0,1):
            starts.setdefault(level,lines)
        for (pre,post,n),expected in CRASH_MAP.items():
            # Representative first-lap counts: entering from one line below the level start,
            # within from two lines after it.
            lines = starts[post]-1 if post!=pre else starts[pre]+2
            got = [crash_outcomes(pre,lines,n,rows_high=n*high,pushdown_raw=0,clear_count_carry=carry)
                   for carry,high in ((0,0),(1,0),(0,1),(1,1))]
            self.assertEqual({o['post_level'] for o in got},{post})
            for e,o in zip(expected,got):
                self.assertAlmostEqual(e,o['fatal'],delta=5e-4,msg=(pre,post,n))

    def test_first_lap_table_has_exactly_crash_map_keys(self):
        table = fatal_hazard_table(0)
        self.assertEqual(set(table),set(CRASH_MAP))
        for key,e in table.items():
            got = (e['base'],e['clear_count_carry'],e['all_rows_high'],e['carry_and_high'])
            for a,b in zip(CRASH_MAP[key],got):
                self.assertAlmostEqual(a,b,delta=5e-4)

    def test_model_crash_probability_wrapper(self):
        self.assertEqual(model_crash_probability(154,1489,1,0,0,False,None),1.)
        self.assertAlmostEqual(model_crash_probability(156,1505,1,0,0,True,None),.417,places=3)
        self.assertEqual(model_crash_probability(156,1505,1,0,0,False,None),0.)
        # Unknown rows/carry -> worst case, which for 167->168 triples is a mixed-row case.
        self.assertAlmostEqual(model_crash_probability(167,1679,3),.875,places=3)
        self.assertAlmostEqual(model_crash_probability(249,3150,0,piece_class='other'),.583,places=3)
        self.assertEqual(model_crash_probability(249,3150,0,pushdown_raw=3,piece_class='other'),0.)


class SecondLap(unittest.TestCase):
    def test_wrap_and_lap2_starts(self):
        starts = {}
        for lap,level,lines in walk_progress(18,0,2):
            starts.setdefault((lap,level),lines)
        self.assertEqual((starts[0,255],starts[1,0],starts[1,155],starts[1,190],starts[1,255]),
                         (3290,3300,4850,6000,6650))
        self.assertEqual(starts[1,190]-starts[1,189],810)   # lap-2 long level, like 235 on lap 1

    def test_lap2_differences(self):
        lap1,lap2 = fatal_hazard_table(0),fatal_hazard_table(1)
        changed = sorted(k for k in set(lap1)|set(lap2)
                         if k not in lap1 or k not in lap2
                         or any(abs(lap1[k][v]-lap2[k][v])>1e-9 for v in ('base','clear_count_carry',
                                'all_rows_high','carry_and_high','any_rows_mix')))
        self.assertEqual(changed,[(155,156,1),(165,166,4),(169,170,3),(169,170,4)])
        self.assertNotIn((155,156,1),lap2)
        self.assertAlmostEqual(lap2[169,170,3]['base'],0.)
        self.assertAlmostEqual(lap2[169,170,4]['base'],1.)

    def test_lap2_zero_line_rules(self):
        for lap,level,lines in walk_progress(18,0,2):
            if lap!=1 or level<247:
                continue
            for piece in ('TJ','hZ','other'):
                f = lambda raw: crash_outcomes(level,lines,0,rows_high=0,pushdown_raw=raw,clear_count_carry=False,
                                               piece_class=piece)['fatal']
                self.assertEqual(f(0)>0,piece in NEEDS_PUSHDOWN.get(level,()),(level,piece))
                if piece in NEEDS_PUSHDOWN.get(level,()):
                    self.assertEqual(f(3),0.)


class Units(unittest.TestCase):
    def test_jitter_and_pushdown_categories(self):
        self.assertEqual(sum(JITTER.values()),48)
        self.assertEqual((min(JITTER),max(JITTER)),(0,8))
        self.assertEqual([pushdown_category(r) for r in (0,1,2,7,8,17,18,23,24)],
                         ['none','none','p','p','P','P','p','p','P'])

    def test_windows(self):
        self.assertEqual(classify(29739),'crash')
        self.assertEqual(classify(29745-0),'ok')
        self.assertEqual(classify(29763-60),'ok')          # switch 2 red: no crash on NTSC
        self.assertEqual(classify(29763-383),'graphics')   # switch 6 red
        self.assertEqual(classify(30407-562),'confetti_limited')
        self.assertEqual(classify(31329-562),'confetti_infinite')
        self.assertEqual(classify(31757-562,next_box=False),'line_lag')
        self.assertEqual(classify(30407-562,next_box=False),'ok')   # no limited confetti without next box

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            crash_outcomes(249,3150,0,rows_high=0,pushdown_raw=0,clear_count_carry=False)
        with self.assertRaises(ValueError):
            crash_outcomes(170,1700,2,rows_high=3,pushdown_raw=0,clear_count_carry=False)


if __name__=='__main__':
    unittest.main()
