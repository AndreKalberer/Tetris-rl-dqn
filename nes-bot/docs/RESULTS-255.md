# Verified original-cartridge results

Every game below started at level 18 through the real menus with controller input only and was
replayed in a fresh MesenCE process from its recorded inputs, matching RAM, picture, clocks and
result exactly (`case_state: verified`). Seed = frames waited on the menu. Uncapped score =
`nes_original/theoretical_score.py` (the cartridge itself shows 999,999).

## Rebirth (level 255, wrap to 0, back to level 255)

| Bot | Seed | Absolute level | Lines | Uncapped score |
|---|---|---|---|---|
| v8b | 19211 | 511 | 6,651 | 43,047,996 |

## Score bot (level 255, score per lap)

| Bot | Seed | Lines | Uncapped score | Tetrises |
|---|---|---|---|---|
| score50 (`run-score50.sh`) | 957 | 3,290 | 58,187,903 | 185 |
| score50 (`run-score50.sh`) | 24340 | 3,290 | 55,723,090 | 190 |
| score80 (`run-score80.sh`) | 957 | 3,290 | 79,973,267 | 328 |
| score80 (`run-score80.sh`) | 15868 | 3,291 | 80,891,827 | 328 |

All played headless on a Linux cloud machine; the uncapped score matched the real counter at
every check (0 mismatches). A third score80 game (seed 24340) topped out at level 235 with
52,874,509; the simulator, given the same pieces, makes the same 6,458 placements and tops out
at the same point.

## Level 255

| Bot version | Seed | Lines | Uncapped score |
|---|---|---|---|
| crash-map-v5 | 129 | 3,290 | 22,219,561 |
| crash-map-v5 | 145 | 3,290 | 22,302,336 |
| crash-map-v5 | 177 | 3,290 | 22,479,627 |
| v7 | 49 | 3,290 | 22,374,551 |
| v7 | 193 | 3,290 | 22,529,980 |
| v7 | 209 | 3,290 | 22,454,000 |
| v7 | 225 | 3,290 | 22,453,239 |
| v7-test25 | 10329 | 3,290 | 22,128,474 |
| v7-test25 | 19211 | 3,290 | 22,247,779 |
| v7-test25 | 26300 | 3,290 | 22,526,987 |
| v7-test25 | 27885 | 3,290 | 22,252,154 |
| v7-test25 | 30418 | 3,290 | 22,330,550 |
| v8 | 13858 | 3,290 | 22,374,633 |
| v8b | 8600 | 3,290 | 22,311,367 |
| v8b | 10298 | 3,290 | 22,301,265 |
| v8b | 29716 | 3,290 | 22,505,255 |
| v8b | 30189 | 3,290 | 22,405,322 |
| v8b | 30804 | 3,290 | 22,127,456 |
| v8c | 5700 | 3,290 | 22,380,135 |
| v8c | 17452 | 3,290 | 22,369,620 |

20 games; uncapped score mean 22,353,715 (range 22,127,456–22,529,980).

## Where games were lost (versions v5 and later)

Most losses were top-outs at levels 157–158, where every single carries at least 25% crash
risk; earlier versions also lost games at 169–170 (singles only) and to pushdown crashes at 249.
Two v8 games lost to a one-off 2.4 s compile stall at level 156 (fixed in v8c). Run folders
(`runs/`, not in git) hold every case, including failures, with their verification status.
