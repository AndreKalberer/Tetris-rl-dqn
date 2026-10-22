# Tetris RL: Offline Sprint

A local Tetris research project with a playable offline game, a trained board-scoring
neural policy, reachable-move planning, and reproducible evaluation.

The checkpoint before NES work is documented in [CHECKPOINT.md](CHECKPOINT.md),
including retained results and instructions for restoring models and saved games.

Experimental NES work lives separately under [nes/](nes/README.md). Run
`.\play-nes.ps1` for human play or `.\play-nes.ps1 -Bot -StartLevel 18` for the
untrained stacking probe. This provisional engine has source-level mechanics
checks and an experimental [NES learner](nes/TRAINING.md). Whole-ROM parity is
still pending. An [original-game emulator adapter](nes_original/README.md) now
runs Nintendo USA/NTSC A-Type in nes-py, using genuine controller menus and
read-only RAM observations. It is separate from the simulator and its trained
policies. The modern checkpoint's results do not transfer to NES.

The offline bot uses **legal timed inputs**: move, rotate, hold, and hard drop.
Gravity and lock timing remain active. The older placement-action training
sandbox is also included, and its scores are kept separate.

## Quick start (Windows, Python 3.13)

From the extracted project directory:

```powershell
.\setup.ps1
.\play.ps1
```

Setup creates `.venv`, installs pinned CPU dependencies, and runs the tests. The
release includes trained weights; training is not required to watch or play.
Installation needs package downloads once; gameplay runs offline afterward.
The tested full dependency lock is `requirements-lock-cpu-windows.txt`.

Human play and reproducible evaluation:

```powershell
.\play.ps1 -Human
.\evaluate.ps1
```

If scripts are restricted in your environment, the equivalent commands are:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m offline.play
.\.venv\Scripts\python.exe -m offline.play --human
.\.venv\Scripts\python.exe -m pytest -q
```

Controls: arrows move/soft drop, Up or X rotates clockwise, Z rotates
counterclockwise, Space hard drops, C holds. B toggles bot/human, P pauses,
R restarts the seed, N starts the next seed, and Esc quits. The display shows
lines, pieces, game time, hold, and next pieces. It pauses after a suspend-sized
clock gap instead of fast-forwarding an unseen game.

## Offline game and acceptance result

**Offline Sprint v1** has a 10x20 visible board, four hidden spawn rows, seven-bag,
next five, hold, SRS-style kicks, 60 Hz physics, gravity at two rows/sec, a
0.5-second lock delay, and at most 15 grounded lock resets. Its exact rules and
strict hidden-row top-out behavior are in [offline/RULES.md](offline/RULES.md).
This is an independent offline ruleset, not an official Tetris product or a
certification of the proprietary Tetris Guideline. It does not control another
commercial game or an online service.

The v1 acceptance target was at least 18 successful 40-line games out of 20
reserved starts. The frozen policy/controller passed **20/20** seeds 91000–91019:

- Every game cleared at least 40 lines before a natural loss or resource cutoff.
- Zero rejected bot inputs, predicted/actual board mismatches, or planning overruns.
- Every game was verified by replaying only its frame-numbered primitive inputs.
- Median game time was **74.41 simulation seconds**, range **70.00–76.35**.
  Headless wall time is not the speed of a real-time game.

The bot waits 30 frames per placement and issues at most one input every three
frames. It cannot set the game board or active pose. The controller searches a
bounded subset of reachable horizontal/rotation paths, then hard drops from the
current reachable pose. It does not search every possible timing-dependent tuck.
Thinking longer than its 0.5-second budget makes a trial inconclusive.

The earlier unrestricted planner cleared a development game but exceeded its
thinking budget; it was rejected as inconclusive. The search was bounded before
reserved evaluation. Engine/controller/evaluator/rule hashes and the policy hash
were frozen in the acceptance protocol. Time/frame cutoffs never count as wins.
Raw per-input traces and protocols are included in release evidence.

Record a complete offline game without opening a window:

```powershell
python -m offline.play --seed 1001 --record runs/offline-demo.mp4 --screenshot runs/offline-final.png
```

The default learned model is `models/seed1-100000.pt`. Choose another retained
model or disable extra lookahead with:

```powershell
python -m offline.play --checkpoint models/seed2-60000.pt --beam 1
```

## Research question and findings

The focused question is: **how does one extra move of lookahead change a learned
placement policy's strength, consistency, and CPU cost?** This is a reproducible
engineering study, not a claim of a new algorithm or state-of-the-art Tetris.

The research agent scores resulting boards with a shared 128/64-hidden-unit MLP.
Inputs contain locked occupancy, structural features, held piece, and the visible
queue. Training uses Double DQN, legal-only exploration/targets, deduplicated
rotations, and packed replay. Search proposes the best four first placements,
then scores exact immediate reward plus discounted next-placement value. Newly
drawn hidden queue entries stay masked; this is approximate two-ply search.

The study retained three independently trained agents (seeds 0/1/2), with matched
60k snapshots and additional 100k snapshots for seeds 1/2. All five snapshots
were evaluated with/without search on the same five new streams 70000–70004,
at a 5,000-placement cap: 50 cases, all retained.

At matched 60k, one-ply survived to the cap in **10/15 model/stream pairs**;
beam-4 survived in **15/15**. Across all five snapshots, search survived all 25
capped cases. Seed1-100k also survived all five one-ply games, while seed2-100k
died at 389 lines on one stream that its 60k snapshot had survived. Longer
training was therefore not uniformly beneficial.

These are **censored outcomes**, not average lines until loss. The five streams
are reused across models; snapshots from one training seed are correlated.
The historical README score of 75.3 lines/game cannot be reproduced without its
old checkpoint, so no verified historical multiplier is claimed.

CPU profiling found preview/feature construction dominant. Batching preserved
exact model inputs/outcomes on saved states, randomized boards, real-engine
checks, and a complete reference trajectory. The same instrumented 200-move
trace fell from **7.20s to 2.09s**. That is a small profiling sample, not a universal
runtime guarantee. Concurrent benchmark jobs also affect wall-time measurements.

Reproduce the three matched-budget model comparisons:

```powershell
.\research-evaluate.ps1
```

Or run an individual research comparison:

```powershell
python -m dqn.benchmark_lookahead --checkpoint models/seed1-60000.pt --seeds 70000 70001 70002 70003 70004 --cap 5000 --beams 1 4 --output runs/research-repeat
```

An explicit research line-goal evaluator distinguishes a passed line goal,
natural loss, and inconclusive resource limit:

```powershell
python -m dqn.line_challenge --checkpoint models/seed1-100000.pt --seed 80003 --target-lines 10000 --output runs/line-goal
```

The earlier three 10k-line development trials were inconclusive after a large
elapsed-time interruption while still alive around 2,600 lines. They are not
passes or natural losses. The 10k-line research target remains an optional stretch;
it is separate from the completed offline-sprint acceptance.

## Offline endurance (optional extension)

The finished offline controller can be measured beyond the forty-line sprint:

```powershell
python -m offline.endurance --seed 92000 --beam 4 --wall-seconds 1800 --output runs/endurance
```

The retained seed-92000 game reached **38,666 placements and 15,464 lines**, stillalive at its wall-time cutoff (7.4 simulated hours). Its full primitive-input andlocked-board replay verified, with zero rejected inputs, board mismatches, orplanning overruns. This is one unfinished game, not an average or maximum. Aninitial reporting error was fixed and the exact saved game resumed. The firstand last 1,000 placements had average tallest columns of 5.00 and 5.18 cells.The checkpoint backup preserves both segments and their original protocols.This uses the same gravity, lock delay, input rate, and trained policy, with the
line-win condition disabled. There is **no piece, line, or simulation-frame cap**.
Natural top-out is a completed game. The wall-time watchdog, an explicit stop,
or an execution/controller error leaves the result unfinished, not a score until
loss. Reports separate simulated game time, elapsed wall time, and process CPU
seconds. One stream is an individual run, not an average or a global record.

Inputs and locked-board hashes are streamed. Verify or continue an unfinished
run without restarting its piece sequence:

```powershell
python -m offline.endurance --output runs/endurance --verify-only
python -m offline.endurance --resume runs/endurance/state.pkl --wall-seconds 1800 --output runs/endurance-continued
```

Resume files contain pickled local game/controller state; load only files you
trust. Previous segments are preserved and replayed in order. Use a fresh output
directory for each segment. Creating a `STOP` file inside the current output
directory requests a clean stop. This extension does not alter the frozen v1
release archive or its forty-line acceptance evidence.

## Train and resume

```powershell
python -m dqn.train_candidates --steps 20000 --seed 3 --output runs/new-agent
python -m dqn.train_candidates --resume runs/new-agent/latest.pt --steps 100000 --output runs/continued
```

Steps are absolute total placed-piece targets. Full checkpoints preserve online/
target weights, optimizer, replay, RNG, configuration, and the current research
game. Load full pickle-based checkpoints only from trusted sources. The small
`policy.pt` exports load with `weights_only=True`. Five policy snapshots and full
resumable checkpoints are included in the release.

Research training still uses the placement sandbox. The offline adapter filters
choices through timed reachable paths; it is not a newly trained frame-level
agent. Future estimates inside two-ply search remain approximate, while each
executed first-placement path is checked against the offline engine.

Legacy SB3 scripts remain under `dqn/`: `train_dqn_server.py`, `eval_lines.py`,
`diagnose_policy_dqn.py`, and `enjoy_local_dqn.py`. `play_bot.py` and `record_gif.py`
load legacy `.zip` models. Corrected observations and real next-5 input can change
old-checkpoint behavior. Metadata/replay checks guard resume; an explicitly chosen
`--allow-legacy-resume` is a warm start, not exact historical reproduction.

## Code map and limits

`offline/engine.py` owns physics; `offline/bot.py` plans and emits primitive
inputs; `offline/play.py` renders/manual-controls/records; `offline/evaluate.py`
runs and replays the acceptance protocol. `dqn/candidates.py` and `dqn/lookahead.py`
provide research previews/value guidance; `dqn/train_candidates.py` owns learning.

The project has automated tests covering research correctness, checkpoint continuation,
preview parity, resource-cutoff accounting, offline physics/kicks, and timed
path replay. Acceptance traces independently verify 20 full offline games.

Further work is optional: more offline difficulties and unseen streams, soft-drop
and tuck planning, addressing training regressions, broader survival estimates,
or an adapter for a particular other game. Multiplayer, official-client control,
and world-record strength are outside v1. No cloud service is required at runtime.

## License and references

Project code is MIT; see [LICENSE](LICENSE). Dependencies retain their own licenses.
Mechanical rotation conventions were checked against the author's
[Cold Clear implementation](https://github.com/MinusKelvin/cold-clear/blob/master/libtetris/src/piece.rs).
The offline engine/controller were implemented independently.
The Tetris research context is discussed in
[Gabillon, Ghavamzadeh & Scherrer (2013)](https://mohammadghavamzadeh.github.io/PUBLICATIONS/nips13-tetris.pdf).
Its rules and published scores are not interchangeable with this project's tests.
