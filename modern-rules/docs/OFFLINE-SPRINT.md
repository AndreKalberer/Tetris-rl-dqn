# Offline Sprint: modern-rules game, learned placement policy and lookahead study

This is the project's first part: a self-contained, playable offline falling-block game
with modern-style rules, a trained board-scoring neural policy, reachable-move planning
and reproducible evaluation. It is independent of the NES work; none of its results
transfer to the original NES game.

The snapshot taken before NES work started is described in [CHECKPOINT.md](CHECKPOINT.md),
including retained results and how to restore models and saved games.

## Contents

- [Quick start](#quick-start-windows-python-313)
- [Offline game and acceptance result](#offline-game-and-acceptance-result)
- [Research question and findings](#research-question-and-findings)
- [Offline endurance](#offline-endurance-optional-extension)
- [NES-bot port and head-to-head](#nes-bot-port-and-head-to-head-optional-extension)
- [Train and resume](#train-and-resume)
- [Code map and limits](#code-map-and-limits)
- [Legacy placement-DQN scripts](#legacy-placement-dqn-scripts-stable-baselines3)


The offline bot uses **legal timed inputs**: move, rotate, hold, and hard drop.
Gravity and lock timing remain active. The older placement-action training
sandbox is also included, and its scores are kept separate.

## Quick start (Windows, Python 3.13)

From the extracted project directory:

```powershell
.\modern-rules\setup.ps1
.\modern-rules\play.ps1
```

Setup creates `.venv`, installs pinned CPU dependencies, and runs the tests. The
release includes trained weights; training is not required to watch or play.
Installation needs package downloads once; gameplay runs offline afterward.
The tested full dependency lock is `modern-rules/requirements-lock-cpu-windows.txt`.

Human play and reproducible evaluation:

```powershell
.\modern-rules\play.ps1 -Human
.\modern-rules\evaluate.ps1
```

If scripts are restricted in your environment, the equivalent commands are:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r modern-rules/requirements-dev.txt
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
strict hidden-row top-out behavior are in [offline/RULES.md](../offline/RULES.md).
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

The default learned model is `old-dqn/models/seed1-100000.pt`. Choose another retained
model or disable extra lookahead with:

```powershell
python -m offline.play --checkpoint old-dqn/models/seed2-60000.pt --beam 1
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
.\old-dqn\research-evaluate.ps1
```

Or run an individual research comparison:

```powershell
python -m dqn.benchmark_lookahead --checkpoint old-dqn/models/seed1-60000.pt --seeds 70000 70001 70002 70003 70004 --cap 5000 --beams 1 4 --output runs/research-repeat
```

An explicit research line-goal evaluator distinguishes a passed line goal,
natural loss, and inconclusive resource limit:

```powershell
python -m dqn.line_challenge --checkpoint old-dqn/models/seed1-100000.pt --seed 80003 --target-lines 10000 --output runs/line-goal
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

The retained seed-92000 game reached **38,666 placements and 15,464 lines**, still
alive at its wall-time cutoff (7.4 simulated hours). Its full primitive-input and
locked-board replay verified, with zero rejected inputs, board mismatches, or
planning overruns. This is one unfinished game, not an average or maximum. An
initial reporting error was fixed and the exact saved game resumed. The first
and last 1,000 placements had average tallest columns of 5.00 and 5.18 cells.
The checkpoint backup preserves both segments and their original protocols.

This uses the same gravity, lock delay, input rate, and trained policy, with the
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

## NES-bot port and head-to-head (optional extension)

`offline/classic_bot.py` ports the NES bot's hand-weighted evaluator to this game:
the NES Dellacherie features (`nes_original.sim.board_features`) plus a 3-piece
beam search (beam 8) over hold and the visible preview. The executed move uses the
same timed reachable paths and controller timing as the learned bot; deeper plies
use a cheaper rotate-shift-drop family. No network or training is needed.

```powershell
python -m offline.compare_bots --mode sprint --seeds 91000-91019 --output runs/h2h-sprint
python -m offline.compare_bots --mode endurance --cap-lines 5000 --seeds 92000-92007 --output runs/h2h-end
```

Results on a 4-vCPU Linux cloud machine (2026-10-26), one single-threaded game per core:

| | Learned bot (seed1-100k, beam 4) | NES-bot port |
|---|---|---|
| 40-line sprint, seeds 91000-91019 | 20/20, all replays verified | 20/20, all replays verified |
| Pieces / game time per sprint | ~108 / ~74 s | ~101 / ~68.6 s |
| Endurance, seeds 92000-92007, 5,000-line cap | 8/8 alive at cap | 8/8 alive at cap |
| Seed 92000, 50,000-line cap | alive (125,010 pieces) | alive (125,001 pieces) |
| Planning time per move | ~28 ms | ~26 ms |

Both bots also stayed alive to a 1,000-line cap with gravity 4x and 7.5x faster.
Endurance games are checked piece by piece against the planned board (zero
mismatches or rejected inputs) but were not input-replayed. With lock delay and
hold, this ruleset does not separate the two bots on survival.

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

Legacy SB3 scripts remain under `old-dqn/dqn/`: `train_dqn_server.py`, `eval_lines.py`,
`diagnose_policy_dqn.py`, and `enjoy_local_dqn.py`. `old-dqn/play_bot.py` and `old-dqn/record_gif.py`
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


## Legacy placement-DQN scripts (Stable-Baselines3)

Before the current trainer, the project used Stable-Baselines3 DQN with a **macro-action**
wrapper (`old-dqn/tetris_macro_env.py`): one decision per placed piece — rotation, column and
optionally hold — instead of learning dozens of frame-level inputs. That keeps credit
assignment short (one step = one placement) at the cost of not modelling timing-dependent
moves such as tucks and spins.

The legacy `.zip` checkpoints are not included in the repository (`*.zip` is git-ignored),
so the old reported scores (for example 75.3 lines per episode) cannot be reproduced from
this repository and are historical, not controlled benchmarks. The GIF in
[`old-dqn/assets/tetris_bot.gif`](../../old-dqn/assets/tetris_bot.gif) is an illustrative playback, not an
evaluation sample.

To evaluate a legacy checkpoint you have, name it explicitly rather than relying on the
newest file in `models/` (relative to where you run it), and match the observation settings to its training run:

```powershell
python old-dqn/dqn/diagnose_policy_dqn.py --model-path models/YOUR_CHECKPOINT.zip --obs board --hold-actions --next-n 1 --reward-profile tetris --n-steps 20000 --seed 42 --summary
```

The diagnostic uses deterministic predictions and resets successive episodes with
`seed + episode_number`. Its fixed placement budget can stop mid-episode: `lines_per_ep`
and `steps_per_ep` average completed episodes only, and the unfinished episode is reported
separately (`unfinished_episode_lines` / `unfinished_episode_steps` without `--summary`).
`lines_total` includes it. If no episode finishes, there is no episode average. Treat it as a
policy diagnostic; a controlled comparison (random policy vs DQN vs other action
representations on the same seeds and rules, with sample counts and variation) has not been run.
When reporting a run, record the commit, checkpoint SHA-256, dependency versions, seeds,
completed episode count, total lines, invalid-action rate and reward profile.
