# Original Nintendo NTSC A-Type environment

This runs the original game program in **nes-py 9.0.1**, using the Nintendo USA
image shipped inside the installed **gym-tetris 4.0.0** dependency. It does not
use gym-tetris's TetrisEnv reset/menu helpers, which write the game's RNG RAM.
The independent provisional simulator and its one-seed learning pilot remain
under `nes/`; no simulator score becomes an original-game score.

The separate [MesenCE reference adapter](../docs/NES-ORIGINAL-REFERENCE.md) now
supports gameplay, training, evaluation and exact replay. A development run
cleared29 from18:240lines/raw30,255566score,608placements. The first nes-py
clear is retained separately. Neither establishes18/20 reliability or RL gains.
An authored CPU JAM diagnostic found a nes-py fidelity gap relevant to original
late-game crashes; do not treat that core as certified for255/rebirth.

## Setup, run and replay

On Windows with CPython 3.13 and a working bootstrap pip:

```powershell
.\setup-nes-original.ps1
.\.venv-original\Scripts\python.exe -m nes_original.run --level 18 --seed 17 --output runs/original-probe
.\.venv-original\Scripts\python.exe -m nes_original.replay runs/original-probe/replay.json
```

Setup creates a separate `.venv-original`; it does not change `.venv` or any
running bot. If desired, pass `-EnvironmentPath` to reuse another isolated
environment. The installation used during development is outside the repository
at `../nes-original-env/.venv`. The full tested dependency versions are pinned
in `requirements-nes-original-lock-windows.txt`. No separate ROM download is
needed for the verified installed package. No game image is committed here.

The run is headless. To export its first 1,800 original RGB frames as an H.264
video, install ffmpeg on PATH and add `--video demo.mp4` to the replay command.
The entire saved run is still replayed and verified even if the video is shorter.
Interactive original-game human play remains unfinished. Original-game
training is described in [TRAINING.md](TRAINING.md).

## Instrumentation and reproducibility

`OriginalGame` uses raw `nes_py.NESEnv`, enforces game SHA-256
`acab10f2dbad4d4ef5d56b31cd4cea208bb6e2e2c5dba5eb339d711ec04e3535`,
and leaves RAM read-only on the Python side. It boots the game, selects A-Type,
navigates the original menu, and holds A before pressing Start for levels 10–19.
Menu starting levels 0/9/10/18/19 have been tested. Every native step applies a
controller byte and advances the emulator; gameplay frames are never skipped.

The seed is the number of genuine idle frames on the level menu, 0–32,766.
It does not set the original RNG directly and is not a uniform sample of RNG
states. Policies with different timing naturally see different later pieces.
Reset recreates the emulator for deterministic power-on: a native soft reset
was found to retain prior RAM/PPU history. A compulsory idle frame after the
initial active flag is recorded and counted in the episode, not hidden.

The agent receives the 20x10 locked occupancy grid, current/next orientations,
piece/control timers, phase, level, lines, score and its own pending input.
This is a declared **RAM-state interface**, not a vision-only human interface.
It excludes RNG, future pieces, CPU state and raw RAM. Placements count real
lock-to-scan transitions; the failed top-out piece is excluded. The lines high
byte is binary hundreds, not a second packed-decimal byte.

Snapshots are native full-state objects for in-memory branches. They have no
supported portable serialization in this nes-py build. Portable saves instead
record the entire boot/controller trace, dependency versions and final RAM/RGB
SHA-256 hashes. Restoring recreates the game and replays inputs, then checks
both hashes and metrics. This costs the elapsed episode length. It does not
pretend to serialize a native pointer. Cross-instance continuation is tested.

## Controller and remaining fidelity gaps

The current controller is a **heuristic baseline**. Fast provisional mechanics
propose reachable paths from visible state. The original game alone advances
the actual board and scores. There is one input frame in flight at steady
native frame boundaries; its known button state is included before planning.
Candidates use three reaction frames, held DAS, four-frame rotation pulses,
48 navigation frames, then Down. Left+Right and menu buttons are rejected.

RAM sampled at a PPU frame boundary can straddle the game's main-loop counter
updates. A longer original-game trace exposed timer/lock-frame differences
despite earlier short differential tests passing. Runtime therefore waits for
**observed** lock, issuing at most 16 additional Down frames when necessary.
It records timing adjustments and timer differences, and still checks actual
locked pose, post-clear board, lines and termination against the proposal.
Semantic mismatches stop the probe and are reported as errors. This is tested
against the previously failing trajectory, not silently treated as parity.

Planning over 3/60.0988 seconds discards the stale proposal, advances the real
game with idle controls for the full elapsed frame count, and then replans.
`--fixed-reaction` disables runtime timing charges for deterministic engineering
comparisons; it must be labeled accordingly. Limits remain censored outcomes.

One development game, level 18 / menu wait 17, naturally topped out at **175
placements, 60 lines, score 52,597**, still on level 18. All final RAM and RGB
bytes replayed exactly, with zero board/pose mismatches and zero planning
overruns; 145 drop-timing adjustments and 65 lock-timer differences were logged.
This is one untrained heuristic probe, not a reserved benchmark or RL gain.

Ten isolated-environment tests cover genuine menus, deterministic power-on,
native snapshots, portable continuation, original controller/geometry checks at
three speeds, natural top-out/statistics, read-only RAM and latency accounting.
These establish useful instrumentation, not cycle-accurate physical hardware
certification. Full late-game/level-29 behavior, broader states and an independent
emulator comparison remain unvalidated. Actual-emulator training and exact trainer
resume now have tests. Longer smoke runs exposed additional torn RAM samples
that can mix the new active phase with the old piece orientation. The separate
feedback learning controller waits for stable spawn observations; it treats
proposal boards as approximate features, logs target misses/prediction errors,
and audits actual native board accounting and exact episode replay. The strict
preview-audit controller above remains available with its checks intact.
Three-seed learning and level-29/500k acceptance results are still pending.

Upstream sources: https://github.com/Kautenja/nes-py and
https://github.com/Kautenja/gym-tetris. Inspected wheel hashes, smoke findings,
the failed strict-timing probe and corrected original-game trace are retained in
`evidence/nes-original/`. Prior negative simulator results remain unchanged.
