# Faster legal controls and visible-next search

The first development clear of level29 reached240lines/level30 from18,
score230640,602placements. Whole RAM and RGB replay exactly. Six additional
recorded idle frames settle the next spawn. This is a search/controller result
in nes-py, not an RL gain, reserved acceptance, or hardware-fidelity certificate.
The original HUD reads00 at rawlevel30 because its display table has only30
entries. The game has not undergone255rebirth at that point.

The old held-direction action family is restrictive at one-row-per-frame
gravity. A provisional-model empty-board check, excluding initial grace,
covers only columns2..7 at29; alternating press/release every2frames covers
all10. The new native controller sends those ordinary button states, retains
3reaction frames/48navigation/4frame rotation pulses, and leaves all actual
collision/gravity/locking/clears to the original game. No hold, hard drop,
board writes, injected RNG or future-piece queue is used.

Search evaluates current candidates and the ONE visible next piece, with a
fixed beam. Optional CPU compilation makes it practical to expand all current
targets. Future boards/timers are approximate features; actual native outcomes,
accounting, target/prediction misses and exact replays remain audited. Unknown
next-next context is a fixed unused dummy, never read from the emulator.
Compilation warms before gameplay; over-budget runtime proposals are still
discarded and elapsed frames are charged as idle before replanning.

```powershell
# Existing original-game environment plus optional CPU dependencies:
.\.venv-original\Scripts\python.exe -m pip install -r requirements-nes-lookahead.txt
.\.venv-original\Scripts\python.exe -m nes_original.lookahead --tap-period 2 --beam 34 --jit --seeds 17 33 --output runs/new-development
```

During live training, dependencies were instead installed into the separate
scratch target `../nes-lookahead-deps`, without altering the worker environment.
Pass `--jit-deps ../nes-lookahead-deps` in that arrangement.

Five paired DEVELOPMENT settings were tried on menu waits17/33. Retained raw
results are in `evidence/nes-level29`: held one-ply103/44lines, held beam3
lookahead69/130, tap one-ply230/175, tap full lookahead240/212, and tap full
lookahead with squared line reward78/240. There are failures in every setting;
this does not establish18/20 reliability or a reliable score increase. Different
legal input timing changes later pieces in the original time-dependent RNG.
All games replay exactly; late CPU plans are recorded, not silently ignored.

The CPU kernel was checked against3000 randomized reference control frames and
120 future-action cases over boards, shapes, speeds and pulse profiles. These
are implementation checks, not whole-ROM/hardware parity. Exact8-bit original
level progression is used for search; the old linear simulator helper cannot
represent the810-line level235, later cycles or255->0.

## Stronger-prior learning stage

The weaker held-input three-seed study remains preserved. New learning can use
the rapid action family and full lookahead as its fixed prior. A zero residual
initially reproduces that prior; RL must improve on the matched search baseline
before a learning gain can be claimed. Features remain288wide, but the configured
prior is part of policy/checkpoint metadata. New exports are version3; legacy
version2 loads with its original held/heuristic defaults.

```powershell
.\.venv-original\Scripts\python.exe -m nes_original.learn --prior lookahead --tap-period 2 --seeds 0 1 2 --steps 10000 --output runs/rapid-lookahead-study --status runs/rapid-lookahead-study/live.json
```

This stage collects1024teacher-chosen real transitions before updates, uses
epsilon.01->.002 and explores the top4 fixed-prior targets. It retains exact
full-resume checkpoints and selects only on declared validation starts1001/1013;
those remain development validation, not untouched final tests. New prior/action
mode checkpoint continuation is tested through original-game input/RAM/RGB and
optimizer weights. The full reliability/500k goal remains unachieved.

The current nes-py core has a demonstrated CPU-JAM fidelity defect. Native
development results must retain that scope; use the independent MesenCE path
and late-game audit before any hardware-faithful crash/rebirth claim.
