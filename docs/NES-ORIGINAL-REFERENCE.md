# Original Nintendo NTSC A-Type: reference core

The current finish line is start18, clear29 and reach30: 240 lines. A first
development clear exists in nes-py; reliable performance, 500k median score,
learned improvement and late-game/rebirth fidelity remain unproved.

`nes_original.env.OriginalGame` uses nes-py9.0.1. It runs the original program,
drives the real menus and does not use gym-tetris's RNG-writing reset helpers.
However, an authored diagnostic found that nes-py executes instructions after
CPU JAM `$02`. Its fixed-cycle step is also not a general video-frame contract.
Keep its results labeled by core. Do not certify original late-game crashes or
255->0 rebirth using this backend.

`nes_original.mesen.MesenGame` adds a separate Windows MesenCE2.2.1 reference
path. Genuine inputs, read-only RAM instrumentation, no game patches, normal
gravity. Seed means real menu idle frames; later pieces depend on input timing.
An observation exposes the board/current piece/single visible preview/control
counters, never RNG, CPU state, complete RAM or hidden future pieces. Diagnostic
RAM and CPU/PPU clocks are kept outside the policy interface.

## Setup

Use an isolated CPython3.13 Windows environment. Install
`requirements-nes-search.txt` there; do not modify a running worker's environment.
The optional compiled planner is an approximate feature generator, not an emulator.

Use a dedicated portable [official MesenCE2.2.1 build](https://github.com/nesdev-org/MesenCE/releases/tag/2.2.1).
The executable directory must contain `settings.json` with `{}`, its initialized
`MesenCore.dll` and `MesenNesDB.txt`. Do not reuse a customized emulator profile.
The launcher neither downloads nor installs an emulator. It hashes these files
and the Lua bridge and explicitly freezes NTSC, zeroed power-on RAM, CPU/PPU
alignment, no extra scanlines, sprite-limit changes, HD packs or runahead.

Use the existing installed USA image in place, SHA256
`acab10f2dbad4d4ef5d56b31cd4cea208bb6e2e2c5dba5eb339d711ec04e3535`.
No Nintendo game image is committed or included in project archives.

## Play, replay and train

Commands below assume that isolated environment's Python is active. Replace
`C:/path/Mesen.exe` with the dedicated portable executable. Always use fresh
output directories. On other machines, choose appropriate CPU affinities; this
workstation reserves logical2/4 for the separate modern experiment,5/7 for the
learner,8/10 for planning and12/14 for the reference child process.

```powershell
python -m nes_original.lookahead --mesen C:/path/Mesen.exe --beam 34 --tap-period 2 --jit --seeds 17 --output runs/reference-development
python -m nes_original.replay runs/reference-development/seed17/replay.json --mesen C:/path/Mesen.exe
python -m nes_original.learn --mesen C:/path/Mesen.exe --prior lookahead --tap-period 2 --seeds 0 --steps 10000 --output runs/reference-training --status runs/reference-training-status.json
python -m nes_original.learn --mesen C:/path/Mesen.exe --prior lookahead --tap-period 2 --seeds 0 --steps 20000 --resume runs/reference-training/seed0/latest.pt --output runs/reference-training --status runs/reference-training-status.json
python -m nes_original.compare --mesen C:/path/Mesen.exe --job runs/reference-training --seeds 1601 1613 1627 --output runs/reference-fresh-comparison
```

`--jit-deps` can point at an isolated optional dependency target instead of
installing Numba into the active environment. Weight-only policies can be tested
in another backend as a separate transfer experiment. Full trainer continuation
must use the same core/profile/bridge identity; mismatches fail. Training parks
its own core during independent validation and reconstructs it afterward, so
the reference host-idle watchdog does not kill an unattended training instance.

The strong fixed prior combines two-frame horizontal pulses with full search
over current candidates and the one known next piece. It is search, not RL.
The residual Double DQN must beat that same prior/controller on fresh starts to
establish a learning gain. Learning starts after1024 teacher decisions. Retain
failed policies, targets, board discrepancies, selection starts and protocols.

## Timing, proof and limits

Mesen's input-polled boundary is at PPU241:0, with measured zero pending input.
The planner honors `input_pipeline=0`; nes-py's measured pipeline is1. Both keep
three reaction frames,48 navigation frames, two-frame horizontal pulses and
four-frame rotation pulses in this experiment. Late proposals are discarded and
their elapsed idle frames charged in runtime evaluation. No hold or hard drop.

Each completed run reconstructs boot and every input in a fresh core. Reference
proofs compare all2KiB RAM, RGB, CPU cycles/PC and PPU/frame clocks. Core identities
are frozen before replay. Crashes/watchdog faults are distinct from top-outs;
never count a timeout or unfinished game as a successful clear.

Native binary snapshots are not yet exposed by this adapter; portable saves
reconstruct inputs from boot. Named pipes are the default, with a tested file
fallback. Host idle and total-process deadlines kill only the owned child.
Default bounds are30seconds per host action and onehour per process; choose
larger explicit bounds for endurance work. This is not whole-hardware certification.

No legal uninterrupted255->0 game or late-game crash matrix has been validated.
Level235 lasts810lines, and the next-cycle schedule differs. Raw lines wrap at
25600; use `MarathonProgress` at settled boundaries before adding long-run
rewards/accounting. At rawlevel30 the original HUD already displays00; this is
a display-table issue, not rebirth. Current training/evaluation deliberately
stops at30, so it does not exercise those later limits.

Opt-in reference tests:

```powershell
$env:NES_MESEN_EXE = 'C:/path/Mesen.exe'
python -m unittest discover -s tests -p 'test_mesen*.py'
```
