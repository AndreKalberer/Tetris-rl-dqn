# NES work: first engine milestone

The modern version remains frozen at `checkpoint/modern-offline-2026-10-21`.
This branch adds a separate, provisional Nintendo NTSC A-Type engine. See
[RULES.md](RULES.md) for exact mechanics, reference revision and fidelity gaps.

From the project root with the existing environment:

```powershell
.\nes-bot\play-nes.ps1
.\nes-bot\play-nes.ps1 -Bot -StartLevel 18
```

For a fresh checkout or the standalone engine ZIP, run `.\nes-bot\setup-nes.ps1` first.
It installs the small NES-specific requirements and tests, without PyTorch or
the modern training environment. Python 3.13 is the tested interpreter.

Equivalent Python commands:

```powershell
.\.venv\Scripts\python.exe -m nes.play
.\.venv\Scripts\python.exe -m nes.play --bot --start-level 18
.\.venv\Scripts\python.exe -m pytest -q
```

Human controls: arrows move/soft drop, X is A (clockwise), Z is B
(counterclockwise), P pauses, R restarts, N advances the seed, B toggles the
untrained probe, Esc quits. The display shows score, lines, level and current
gravity. No ROM, hold or hard drop is included. An experimental NES-specific
learned policy is available separately; see [TRAINING.md](TRAINING.md).

Watch it with `.\nes-bot\play-nes.ps1 -Checkpoint nes-bot/models/nes-seed0-best.pt -StartLevel 18`
after installing the full CPU dependencies with modern-rules/setup.ps1.

Run a small probe suite (choose a new output folder each time):

```powershell
.\.venv\Scripts\python.exe -m nes.evaluate --start-level 18 --seeds 0 1 2 --output runs/nes-repeat
```

Record a headless probe demo, with a verified input replay:

```powershell
.\.venv\Scripts\python.exe -m nes.play --start-level 18 --seed 1 --record runs/nes-demo.mp4 --screenshot runs/nes-demo.png
```

Snapshots are plain JSON. `Game.from_state(state)` restores the full game.
`nes.evaluate.replay(row)` verifies a saved episode's input and full-state hashes.
No checkpoint loading with pickle is needed for this new engine.

## Independent routine validation

Optional developer validation uses py65 separately from the normal runtime:

```powershell
.\.venv\Scripts\python.exe -m pip install -r nes-bot/requirements-nes-validation.txt
.\.venv\Scripts\python.exe nes-bot/tools/verify_nes_reference.py --reference-dir PATH_TO_REFERENCE --cases 3000 --output runs/nes-reference-check.json
```

Supply `main.asm` and `constants.asm` from the pinned public disassembly revision
linked in RULES.md. The verifier checks their hashes first. It compares all 19
geometries and the rotation/spawn/gravity tables, plus 3,000 piece draws and
3,000 active control states against selected 6502 routines assembled in memory.
It does not certify the complete frame scheduler, ROM, PPU or late-game behavior.

## Next milestones

1. Compare complete input traces with the original game in an emulator when
   available, especially startup/spawn, top-boundary behavior and frame timing.
2. Add timed reachable-action planning and freeze the controller/latency limits.
3. Broaden NES-specific learner training across seeds and investigate regressions.
4. Measure consistent level-19/29 reach rates, score, lines, and regressions on
   reserved starts across independent training seeds.

The untrained probe is an engine exercise, not completion of the NES project.
