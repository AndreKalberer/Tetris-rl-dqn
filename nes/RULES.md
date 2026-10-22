# Nintendo NTSC A-Type: provisional research engine

This is an independent mechanics implementation, not a ROM or an emulator.
The target is Nintendo's USA/NTSC NES Tetris A-Type. Whole-game emulator parity
has not yet been demonstrated. No NES RL policy has been trained.

## Sources and reproducibility

- Original reverse-engineering analysis: [MeatFighter](https://meatfighter.com/nintendotetrisai/).
- Public annotated disassembly: [CelestialAmber/TetrisNESDisasm](https://github.com/CelestialAmber/TetrisNESDisasm/tree/94581c0306fa0009fe11fde48222875549d15b8b),
  revision `94581c0306fa0009fe11fde48222875549d15b8b`.
- The code here is independently written. Only factual geometry and mechanical
  constants are represented; no ROM, disassembled game program or Nintendo art
  is distributed in this project.
- The optional `tools/verify_nes_reference.py` reads SHA-256-pinned reference
  source files supplied separately. It assembles selected routines in memory
  using py65 1.2.0 and compares the results with this engine. It does not build
  a game ROM. The retained report identifies source/engine/verifier hashes.

## Implemented mechanics

- Board: 10x20 visible cells, active-piece coordinates may extend to y=-2.
  Spawn pivot is (5,0). Occupancy above the visible board is not persisted.
- Piece order and IDs are NES-specific: T/J/Z/O/S/L/I. The 19 Nintendo
  orientations are stored explicitly. I/S/Z have two orientations; O has one.
  Clockwise and counterclockwise use the Nintendo lookup tables. No kicks.
- Controller state is a held Left/Right/Down/A/B bitmask. A rotates clockwise,
  B counterclockwise. Rotations require a new press; A has priority when both
  rotation buttons are newly pressed. There is no hold, hard drop, or 180 turn.
- Processing order in an active frame is horizontal shift, rotation, then drop.
  Shifting and rotating may therefore occur in the same gravity frame.
- Horizontal DAS: immediate shift on a new press, 16-frame initial repeat delay,
  then six frames per repeated shift. Blocked shifts set the charge to 16.
  Charge persists through release and entry; it does not accumulate during entry.
  Down suppresses horizontal movement. Engine internals reproduce Right priority
  for simultaneous Left/Right; the UI and probe disallow that combination.
- Soft drop: the Down press arms the counter, first forced drop on the third
  held frame, then one every two frames. Gravity and soft drop do not move two
  rows in one frame. The initial 96-frame hanging-piece delay can be cancelled
  with a new Down press. Horizontal/rotation controls work during that delay.
- Clock: nominal NTSC 60.0988 frames/sec. Gravity intervals by level:
  0..9: 48/43/38/33/28/23/18/13/8/6 frames per row;
  10..12: 5; 13..15: 4; 16..18: 3; 19..28: 2; 29+: 1.
- Lock begins on a failed downward attempt. Merely touching the stack does not
  start a modern resettable timer. No input can rescue a piece once lock state
  has begun. Game over is checked in lock state, as in the reference routine.
- Entry is modeled through lock, four row scans, clear animation, score/goal
  states and spawn. A simulated display-copy cursor advances four rows/frame.
  This reproduces the source-derived 10..18-frame non-clearing entry delay.
  Five line-clear animation passes on four-frame boundaries add 17..20 frames.
  Lines/score become visible at the score state, after lock and scanning.
- Scoring uses 40/100/300/1200 times level+1, using the level after the transition.
  Packed-BCD addition and the documented soft-drop scoring quirk are reproduced;
  a continuous final soft drop is required for drop credit. Score caps at 999999.
- Starting levels: 0..19. The first transition is at
  min(start*10+10, max(100,start*10-50)) lines, then every ten additional lines.

## Randomizer and observations

The 16-bit LFSR starts from boot state 0x8988, advances every simulated frame,
and may advance once more for a piece-selection reroll. Piece selection uses
the incrementing byte counter and the previous **orientation ID**, including
the original reroll behavior. This is not a seven-bag or a memoryless uniform
randomizer. Changing input timing can change future pieces.

The experiment `seed` is a number of LFSR warmup frames modulo 32767. Zero uses
the boot state; this is a reproducible simulation convention, not a claim about
menu timing or seeds in an actual ROM. Current and preview pieces use the
source initialization sequence. Observations expose only the locked board,
current pose/control counters, one next piece and visible statistics. The
randomizer state is omitted. Full JSON save states retain it for exact resume.

## Current fidelity limits

- Whole-ROM/controller-trace comparison is pending; no ROM is available in this
  workspace. Source-level routine checks cannot establish complete compatibility.
- Hidden-row memory wrapping, clears at row zero, and late-game overflow/crash
  glitches are not fully emulated. Above-board locked cells are truncated and
  row-zero clears use safe ordinary compaction. These are explicit differences.
- CPU instruction timing, PPU/audio rendering cost and lag frames are not modeled.
  Menus, music, ending animations, pause behavior and controller polling hardware
  are outside this engine. The manual UI pauses the simulation clock.
- Level progression and line totals extend mathematically beyond the ordinary
  range instead of reproducing all byte/BCD overflow behavior. Research results
  beyond level 29 require a separately stated validation scope.

## Stacking probe and evaluation scope

The supplied controller is a fixed, untrained geometric heuristic. It proposes
vertical placements, then tries to reach them through real held buttons and DAS.
Proposals are not guaranteed reachable and the engine never teleports a piece.
It does not use the modern learned policy or inspect the randomizer.

Planning latency is recorded but currently not charged as game frames. Thus all
probe scores are small development samples, not human-comparable results or
claims about the original cartridge. Fix a reaction/latency and button-rate
protocol before serious policy comparisons.

Each probe stores button-state changes, full-state hashes at every lock and a
final-state hash. Replay reconstructs the complete game including RNG/counters.
A resource cutoff is unfinished, not a natural top-out. Retain all seeds,
including failures. The initial draft's J-piece table error and superseded
probes remain documented in progress history; corrected probes use a new folder.
