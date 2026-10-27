# Offline Sprint v1

This is a self-contained offline falling-block game, with modern-style controls.
It is not an official Tetris product or a certification of the proprietary Tetris
Guideline. Its rules are frozen for the acceptance benchmark.

- Board: 10 columns, 20 visible rows, 4 hidden spawn rows.
- Randomizer: seeded seven-bag; five upcoming pieces are visible. Hold is allowed
  once per placement and resets the active piece's rotation and spawn position.
- Rotations: clockwise/counterclockwise SRS-style rotations, with separate I and
  JLSTZ kick tests in fixed order. O geometry does not rotate or reset lock delay.
  No 180-degree rotation. Matrix coordinates increase downward; kick y increases
  upward. Mechanical conventions were cross-checked against the primary
  [Cold Clear rotation implementation](https://github.com/MinusKelvin/cold-clear/blob/master/libtetris/src/piece.rs).
  The game and controller code here were written independently.
- Clock: 60 fixed simulation frames/sec. Gravity moves one row every 30 frames
  (2 rows/sec), with no speed increase during the 40-line sprint.
- Lock delay: 30 grounded frames, with at most 15 resets by successful grounded
  movement/rotation. Leaving the ground clears the grounded timer. Hard drop
  locks immediately at the lowest unobstructed vertical position from the
  **current reachable pose**. Soft drop moves one row per input.
- Top-out: spawn overlap, or locked cells remaining in the hidden buffer after
  line clearing. This strict hidden-row rule is an explicit local choice.
- Goal: clear at least 40 lines. There is no versus garbage, attack scoring,
  T-spin bonus, line-clear animation delay, or official-client integration.
- Bot: waits 30 frames per placement to plan, emits at most one primitive input
  every 3 frames (20 inputs/sec), and never sets the engine's board or pose.
  Search is over reachable paths under the same gravity/lock rules. Search may
  omit timing-dependent tucks; it expands at most 12 horizontal/rotation commands
  before hard drop and does not search soft-drop tucks. All returned paths must
  replay legally. A planner
  taking longer than its 0.5-second budget invalidates that acceptance trial.
- Evaluation safety limits: 36,000 simulated frames (10 game minutes) or 120 wall
  seconds per trial. A cutoff is **inconclusive**, not a win or a natural loss.
  Passing requires 40 lines, zero rejected bot inputs, zero predicted/actual board
  mismatches, zero planner overruns, and an exact primitive-input replay.

Development seeds are 1000–1009. The reserved acceptance set is 91000–91019 and
will be used only after implementation and difficulty are frozen. V1 requires at
least 18 passes out of all 20; inconclusive trials do not count as passes.
