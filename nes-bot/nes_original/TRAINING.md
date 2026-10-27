# Original-game level-29 training

Current milestone: start original NTSC A-Type at level18, clear level29 and
reach level30 (240lines). First establish a replay-verified development success;
then retain the larger reliability/score/three-training-seed acceptance study.

```powershell
.\nes-bot\setup-nes-original.ps1 -Learning
.\.venv-original\Scripts\python.exe -m nes_original.learn --output runs/original-training --status runs/original-training/live.json --seeds 0 1 2 --steps 10000
```

The first local stage has three independent seeds run sequentially, with two
CPU threads, Windows affinitymask160 (logical5/7 on this16-processor laptop)
and BelowNormal priority. Modern worker logical2/4 are separate. Each seed has
10k actual decision transitions including256 teacher-chosen rollout decisions.
No simulator-trained policy is loaded. Budgets are engineering stages, not wins.

## Native feedback action interface

The original game always supplies actual physics, board, score, death and next
observations. No RNG/ROM/RAM edits. A fast provisional control model proposes
candidate target poses and approximate after-board features. Those predictions
are explicitly not asserted to be exact transitions in this learning interface.
The existing strict `nes_original.run.Controller` and its preview checks remain
separate and unchanged in meaning; longer strict failures remain saved.

A chosen action asks a native feedback controller to try the nominal target
column/orientation: three reaction frames, at most48 navigation frames, horizontal
changes no faster than once every6 native frames, rotation pulses every4 frames,
then Down until observed lock. Input states and rotations are ordinary controller
buttons. Actual missed targets and differing predicted/actual boards are logged,
not rewritten or labeled as the predicted state.

Saved player RAM can expose the new active phase before the new orientation is
copied. Entry waits for matching consecutive active orientations before the next
decision. A cross-piece rotation is rejected instead of entering an unbounded
rotation search; a forced drop has no rotation target. All idle/settle frames
are actual native frames in the replay.

Actual post-entry occupancy must equal previous occupancy +4 per locked piece
-10 per cleared line. Original above-board memory writes are counted separately
as top-boundary events, where visible four-cell conservation does not apply.
All completed training episodes are reconstructed in another native instance
and checked against whole-RAM and RGB hashes. Validation episodes do the same.
Native accounting failures stop the worker and retain failure inputs; they are
never counted as ordinary deaths or fed as fabricated experience.

## Learner and selection

The encoding has **288 features**:267 prior simulator-derived visible
proposal features (including200 predicted occupancy cells),20 target-pose fields,
and a clipped fixed heuristic prior. The residual MLP has128/64 hidden units;
zero output initialization initially reproduces the fixed clipped prior.
Double DQN uses replay3000, batch64, update every4 decisions, target sync500,
Adam5e-5, gamma.99, gradient clip1, target clip[-20,20], epsilon.05->.01 over10k.
The warm rollout is off-policy native experience with real TD rewards, not
supervised heuristic labels. Teacher decisions and optimizer updates are recorded.

Reward is0.02 +0.1*lines^2 +actual_score_delta/100000 -5*death
+5*completing29, plus gamma*Phi(next)-Phi(before). Phi is
-0.02*column_height_sum -0.25*holes -0.02*bumpiness, zero on terminal/goal.
This is experimental shaping, with no assumed RL improvement.

Menu-wait training starts are0..511, all level18. Validation starts1001/1013
are declared before running and never described as reserved final tests. Every
2500 decisions, retain policy and full checkpoints; choose by validation goal
successes, then median lines, then mean score. Fixed logical training/selection
timing isolates policy from host jitter. The selected policy is additionally
tested with runtime planning latency charged as idle before replanning.
The heuristic gets the same native controller, observations and clipped prior.

Full checkpoints preserve model/target/optimizer/replay, random states, metrics,
and original-game input journal. Exact next decisions, optimizer weights and
RAM/RGB after load are tested. Restore costs replaying the current episode.

```powershell
.\.venv-original\Scripts\python.exe -m nes_original.learn --resume runs/original-training/seed0/latest.pt --output runs/original-training --status runs/original-training/live.json --seeds 0 --steps 20000
```

Only load trusted full checkpoints. The weights-only exports have a distinct
original-game kind and reject previous simulator/modern policy formats. A STOP
file in the job directory requests a clean checkpointed stop between decisions.
The worker records status independently of progress.md to avoid concurrent edits.
