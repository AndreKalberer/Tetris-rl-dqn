# First NES learning pilot

This is an experimental simulator-trained policy, not an original-ROM or
human/competitive strength result. Only one training seed has been used so far.

## Method

The shared candidate MLP has 128/64 hidden units and one value output. Its 67
features describe structural before/after boards, the single visible next piece,
level context and timed-action details. It does not read hidden RNG, a future
queue or hold, and is not initialized from the modern model.

Candidate programs wait three reaction frames, use held horizontal DAS and
four-frame rotation pulses, and navigate for at most 48 frames before Down.
The selected program is executed through Game.tick. Control state, post-clear
occupancy, lines and terminal state must match the preview. This is a bounded
legal action family, not an exhaustive search of every possible timing/tuck.

- 1,000 teacher decisions (5% exploratory) collect candidate-score targets.
- 5,000 supervised updates fit those heuristic scores.
- Double DQN then runs to 50,000 RL decisions: replay 5,000, batch 64, update
  every four decisions, target sync every 500, discount 0.99, learning rate
  0.0001, epsilon 0.15 to 0.03 over 20k decisions, two CPU threads.
- Episode starts sample levels 0/9/18/19 and warmup seeds below 10,000.
- Reward: 0.1 + lines squared - 10*death - 0.05*change in total column height
  - 0.3*change in holes - 0.02*change in bumpiness. This is explicit shaping,
  not a claim that it preserves the original game's optimal policy.

Teacher updates are separate from RL decisions; a fatal decision may not lock a
piece. Actual placements are logged separately. Full checkpoints retain online/
target/optimizer/replay/RNG/game state, with exact continuation tested. Safe
weights-only policies have a distinct NES kind and reject modern model files.

## Selection and timing

All warm and 5k-interval policies are retained, including regressions. Selection
uses six games: seeds 10101/13007 at starting levels 0/18/19, with fixed logical
three-frame reaction. CPU timings are recorded without changing those trajectories.
Mean observed lines is a selection statistic; cutoffs remain censored, not wins.

Final runtime games compare the selected model, imitation-only model, matched
reachable-action heuristic and random baseline on fresh declared starts. Late
plans are discarded and their full elapsed frames are charged as idle gameplay
before replanning. Gravity and RNG continue. Every input and lock hash replays.

Initial pilots used strict timing cutoffs. A charged-latency selection pilot then
showed that host jitter changes NES pieces and can affect snapshot selection.
Those results remain engineering evidence. Final selection separates deterministic
logical policy strength from charged-runtime performance. All earlier reserved
sets are retained as pilot data, not described as fresh final tests.

These are seeded starts, not guaranteed identical later piece sequences when
controllers spend different frame counts. Multiple starting levels from one
warmup seed are not independent training runs. ROM parity, broader training seeds,
and consistent level-19/29 performance remain unfinished project goals.

## Watch and resume

Use modern-rules/setup.ps1 for the full CPU learning environment. Minimal nes-bot/setup-nes.ps1 still
supports the untrained engine without PyTorch.

```powershell
.\nes-bot\play-nes.ps1 -Checkpoint nes-bot/models/nes-seed0-best.pt -StartLevel 18
.\.venv\Scripts\python.exe -m nes.learn --steps 20000 --seed 1 --output runs/nes-seed1
.\.venv\Scripts\python.exe -m nes.learn --resume runs/nes-training-v2/seed0/latest.pt --prior-policy nes-bot/models/nes-seed0-best.pt --steps 100000 --reserved-seeds 23029 29033 32603 --output runs/nes-continued
```

Steps are absolute RL-decision targets. Use new output folders and declare new
final starts before testing. The trainer's older periodic checks use strict
cutoffs; nes.benchmark_learning performs final fixed-reaction selection and
charged-runtime tests. Only load trusted full pickle-based training checkpoints.

In the portable backup, checkpoints/nes-seed0-50000.pt resumes the final training
state; checkpoints/nes-seed0-selected.pt resumes the selected earlier state.
nes-bot/evidence/nes-learning records all actual final cases and the selected snapshot.
Training loss and update counts alone do not establish a strength improvement.

## First results: no reliable RL gain yet

50k decisions were trained, but fixed-reaction validation selected the earlier
15k snapshot. The 25k..50k snapshots generally regressed; they remain preserved.

On final seeds 23017/29021/32501 at starts 0/18/19, all 36 runtime games ended
naturally and replayed exactly. Mean cleared lines were 55.33 for selected RL,
58.44 for the reachable-action heuristic, 72.67 for imitation-only, and 0 for
random. Two baseline games needed one late-plan delay each; the six idle frames
were charged and recorded in each. The selected RL games had no overruns.

From level 0, RL reached levels 16/4/9; imitation reached 19/7/19. These are only
three starts per condition. An earlier pilot RL game reached level 20 from an
18 start; the final starts did not reproduce that gain. The earlier result is
retained, not presented as consistent performance.

The default learned file is the validation-selected RL policy, not the last
training snapshot. The separately saved nes-seed0-imitation.pt is available for
comparison. Fresh outcomes do not retroactively change the selection protocol.
This pilot supports investigating stability and generalization before assuming
that a longer run or more hardware will improve the agent.
